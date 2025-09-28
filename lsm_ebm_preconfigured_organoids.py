#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU-safe end-to-end pipeline (single script) for:
- Downloading DANDI:001603 (draft) locally
- Pre-scanning NWB files (labels + data volume)
- Building windows from HD‑MEA recordings
- Training an LSM→EBM model (features computed on-the-fly; no giant tensors)
- Evaluating IMPUTATION and PREDICTION vs AR(2) and Linear baselines
- Saving per-window CSVs + paired Wilcoxon (Holm-corrected) significance tables
- Progress bars throughout

Run (examples):
  python organoid_lsm_ebm_cpu.py --download --data_dir data/001603 --out_dir results
  python organoid_lsm_ebm_cpu.py --data_dir data/001603 --out_dir results --n_channels 16 --fs_target 1000 --win_seconds 1.5 --stride_seconds 0.75

Notes:
- Forces CPU to avoid GPU/OOM variability.
- Uses streaming feature extraction and on-the-fly training batches (no precomputed H arrays).
- Requires ~8 NWB files from DANDI:001603 (3 min each, ~1,020 channels/file).
"""

import os, sys, json, math, glob, argparse, datetime, random, subprocess
import numpy as np

# ------------------- Robust imports / installs -------------------
def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

# Numpy
try:
    import numpy as np
except Exception:
    _pip("numpy>=1.24,<3"); import numpy as np  # CPU-safe

# Torch (CPU)
try:
    import torch, torch.nn as nn, torch.nn.functional as F
except Exception:
    _pip("torch>=2.1.0"); import torch, torch.nn as nn, torch.nn.functional as F

# SciPy
try:
    from scipy.signal import resample_poly, butter, filtfilt, hilbert
    from scipy.stats import wilcoxon
except Exception:
    _pip("scipy>=1.10.0"); from scipy.signal import resample_poly, butter, filtfilt, hilbert
    from scipy.stats import wilcoxon

# Pandas
try:
    import pandas as pd
except Exception:
    _pip("pandas>=1.5.0"); import pandas as pd

# tqdm
try:
    from tqdm.auto import tqdm
except Exception:
    _pip("tqdm"); from tqdm.auto import tqdm

# HDF5 / NWB
try:
    import h5py
except Exception:
    _pip("h5py>=3.8.0"); import h5py
try:
    from pynwb import NWBHDF5IO
    from pynwb.ecephys import ElectricalSeries
except Exception:
    _pip("pynwb>=2.6.0"); from pynwb import NWBHDF5IO
    from pynwb.ecephys import ElectricalSeries

# ------------------- Reproducibility & device -------------------
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cpu")  # force CPU

# ------------------- CLI -------------------
def build_cli():
    ap = argparse.ArgumentParser(description="Organoid LSM→EBM (CPU, streaming, DANDI:001603).")
    ap.add_argument("--download", action="store_true", help="Download DANDI:001603/draft into --data_dir (requires 'dandi' CLI).")
    ap.add_argument("--dandiset", type=str, default="001603", help="DANDI set to download (default: 001603).")
    ap.add_argument("--data_dir", type=str, default=os.path.join("data","001603"), help="Local data directory for NWB files.")
    ap.add_argument("--out_dir",  type=str, default="results", help="Directory for outputs.")
    ap.add_argument("--max_files", type=int, default=0, help="0 => use all NWB files; else cap.")
    ap.add_argument("--n_channels", type=int, default=16, help="Number of channels to read per file (memory/speed tradeoff).")
    ap.add_argument("--fs_target", type=float, default=1000.0, help="Target sampling rate (Hz).")
    ap.add_argument("--win_seconds", type=float, default=1.5, help="Window length (seconds).")
    ap.add_argument("--stride_seconds", type=float, default=0.75, help="Stride (seconds).")
    ap.add_argument("--max_seconds_per_file", type=float, default=180.0, help="Max seconds to read from each file.")
    ap.add_argument("--max_windows_per_file", type=int, default=1500, help="Cap windows per file (controls memory).")
    ap.add_argument("--train_max_windows", type=int, default=3000, help="Cap #train windows (controls memory).")
    ap.add_argument("--val_max_windows",   type=int, default=1000, help="Cap #val windows (controls memory).")
    ap.add_argument("--test_eval_max",     type=int, default=1000, help="Cap #test windows for metrics.")
    ap.add_argument("--epochs", type=int, default=8, help="Training epochs.")
    ap.add_argument("--batch_size", type=int, default=32, help="Training batch size (windows).")
    ap.add_argument("--gaps", type=int, nargs="+", default=[25,50,100,200], help="Gap lengths (samples at fs_target) for imputation.")
    ap.add_argument("--horizons", type=int, nargs="+", default=[50,100,250,500], help="Forecast horizons (samples at fs_target).")
    ap.add_argument("--seed", type=int, default=1337, help="Global random seed.")
    return ap

# ------------------- Download (DANDI CLI) -------------------
def ensure_dandi_cli():
    try:
        subprocess.run(["dandi", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception:
        _pip("dandi>=0.60.0")

def download_dandiset(dandiset_id: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    ensure_dandi_cli()
    cmd = ["dandi", "download", f"DANDI:{dandiset_id}/draft", "--output-dir", outdir, "--existing", "skip"]
    print("[download]"," ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("DANDI download failed. Make sure 'dandi' CLI is installed and you have internet access.")

# ------------------- Dataset pre-scan (labels + volume) -------------------
def list_nwb_files(data_dir, max_files=0, prefer_ecephys=True):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.nwb"), recursive=True))
    if prefer_ecephys:
        ef = [f for f in files if "ecephys" in os.path.basename(f).lower()]
        if ef: files = ef
    if max_files and max_files > 0:
        files = files[:max_files]
    return files

def _first_electrical_series(nwb):
    # Prefer acquisitions
    for name, obj in nwb.acquisition.items():
        if isinstance(obj, ElectricalSeries): return name, obj
    # Fallback in processing modules
    for pm in nwb.processing.values():
        for name, obj in pm.data_interfaces.items():
            if isinstance(obj, ElectricalSeries): return name, obj
    return None, None

def _safe_rate(es):
    if getattr(es, "rate", None) is not None:
        return float(es.rate)
    ts = getattr(es, "timestamps", None)
    if ts is not None:
        t = np.array(ts[:1000], dtype=np.float64)
        if t.size >= 2:
            dt = np.median(np.diff(t))
            if dt > 0:
                return float(1.0/dt)
    return None

def pre_scan(data_dir, fs_target=1000.0, win_seconds=1.5, stride_seconds=0.75, n_channels_est=16, max_files=0):
    files = list_nwb_files(data_dir, max_files=max_files, prefer_ecephys=True)
    if not files:
        raise FileNotFoundError(f"No NWB files under {data_dir}")

    rows=[]; bad=0
    for fp in files:
        try:
            with NWBHDF5IO(fp, "r", load_namespaces=True) as io:
                nwb = io.read()
                subj = getattr(nwb, "subject", None)
                subject_id  = getattr(subj, "subject_id", None) if subj else None
                session_id  = getattr(nwb, "session_id", None)
                sname, es   = _first_electrical_series(nwb)
                if es is None:
                    print(f"[warn] no ElectricalSeries in {os.path.basename(fp)}"); continue
                fs = _safe_rate(es)
                try:
                    shape = es.data.shape
                except Exception as e:
                    print(f"[warn] cannot read shape for {os.path.basename(fp)}: {e}")
                    bad+=1; continue
                if len(shape)!=2:
                    print(f"[warn] unexpected shape {shape} in {os.path.basename(fp)}"); bad+=1; continue
                samples_axis = 0 if shape[0] >= shape[1] else 1
                T_src = int(shape[samples_axis]); C_src = int(shape[1 - samples_axis])
                dur_s = (T_src / fs) if (fs and fs>0) else np.nan
                rows.append(dict(file=os.path.basename(fp), subject_id=subject_id, session_id=session_id,
                                 series_name=sname or getattr(es,"name",None), fs=fs, samples=T_src, channels=C_src, duration_s=dur_s))
        except Exception as e:
            print(f"[warn] skipping {os.path.basename(fp)}: {e}"); bad+=1

    if not rows:
        raise RuntimeError("No usable ElectricalSeries found.")

    # Labels
    subjects = sorted({r["subject_id"] for r in rows if r["subject_id"]})
    sessions = sorted({r["session_id"] for r in rows if r["session_id"]})
    series   = sorted({r["series_name"] for r in rows if r["series_name"]})
    # Durations
    dur_list = [r["duration_s"] for r in rows if isinstance(r["duration_s"], (int,float)) and np.isfinite(r["duration_s"])]
    tot_s = float(np.sum(dur_list)) if dur_list else 0.0
    # Channels
    ch_list = [r["channels"] for r in rows if isinstance(r["channels"], (int,float))]

    # Print concise summary
    print("\n=== Dataset pre-scan ===")
    print(f"Files found: {len(rows)} (warnings: {bad})")
    print(f"Subjects (labels): {', '.join(subjects) if subjects else 'unknown'}")
    print(f"Sessions (labels): {', '.join(sessions) if sessions else 'unknown'}")
    print(f"ElectricalSeries (labels): {', '.join(series) if series else 'unknown'}")
    print(f"Total duration: {tot_s/3600.0:.2f} h")
    if ch_list:
        print(f"Channels per file (min/median/max): {int(np.min(ch_list))}/{int(np.median(ch_list))}/{int(np.max(ch_list))}")

    # Window estimate
    W = int(round(win_seconds*fs_target)); S = int(round(stride_seconds*fs_target))
    est_total_windows = 0
    for r in rows:
        fs = r["fs"]; T = r["samples"]; C = r["channels"]
        if not fs or fs <= 0 or T <= 0: continue
        T_tgt = int(round(T * (fs_target / fs)))
        if T_tgt < W: continue
        n_wins_per_chan = 1 + (T_tgt - W)//S
        use_ch = min(n_channels_est, int(C))
        est_total_windows += int(use_ch * n_wins_per_chan)
    print(f"Planned preprocessing: fs={fs_target:.1f} Hz, window={win_seconds:.2f} s, stride={stride_seconds:.2f} s, channels_used={n_channels_est}")
    print(f"Estimated total windows: ~{est_total_windows}")
    return files

# ------------------- Signal IO & windowing -------------------
def extract_traces_from_nwb(nwb_path, n_channels=16, fs_target=1000.0, max_seconds=180.0):
    with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
        nwb = io.read()
        es = None
        for _, obj in nwb.acquisition.items():
            if isinstance(obj, ElectricalSeries): es = obj; break
        if es is None:
            for pm in nwb.processing.values():
                for _, obj in pm.data_interfaces.items():
                    if isinstance(obj, ElectricalSeries): es = obj; break
                if es is not None: break
        if es is None:
            raise RuntimeError(f"No ElectricalSeries in {nwb_path}")

        if getattr(es, "rate", None) is not None:
            fs_src = float(es.rate)
        else:
            ts = np.array(es.timestamps[:1000], dtype=np.float64)
            if ts.size < 2: raise RuntimeError("Cannot infer sampling rate.")
            fs_src = float(1.0/np.median(np.diff(ts)))

        data = es.data
        shape = data.shape
        if len(shape)!=2:
            raise RuntimeError(f"Unexpected data shape {shape} in {nwb_path}")
        samples_axis = 0 if shape[0] >= shape[1] else 1
        T_src = int(shape[samples_axis]); C_src = int(shape[1 - samples_axis])
        T_read = int(min(max_seconds*fs_src, T_src))
        C_read = int(min(n_channels, C_src))
        if samples_axis==0:
            X = np.array(data[:T_read, :C_read], dtype=np.float32)
        else:
            X = np.array(data[:C_read, :T_read], dtype=np.float32).T

    # Resample
    if abs(fs_src - fs_target) > 1e-6:
        from fractions import Fraction
        frac = Fraction(float(fs_target)/float(fs_src)).limit_denominator(1000)
        X = resample_poly(X, frac.numerator, frac.denominator, axis=0).astype(np.float32)

    # Low-pass to ~LFP band (<=300 Hz)
    nyq = 0.5*fs_target
    cutoff = min(300.0, nyq*0.95)
    if cutoff < nyq:
        b, a = butter(4, cutoff/nyq, btype='low')
        X = filtfilt(b, a, X, axis=0).astype(np.float32)
    return X, float(fs_target)

def make_windows(U, W, S):
    T = U.shape[0]
    starts = [s for s in range(0, max(1, T-W+1), S) if s+W <= T]
    return np.stack([U[s:s+W] for s in starts], axis=0) if starts else np.empty((0, W, U.shape[1]), np.float32)

def split_tr_va_te_by_file(wins_per_file, rtr=0.7, rva=0.15):
    files_nonempty = [w for w in wins_per_file if w is not None and w.size>0]
    if len(files_nonempty)==0:
        return (np.empty((0,0,0), np.float32),)*3
    if len(files_nonempty) >= 3:
        idx = np.arange(len(files_nonempty)); np.random.shuffle(idx)
        ntr = max(1, int(np.floor(rtr*len(files_nonempty))))
        nva = max(1, int(np.floor(rva*len(files_nonempty))))
        if ntr+nva > len(files_nonempty)-1:
            ntr = max(1, ntr-1)
        tr_idx = idx[:ntr]; va_idx = idx[ntr:ntr+nva]; te_idx = idx[ntr+nva:]
        def cat(ids):
            arrs = [files_nonempty[i] for i in ids if files_nonempty[i].size>0]
            return np.concatenate(arrs, axis=0) if arrs else np.empty((0,0,0), np.float32)
        U_tr = cat(tr_idx); U_va = cat(va_idx); U_te = cat(te_idx)
        if len(U_tr)==0 or len(U_va)==0 or len(U_te)==0:
            U_all = np.concatenate(files_nonempty, axis=0)
            return _split_windows_level(U_all, rtr, rva)
        return U_tr, U_va, U_te
    return _split_windows_level(files_nonempty[0], rtr, rva)

def _split_windows_level(U, rtr=0.7, rva=0.15):
    N=len(U)
    if N<3:
        return U, np.empty((0,)+U.shape[1:], U.dtype), np.empty((0,)+U.shape[1:], U.dtype)
    idx=np.arange(N); np.random.shuffle(idx)
    ntr=max(1,int(round(rtr*N))); nva=max(1,int(round(rva*N)))
    if ntr+nva>=N: ntr=max(1,ntr-1)
    tr=U[idx[:ntr]]; va=U[idx[ntr:ntr+nva]]; te=U[idx[ntr+nva:]]
    if len(va)==0: va=tr[-1:]; tr=tr[:-1]
    if len(te)==0: te=tr[-1:]; tr=tr[:-1]
    return tr, va, te

# ------------------- Normalization -------------------
def zfit(arr, eps=1e-3, floor=None):
    if arr.ndim==2:
        mu = arr.mean(0, keepdims=True).astype(np.float32)
        sd = arr.std(0,  keepdims=True).astype(np.float32)
    else:
        mu = arr.mean((0,1), keepdims=True).astype(np.float32)
        sd = arr.std((0,1),  keepdims=True).astype(np.float32)
    base = float(np.nanmedian(sd)) if np.isfinite(sd).all() else 1.0
    if floor is None: floor = max(eps, 0.05*base)
    sd = np.maximum(sd, floor).astype(np.float32)
    mu = np.nan_to_num(mu, nan=0., posinf=0., neginf=0.).astype(np.float32)
    return mu, sd

def zapply_t(x, mu, sd):  return torch.nan_to_num((x-mu)/sd, nan=0., posinf=0., neginf=0.)
def zinvert_t(x, mu, sd): return torch.nan_to_num(x*sd+mu,    nan=0., posinf=0., neginf=0.)

# ------------------- Baselines -------------------
def linear_interp(y, mask):
    out=y.copy(); T,D=y.shape; t=np.arange(T)
    for d in range(D):
        m=mask[:,d]
        if m.any():
            obs=~m
            if obs.sum()>=2:
                out[m,d]=np.interp(t[m], t[obs], y[obs,d])
    return out

def ar2_impute(y, mask, ridge=1e-3):
    T,D=y.shape; y0=linear_interp(y,mask); out=y0.copy()
    for d in range(D):
        yy=y0[:,d]
        if T<8: continue
        X=np.stack([yy[1:-1], yy[:-2]],1); tgt=yy[2:]
        A=X.T@X + ridge*np.eye(2); b=X.T@tgt; a=np.linalg.solve(A,b)
        idx=np.where(mask[:,d])[0]
        if idx.size==0: continue
        runs=np.split(idx, np.where(np.diff(idx)!=1)[0]+1)
        for run in runs:
            s,e=run[0], run[-1]
            if s-2<0: continue
            for t in range(s,e+1):
                out[t,d]=a[0]*out[t-1,d]+a[1]*out[t-2,d]
    return out

def ar2_forecast(y, k, ridge=1e-3):
    T,D=y.shape; out=[]
    for d in range(D):
        yy=y[:,d]
        if T<8:
            out.append(np.repeat(yy[-1], k).astype(np.float32)); continue
        X=np.stack([yy[1:-1], yy[:-2]],1); tgt=yy[2:]
        A=X.T@X + ridge*np.eye(2); b=X.T@tgt; a=np.linalg.solve(A,b)
        y1,y2=yy[-1],yy[-2]; preds=[]
        for _ in range(k):
            y_next=a[0]*y1 + a[1]*y2
            preds.append(float(y_next)); y2,y1=y1,y_next
        out.append(np.array(preds, dtype=np.float32))
    return np.stack(out,1)

def linear_extrap_forecast(y, k, fit_len=100):
    T,D=y.shape; t=np.arange(T); out=[]
    for d in range(D):
        yy=y[:,d]; L=min(fit_len,T)
        A=np.vstack([t[-L:], np.ones(L)]).T
        a,b=np.linalg.lstsq(A, yy[-L:], rcond=None)[0]
        t_fut=np.arange(T, T+k)
        out.append((a*t_fut + b).astype(np.float32))
    return np.stack(out,1)

# ------------------- Reservoir (LSM) -------------------
STATE_DIM = 256           # CPU-safe default
RES_CONN_PROB = 0.10
ADD_HILBERT = True
LC_LEVELS = (0.5, 1.0, 2.0, 4.0)
H_CLAMP, YZ_CLAMP, ECLAMP = 8.0, 8.0, 1e3

def build_reservoir_graph(N:int, in_dim:int, conn_prob:float, seed:int, input_scale:float=1.0):
    rng = np.random.RandomState(seed)
    mask = (rng.rand(N,N) < conn_prob).astype(np.float32)
    np.fill_diagonal(mask, 0.0)
    W_base = (rng.randn(N,N).astype(np.float32) / np.sqrt(max(1e-6, conn_prob*N))) * mask
    Win_base = (rng.randn(N,in_dim).astype(np.float32) * (input_scale/np.sqrt(in_dim)))
    return W_base, mask, Win_base

class LSM(nn.Module):
    def __init__(self, W_base:np.ndarray, Win_base_1d:np.ndarray, n_in:int,
                 tau_m:float=20.0, v_th:float=1.0, t_ref:int=2, tau_syn:float=5.0,
                 exc_frac:float=0.8, w_scale:float=0.7, seed:int=123):
        super().__init__()
        N=W_base.shape[0]
        n_exc=int(exc_frac*N)
        sign=np.ones(N, dtype=np.float32); sign[n_exc:]=-1.0
        W=np.abs(W_base).astype(np.float32); W=W * sign[:,None]
        W=W * (w_scale/max(1e-6, np.max(np.abs(W))))
        self.register_buffer('W_syn', torch.tensor(W, device=DEVICE, dtype=torch.float32))
        Win=np.tile(Win_base_1d, (1,n_in)).astype(np.float32)
        self.register_buffer('Win', torch.tensor(Win, device=DEVICE, dtype=torch.float32))
        self.tau_m=float(tau_m); self.v_th=float(v_th); self.t_ref=int(t_ref); self.tau_syn=float(tau_syn)
        self.v_reset=0.0; self.N=N
    def forward(self, S_in): # (B,T,M)
        B,T,M=S_in.shape; N=self.N
        v=torch.zeros(B,N, device=S_in.device)
        ref=torch.zeros(B,N, device=S_in.device, dtype=torch.int32)
        s_prev=torch.zeros(B,N, device=S_in.device)
        rate=torch.zeros(B,N, device=S_in.device)
        S_all=[]; R_all=[]; alpha=1.0/self.tau_syn
        for t in range(T):
            I_in  = F.linear(S_in[:,t,:], self.Win)
            I_syn = F.linear(s_prev, self.W_syn)
            dv = (-(v - 0.0) + I_in + I_syn)/self.tau_m
            v = v + dv
            can_spike = (ref<=0)
            s = (v >= self.v_th) & can_spike
            sf = s.float()
            v = torch.where(s, torch.full_like(v, self.v_reset), v)
            ref = torch.where(s, torch.full_like(ref, self.t_ref), ref)
            ref = torch.clamp(ref-1, min=0)
            rate = (1.0 - alpha)*rate + alpha*sf
            S_all.append(sf); R_all.append(rate); s_prev = sf
        return torch.stack(S_all,1), torch.stack(R_all,1)

def level_cross_encode_batch(U_bt: torch.Tensor, levels=LC_LEVELS):
    U = U_bt.detach().cpu().numpy()
    B,T,_ = U.shape; K=len(levels)
    S = np.zeros((B,T,2*K), dtype=np.float32)
    for b in range(B):
        x = U[b,:,0]; std = np.std(x) + 1e-6
        for k,mul in enumerate(levels):
            delta = float(mul*std); acc=0.0
            for t in range(1,T):
                acc += (x[t]-x[t-1])
                if acc >= delta: S[b,t,2*k]=1.0;  acc -= delta
                elif acc <= -delta: S[b,t,2*k+1]=1.0; acc += delta
    return torch.as_tensor(S, device=DEVICE, dtype=torch.float32)

def hilbert_feats_batch(U_batch_t: torch.Tensor):
    U = U_batch_t.detach().cpu().numpy()
    feats=[]
    for b in range(U.shape[0]):
        x = U[b,:,0].astype(np.float64)
        z = hilbert(x)
        amp = np.abs(z).astype(np.float32)
        ph  = np.angle(z).astype(np.float32)
        feats.append(np.stack([amp, np.cos(ph), np.sin(ph)], axis=1))
    F = np.stack(feats, axis=0).astype(np.float32)
    return torch.as_tensor(F, device=DEVICE, dtype=torch.float32)

class LSMFeatures:
    def __init__(self, lsm:LSM, mode="rates", add_hilbert=True):
        self.lsm=lsm; self.mode=mode; self.add_hilbert=add_hilbert
    def __call__(self, U_batch_t: torch.Tensor):
        S_in = level_cross_encode_batch(U_batch_t)
        with torch.no_grad():
            S, R = self.lsm(S_in)
        F_res = R if self.mode=="rates" else S
        if self.add_hilbert:
            F_h = hilbert_feats_batch(U_batch_t)
            mu = F_h.mean(dim=1, keepdim=True); sd = F_h.std(dim=1, keepdim=True) + 1e-3
            F_h = (F_h - mu)/sd
            F = torch.cat([F_res, F_h], dim=-1)
        else:
            F = F_res
        return F.detach()

# ------------------- EBM head -------------------
class TrueEBMHead(nn.Module):
    def __init__(self, in_dim:int, feat_dim:int, rank:int=4, gamma_state:float=None, kappa:float=0.3):
        super().__init__()
        if gamma_state is None: gamma_state = in_dim/float(feat_dim)
        self.gamma_state_val=float(gamma_state); self.kappa=float(kappa)
        self.in_dim=in_dim; self.feat_dim=feat_dim; self.rank=rank
        self.a_diag_raw=nn.Parameter(torch.zeros(feat_dim))
        if rank>0:
            self.U=nn.Parameter(torch.randn(feat_dim, rank)*0.01)
            self.V=nn.Parameter(torch.randn(feat_dim, rank)*0.01)
        else:
            self.register_parameter('U', None); self.register_parameter('V', None)
        self.B=nn.Parameter(torch.randn(feat_dim, in_dim)*0.01)
        self.C=nn.Parameter(torch.randn(in_dim, feat_dim)*0.01)
        self.log_sigma_y=nn.Parameter(torch.zeros(in_dim)); self.log_sigma_h=nn.Parameter(torch.zeros(feat_dim))
    def _D(self): return 0.99*torch.tanh(self.a_diag_raw)
    def _A(self):
        D=torch.diag(self._D())
        return D + self.U@self.V.t() if (self.rank>0 and self.U is not None) else D
    def step(self, h, u_prev):
        A=self._A()
        pre=F.linear(h,A)+F.linear(u_prev,self.B)
        h_next=(1-self.kappa)*h + self.kappa*pre
        y_pred=F.linear(h_next, self.C)
        h_next=torch.clamp(h_next, -H_CLAMP, H_CLAMP); y_pred=torch.clamp(y_pred, -YZ_CLAMP, YZ_CLAMP)
        return h_next, y_pred
    def energy(self, Hseq, Yseq, teacher=True):
        B,T,_=Yseq.shape; h=Hseq[:,0,:]; u_prev=Yseq[:,0,:]
        sy=torch.exp(self.log_sigma_y); sh=torch.exp(self.log_sigma_h)
        E=torch.zeros(B, device=Yseq.device)
        for t in range(1,T):
            h, y_pred=self.step(h, u_prev)
            Ey=((Yseq[:,t,:]-y_pred)/sy).pow(2).mean(1).mul_(0.5).clamp_(max=ECLAMP)
            Eh=((Hseq[:,t,:]-h)/sh).pow(2).mean(1).mul_(0.5).clamp_(max=ECLAMP)
            Ey=torch.nan_to_num(Ey, nan=ECLAMP); Eh=torch.nan_to_num(Eh, nan=ECLAMP)
            E=E + Ey + self.gamma_state_val*Eh
            u_prev=Yseq[:,t,:] if teacher else y_pred
        return torch.nan_to_num(E/max(1,(T-1)), nan=ECLAMP)

# ------------------- Streaming prefit (ridge, no giant arrays) -------------------
class OnlineMoments:
    """Welford per-feature mean/std over sequences (B,T,F) streamed in chunks."""
    def __init__(self, F_dim:int):
        self.n=0
        self.mean=np.zeros((1,1,F_dim), dtype=np.float64)
        self.M2  =np.zeros((1,1,F_dim), dtype=np.float64)
    def update(self, X_bt: np.ndarray):
        # X_bt: (B,T,F)
        B,T,F = X_bt.shape
        X = X_bt.reshape(-1, F).astype(np.float64)
        for i in range(X.shape[0]):
            self.n += 1
            delta = X[i:i+1] - self.mean.reshape(1,-1)
            self.mean += delta / self.n
            delta2 = X[i:i+1] - self.mean.reshape(1,-1)
            self.M2 += delta * delta2
    def finalize(self, eps=1e-3, floor_frac=0.05):
        if self.n < 2:
            mu = self.mean.astype(np.float32).reshape(1,1,-1)
            sd = np.ones_like(mu, dtype=np.float32)
            return mu, sd
        var = self.M2 / (self.n - 1 + 1e-9)
        sd = np.sqrt(np.maximum(var, eps**2)).astype(np.float32).reshape(1,1,-1)
        mu = self.mean.astype(np.float32).reshape(1,1,-1)
        base = float(np.nanmedian(sd))
        floor = max(eps, floor_frac*base)
        sd = np.maximum(sd, floor).astype(np.float32)
        mu = np.nan_to_num(mu, nan=0., posinf=0., neginf=0.).astype(np.float32)
        return mu, sd

def _ridge_stream_solve(XtX, XtY, lam):
    F = XtX.shape[0]
    A = XtX + lam*np.eye(F, dtype=np.float64)
    W = np.linalg.solve(A, XtY)
    return W

def prefit_streaming(features_fn, head_dim, U_tr, prefit_wins=200, lam_C=1e-1, lam_AB=1e-2, batch_size=32):
    """
    Streaming prefit for decoder C and dynamics (A,B).
    Returns: muY, sdY, muH, sdH, C (np.float32), A_target, B_target
    """
    # Subsample windows for prefit to keep it fast
    N = len(U_tr)
    if N == 0:
        raise RuntimeError("No training windows for prefit.")
    idx = np.random.permutation(N)[:min(prefit_wins, N)]
    # Pass 1: online moments for Y and H
    ym = OnlineMoments(F_dim=U_tr.shape[2])
    hm = OnlineMoments(F_dim=head_dim)
    for i0 in tqdm(range(0, len(idx), batch_size), desc="[prefit pass1: moments]"):
        j = idx[i0:i0+batch_size]
        Ub = torch.as_tensor(U_tr[j], device=DEVICE, dtype=torch.float32)
        Hb = features_fn(Ub).cpu().numpy()     # (B,T,Fh)
        Yb = Ub.cpu().numpy()                  # (B,T,1)
        ym.update(Yb); hm.update(Hb)
    muY, sdY = ym.finalize(); muH, sdH = hm.finalize()

    # Pass 2: accumulate ridge matrices
    # For C: H -> Y (flatten time)
    Fh = head_dim; Dy = U_tr.shape[2]
    XtX_C = np.zeros((Fh, Fh), dtype=np.float64)
    XtY_C = np.zeros((Fh, Dy),  dtype=np.float64)

    # For AB: [h_prev, y_prev] -> h_next
    Zdim = Fh + Dy
    ZtZ = np.zeros((Zdim, Zdim), dtype=np.float64)
    ZtH = np.zeros((Zdim, Fh),   dtype=np.float64)

    for i0 in tqdm(range(0, len(idx), batch_size), desc="[prefit pass2: ridge]"):
        j = idx[i0:i0+batch_size]
        Ub = torch.as_tensor(U_tr[j], device=DEVICE, dtype=torch.float32)
        Hb = features_fn(Ub)  # (B,T,Fh)
        # normalize
        Yz = zapply_t(Ub, torch.as_tensor(muY), torch.as_tensor(sdY))
        Hz = zapply_t(Hb, torch.as_tensor(muH), torch.as_tensor(sdH))
        # C accumulation
        H2 = Hz.reshape(-1, Fh).cpu().numpy().astype(np.float64)
        Y2 = Yz.reshape(-1, Dy).cpu().numpy().astype(np.float64)
        XtX_C += H2.T @ H2
        XtY_C += H2.T @ Y2
        # AB accumulation
        Hprev = Hz[:, :-1, :].reshape(-1, Fh).cpu().numpy()
        Hnext = Hz[:,  1:, :].reshape(-1, Fh).cpu().numpy()
        Yprev = Yz[:, :-1, :].reshape(-1, Dy).cpu().numpy()
        Z = np.concatenate([Hprev, Yprev], axis=1)  # (N*T-1, Fh+Dy)
        ZtZ += Z.T @ Z
        ZtH += Z.T @ Hnext

    # Solve ridges
    C = _ridge_stream_solve(XtX_C, XtY_C, lam_C).T.astype(np.float32)       # (Dy, Fh)
    Wab = _ridge_stream_solve(ZtZ, ZtH, lam_AB).astype(np.float32)          # (Fh+Dy, Fh)
    A_target = Wab[:Fh, :].T                                                # (Fh, Fh)
    B_target = Wab[Fh:, :].T                                                # (Fh, Dy)
    return (muY, sdY, muH, sdH, C, A_target, B_target)

# ------------------- Training (on-the-fly features) -------------------
def ebm_reg(head,wA=1e-4,wUV=1e-4,w_sy=5e-4,w_sh=5e-5):
    reg=wA*(head.a_diag_raw**2).mean()
    if head.U is not None: reg += wUV*((head.U**2).mean()+(head.V**2).mean())
    reg += w_sy*(head.log_sigma_y**2).mean() + w_sh*(head.log_sigma_h**2).mean()
    return reg

def contrastive_loss(head,H,Y,neg_K=2,margin=2.0,teacher=True):
    def make_negatives(Y, K=2, noise_std=0.2, max_shift=None):
        B,T,D=Y.shape; outs=[]
        if max_shift is None: max_shift=max(2,int(T//6))
        for _ in range(K):
            y=Y.clone(); s=np.random.randint(-max_shift, max_shift+1)
            if s!=0: y=torch.roll(y, shifts=s, dims=1)
            if noise_std>0: y=y+noise_std*torch.randn_like(y)
            outs.append(y)
        return torch.cat(outs,0)
    E_pos=head.energy(H,Y,teacher=teacher); Yneg=make_negatives(Y,neg_K); Hrep=H.repeat_interleave(neg_K,0)
    E_neg=head.energy(Hrep,Yneg,teacher=teacher).view(neg_K,-1).transpose(0,1)
    return torch.nan_to_num(E_pos.mean() + F.relu(margin + E_pos.unsqueeze(1) - E_neg).mean(), nan=1e6)

def head_supervised_losses(head,H,Y,tf_k=3):
    Y_hat=F.linear(H, head.C).clamp(-YZ_CLAMP, YZ_CLAMP); L_dec=0.5*((Y_hat-Y)**2).mean()
    B,T,D=Y.shape; h=H[:,0,:]; u_prev=Y[:,0,:]; errs=[]
    for t in range(1,T):
        h,y_pred=head.step(h,u_prev); errs.append(((y_pred-Y[:,t,:])**2).mean())
        h=0.9*h + 0.1*H[:,t,:]; u_prev=Y[:,t,:]
    L_tf=torch.stack(errs).mean() if errs else torch.tensor(0., device=Y.device, dtype=Y.dtype)
    return torch.nan_to_num(L_dec, nan=1e6), torch.nan_to_num(L_tf, nan=1e6)

def train_head_streaming(features_fn, U_tr, U_va, epochs=8, batch_size=32, prefit_wins=200):
    # Build feature dim by a quick probe
    with torch.no_grad():
        Hb_probe = features_fn(torch.as_tensor(U_tr[:1], device=DEVICE, dtype=torch.float32))
        feat_dim = int(Hb_probe.shape[-1])
    # Prefit (streaming)
    muY, sdY, muH, sdH, C_init, A_t, B_t = prefit_streaming(features_fn, feat_dim, U_tr, prefit_wins=prefit_wins, batch_size=batch_size)

    # Head
    Dy = U_tr.shape[2]
    head = TrueEBMHead(in_dim=Dy, feat_dim=feat_dim, rank=4, gamma_state=None, kappa=0.3).to(DEVICE)
    with torch.no_grad():
        head.C.copy_(torch.from_numpy(C_init).to(DEVICE))
        # init A via diag + low-rank from A_t
        diagA = np.clip(np.diag(A_t), -0.98, 0.98)/0.99
        head.a_diag_raw.copy_(torch.from_numpy(np.arctanh(diagA).astype(np.float32)).to(DEVICE))
        # low-rank from residual
        R = A_t - np.diag(np.diag(A_t))
        try:
            U_s, S_s, Vh_s = np.linalg.svd(R, full_matrices=False)
            r = min(4, len(S_s))
            U_lr = (U_s[:,:r]*np.sqrt(np.maximum(S_s[:r],0.)+1e-8)).astype(np.float32)
            V_lr = (Vh_s[:r,:].T*np.sqrt(np.maximum(S_s[:r],0.)+1e-8)).astype(np.float32)
            head.U.copy_(torch.from_numpy(U_lr).to(DEVICE))
            head.V.copy_(torch.from_numpy(V_lr).to(DEVICE))
        except Exception:
            pass
        head.B.copy_(torch.from_numpy(B_t).to(DEVICE))
        head.log_sigma_y.fill_(0.); head.log_sigma_h.fill_(0.)

    opt = torch.optim.Adam(head.parameters(), lr=3e-4, weight_decay=1e-5)
    best={"val":float("inf"), "state":None}
    muY_t=torch.as_tensor(muY, device=DEVICE); sdY_t=torch.as_tensor(sdY, device=DEVICE)
    muH_t=torch.as_tensor(muH, device=DEVICE); sdH_t=torch.as_tensor(sdH, device=DEVICE)

    def batches_idx(N, bs):
        idx=np.arange(N); np.random.shuffle(idx)
        for i in range(0,N,bs): yield idx[i:i+bs]

    for ep in range(1, epochs+1):
        head.train()
        trN = len(U_tr)
        for j in tqdm(batches_idx(trN, batch_size), total=math.ceil(trN/batch_size), desc=f"[train ep{ep:02d}]"):
            Ub = torch.as_tensor(U_tr[j], device=DEVICE, dtype=torch.float32)
            Hb = features_fn(Ub)
            Yz = zapply_t(Ub, muY_t, sdY_t)
            Hz = zapply_t(Hb, muH_t, sdH_t)
            loss = contrastive_loss(head, Hz, Yz, neg_K=2, margin=2.0, teacher=True)
            L_dec, L_tf = head_supervised_losses(head, Hz, Yz, tf_k=3)
            loss = loss + 0.4*L_dec + 0.4*L_tf + ebm_reg(head)
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 0.25); opt.step()

        head.eval()
        with torch.no_grad():
            # sample up to val_max for quick val
            valN = len(U_va)
            vs=[]
            for j in batches_idx(valN, batch_size):
                Ub = torch.as_tensor(U_va[j], device=DEVICE, dtype=torch.float32)
                Hb = features_fn(Ub)
                Yz = zapply_t(Ub, muY_t, sdY_t)
                Hz = zapply_t(Hb, muH_t, sdH_t)
                v=contrastive_loss(head, Hz, Yz, 2, 2.0, teacher=True)
                L_dec, L_tf = head_supervised_losses(head, Hz, Yz, tf_k=3)
                vs.append(float((v + 0.4*L_dec + 0.4*L_tf + ebm_reg(head)).cpu()))
            vmean = float(np.mean(vs)) if vs else float("inf")
        print(f"[ep {ep:02d}] val {vmean:.4f}")
        if vmean < best["val"]:
            best["val"] = vmean
            best["state"] = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}

    if best["state"] is not None: head.load_state_dict(best["state"])
    norms = {"muY":muY, "sdY":sdY, "muH":muH, "sdH":sdH}
    return head, norms

# ------------------- Imputation & Prediction helpers -------------------
class RolloutCfg: 
    def __init__(self, beta_min=0.2, beta_max=0.4): self.beta_min=beta_min; self.beta_max=beta_max

def _mask_random(window: np.ndarray, gap_len: int, win_id: int):
    W = window.shape[0]
    rng = np.random.RandomState(17 + 1000*win_id + 10*gap_len)
    s = int(rng.randint(2, max(3, W-gap_len-2)))
    mask = np.zeros_like(window, dtype=bool); mask[s:s+gap_len,:] = True
    return window.copy(), mask, s

def impute_bidir(head, features_fn, U_win, mask, norms, rcfg:RolloutCfg):
    # seed via AR(2)
    U_seed = ar2_impute(U_win, mask)
    with torch.no_grad():
        H_f   = features_fn(torch.as_tensor(U_seed[None,...], device=DEVICE, dtype=torch.float32))
        U_rev = np.ascontiguousarray(U_seed[::-1])
        H_b_r = features_fn(torch.as_tensor(U_rev[None,...], device=DEVICE, dtype=torch.float32))
    muY=torch.as_tensor(norms["muY"], device=DEVICE); sdY=torch.as_tensor(norms["sdY"], device=DEVICE)
    muH=torch.as_tensor(norms["muH"], device=DEVICE); sdH=torch.as_tensor(norms["sdH"], device=DEVICE)
    Hn_f = zapply_t(H_f, muH, sdH); Hn_b = torch.flip(zapply_t(H_b_r, muH, sdH), dims=[1])
    Yz   = zapply_t(torch.as_tensor(U_win[None,...], device=DEVICE, dtype=torch.float32), muY, sdY)
    idx=np.where(mask[:,0])[0]
    if idx.size==0: return U_win.copy(), (0,-1)
    s,e=int(idx[0]), int(idx[-1])
    beta_vec = np.linspace(rcfg.beta_min, rcfg.beta_max, num=(e-s+1)).astype(np.float32) if e>s else np.array([rcfg.beta_max], dtype=np.float32)

    with torch.no_grad():
        # forward rollout
        A=head._A(); Y_f=Yz.clone()
        h = Hn_f[:,s-1,:] if s>0 else Hn_f[:,s,:]
        u_prev = Yz[:,s-1,:] if s>0 else F.linear(Hn_f[:,s,:], head.C)
        for i,t in enumerate(range(s,e+1)):
            pre=F.linear(h,A)+F.linear(u_prev,head.B); h=(1-head.kappa)*h + head.kappa*pre
            b=float(beta_vec[i]); h=(1.0-b)*h + b*Hn_f[:,t,:]
            y_pred=F.linear(h, head.C).clamp(-YZ_CLAMP, YZ_CLAMP)
            Y_f[:,t,:]=y_pred; u_prev=y_pred
        # backward rollout
        T=Yz.shape[1]; s_r,e_r=T-1-e, T-1-s
        Y_b_r=Yz.flip(1).clone()
        h = Hn_b[:,s_r-1,:] if s_r>0 else Hn_b[:,s_r,:]
        u_prev = Yz.flip(1)[:,s_r-1,:] if s_r>0 else F.linear(Hn_b[:,s_r,:], head.C)
        for i,t in enumerate(range(s_r,e_r+1)):
            pre=F.linear(h,A)+F.linear(u_prev,head.B); h=(1-head.kappa)*h + head.kappa*pre
            b=float(beta_vec[i]); h=(1.0-b)*h + b*Hn_b[:,t,:]
            y_pred=F.linear(h, head.C).clamp(-YZ_CLAMP, YZ_CLAMP)
            Y_b_r[:,t,:]=y_pred; u_prev=y_pred
        Y_b = Y_b_r.flip(1)
        # blend
        w = torch.linspace(0,1,steps=(e - s + 1), device=DEVICE).view(1,-1,1) if e>s else torch.ones(1,1,1, device=DEVICE)*0.5
        Y_blend=Yz.clone(); Y_blend[:,s:e+1,:]=(1-w)*Y_f[:,s:e+1,:] + w*Y_b[:,s:e+1,:]
        Y_imp = zinvert_t(Y_blend, muY, sdY)[0].detach().cpu().numpy()
    return Y_imp, (s,e)

def predict_k_steps(head, features_fn, U_ctx, norms, k):
    with torch.no_grad():
        H = features_fn(torch.as_tensor(U_ctx[None,...], device=DEVICE, dtype=torch.float32))
    muY=torch.as_tensor(norms["muY"], device=DEVICE); sdY=torch.as_tensor(norms["sdY"], device=DEVICE)
    muH=torch.as_tensor(norms["muH"], device=DEVICE); sdH=torch.as_tensor(norms["sdH"], device=DEVICE)
    Hn = zapply_t(H, muH, sdH)
    Yz = zapply_t(torch.as_tensor(U_ctx[None,...], device=DEVICE, dtype=torch.float32), muY, sdY)
    with torch.no_grad():
        h = Hn[:,0,:]; u_prev=Yz[:,0,:]
        for t in range(1, Yz.shape[1]):
            h,_ = head.step(h, u_prev); h=0.9*h + 0.1*Hn[:,t,:]; u_prev=Yz[:,t,:]
        preds=[]
        for _ in range(k):
            h,y_pred = head.step(h, u_prev)
            preds.append(y_pred.clamp(-YZ_CLAMP, YZ_CLAMP)); u_prev=y_pred
        Yp = torch.stack(preds,1)
        Y_hat = zinvert_t(Yp, muY, sdY)[0].detach().cpu().numpy()
    return Y_hat

# ------------------- Metrics & stats -------------------
def metrics_on_mask(y_true, y_hat, mask):
    m=mask.astype(bool)
    if not m.any(): return {"mae":np.nan,"rmse":np.nan}
    diff=y_hat[m]-y_true[m]
    return {"mae":float(np.mean(np.abs(diff))), "rmse":float(np.sqrt(np.mean(diff**2)))}

def prediction_metrics(y_true_future, y_hat_future):
    diff=y_hat_future - y_true_future
    return {"mae":float(np.mean(np.abs(diff))), "rmse":float(np.sqrt(np.mean(diff**2)))}

def holm_bonferroni(pvals):
    order = np.argsort(pvals); m=len(pvals); adj=np.ones_like(pvals, dtype=np.float64)
    for rank,k in enumerate(order, start=1):
        adj[k] = min(1.0, pvals[k]*(m - rank + 1))
    return adj, order

def paired_table_and_significance(df, task_name, cols=('MAE_EBM','MAE_AR2','MAE_Lin')):
    rows=[]; key_field = 'gap' if task_name=='imputation' else 'horizon'
    if len(df)==0: return pd.DataFrame(rows)
    for key in sorted(df[key_field].unique()):
        sub = df[df[key_field]==key].dropna(subset=list(cols))
        piv = sub.pivot_table(index='window_id', values=list(cols), aggfunc='first').dropna()
        if len(piv)==0: continue
        try:
            _, p1 = wilcoxon(piv['MAE_EBM'].values, piv['MAE_AR2'].values, alternative='two-sided', zero_method='wilcox', mode='auto')
        except Exception:
            p1 = np.nan
        try:
            _, p2 = wilcoxon(piv['MAE_EBM'].values, piv['MAE_Lin'].values, alternative='two-sided', zero_method='wilcox', mode='auto')
        except Exception:
            p2 = np.nan
        rows.append({"task":task_name, key_field:int(key), "N":int(len(piv)),
                     "p_raw_EBM_vs_AR2":float(p1) if np.isfinite(p1) else np.nan,
                     "p_raw_EBM_vs_Lin":float(p2) if np.isfinite(p2) else np.nan})
    out = pd.DataFrame(rows)
    if len(out)>0:
        pvals=[]
        for _,r in out.iterrows():
            if np.isfinite(r["p_raw_EBM_vs_AR2"]): pvals.append(r["p_raw_EBM_vs_AR2"])
            if np.isfinite(r["p_raw_EBM_vs_Lin"]): pvals.append(r["p_raw_EBM_vs_Lin"])
        pvals = np.array(pvals, dtype=np.float64) if len(pvals)>0 else np.array([1.0])
        adj,_ = holm_bonferroni(pvals); k=0
        for i in range(len(out)):
            if np.isfinite(out.loc[i,"p_raw_EBM_vs_AR2"]): out.loc[i,"p_holm_EBM_vs_AR2"]=float(adj[k]); k+=1
            else: out.loc[i,"p_holm_EBM_vs_AR2"]=np.nan
            if np.isfinite(out.loc[i,"p_raw_EBM_vs_Lin"]): out.loc[i,"p_holm_EBM_vs_Lin"]=float(adj[k]); k+=1
            else: out.loc[i,"p_holm_EBM_vs_Lin"]=np.nan
    return out

# ------------------- Main -------------------
def main():
    args = build_cli().parse_args()
    set_seed(args.seed)

    # Prepare dirs
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.out_dir, ts); os.makedirs(outdir, exist_ok=True)
    def savejson(obj, name): 
        p=os.path.join(outdir, name); 
        with open(p, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2)
        print("[json]", p)

    # Download if requested
    if args.download:
        download_dandiset(args.dandiset, args.data_dir)

    print("DATA root:", args.data_dir)
    files = pre_scan(args.data_dir, fs_target=args.fs_target, win_seconds=args.win_seconds,
                     stride_seconds=args.stride_seconds, n_channels_est=args.n_channels, max_files=args.max_files)

    # Load signals -> per-channel windows
    W = int(round(args.win_seconds*args.fs_target))
    S = int(round(args.stride_seconds*args.fs_target))
    wins_per_file=[]; counts=[]
    for fp in files:
        try:
            X, fs = extract_traces_from_nwb(fp, n_channels=args.n_channels, fs_target=args.fs_target, max_seconds=args.max_seconds_per_file)
            mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-6
            Xn = (X - mu)/sd
            # stack per-channel windows along batch
            per_chan = [make_windows(Xn[:,c:c+1], W=W, S=S) for c in range(min(args.n_channels, Xn.shape[1]))]
            Wcat = np.concatenate(per_chan, axis=0) if per_chan else np.empty((0,W,1), np.float32)
            if len(Wcat) > args.max_windows_per_file:
                idx = np.random.permutation(len(Wcat))[:args.max_windows_per_file]
                Wcat = Wcat[idx]
            wins_per_file.append(Wcat.astype(np.float32)); counts.append(len(Wcat))
            print(f"Loaded {os.path.basename(fp)} -> windows {Wcat.shape}")
        except Exception as e:
            print(f"[warn] Skipping {os.path.basename(fp)}: {e}")
    if len(wins_per_file)==0 or all(w.size==0 for w in wins_per_file):
        raise RuntimeError("No windows generated. Adjust --n_channels/--win_seconds/--stride_seconds.")

    # Split
    U_tr, U_va, U_te = split_tr_va_te_by_file(wins_per_file, 0.7, 0.15)
    # subsample for memory control
    if args.train_max_windows and len(U_tr) > args.train_max_windows:
        idx = np.random.permutation(len(U_tr))[:args.train_max_windows]; U_tr = U_tr[idx]
    if args.val_max_windows and len(U_va) > args.val_max_windows:
        idx = np.random.permutation(len(U_va))[:args.val_max_windows]; U_va = U_va[idx]
    print(f"Split windows | train: {len(U_tr)} | val: {len(U_va)} | test: {len(U_te)}")

    # Build LSM features
    W_base_np, _, Win_base_np = build_reservoir_graph(STATE_DIM, in_dim=1, conn_prob=RES_CONN_PROB, seed=args.seed, input_scale=1.0)
    lsm = LSM(W_base=W_base_np, Win_base_1d=Win_base_np, n_in=2*len(LC_LEVELS),
              tau_m=20.0, v_th=1.0, t_ref=2, tau_syn=5.0, seed=args.seed+11).to(DEVICE).eval()
    features_fn = LSMFeatures(lsm, mode="rates", add_hilbert=ADD_HILBERT)

    # Train head (streaming)
    head, norms = train_head_streaming(features_fn, U_tr, U_va, epochs=args.epochs, batch_size=args.batch_size, prefit_wins=min(200, len(U_tr)))

    # --------- IMPUTATION ---------
    recs=[]; rcfg=RolloutCfg(0.2,0.4); WN=min(len(U_te), args.test_eval_max)
    for L in args.gaps:
        pbar=tqdm(range(WN), desc=f"[impute gap={L} samp (~{L/args.fs_target*1000:.0f} ms)]")
        for i in pbar:
            y = U_te[i]
            _, mask, start = _mask_random(y, L, i)
            y_lin = linear_interp(y, mask)
            y_ar2 = ar2_impute(y, mask)
            y_ebm, (s,e) = impute_bidir(head, features_fn, y, mask, norms, rcfg)
            mE = metrics_on_mask(y, y_ebm, mask); mA = metrics_on_mask(y, y_ar2, mask); mL = metrics_on_mask(y, y_lin, mask)
            recs.append({"task":"imputation","gap":L,"window_id":i,"start":int(start),
                         "MAE_EBM":mE["mae"],"RMSE_EBM":mE["rmse"],
                         "MAE_AR2":mA["mae"],"RMSE_AR2":mA["rmse"],
                         "MAE_Lin":mL["mae"],"RMSE_Lin":mL["rmse"]})
    df_imp = pd.DataFrame(recs)
    path = os.path.join(outdir, "IMPUTATION_perwindow.csv"); df_imp.to_csv(path, index=False); print("[saved]", path)
    sig_imp = paired_table_and_significance(df_imp, "imputation", cols=('MAE_EBM','MAE_AR2','MAE_Lin'))
    path = os.path.join(outdir, "IMPUTATION_significance.csv"); sig_imp.to_csv(path, index=False); print("[saved]", path)

    # --------- PREDICTION ---------
    recs=[]; WN=min(len(U_te), args.test_eval_max)
    for H in args.horizons:
        pbar=tqdm(range(WN), desc=f"[predict horizon={H} samp (~{H/args.fs_target*1000:.0f} ms)]")
        for i in pbar:
            y = U_te[i]
            k = min(H, max(1, y.shape[0]//4))
            ctx, fut = y[:-k,:], y[-k:,:]
            y_hat_ebm = predict_k_steps(head, features_fn, ctx, norms, k)
            y_hat_ar2 = ar2_forecast(ctx, k)
            y_hat_lin = linear_extrap_forecast(ctx, k)
            mE = prediction_metrics(fut, y_hat_ebm); mA = prediction_metrics(fut, y_hat_ar2); mL = prediction_metrics(fut, y_hat_lin)
            recs.append({"task":"prediction","horizon":H,"k_used":k,"window_id":i,
                         "MAE_EBM":mE["mae"],"RMSE_EBM":mE["rmse"],
                         "MAE_AR2":mA["mae"],"RMSE_AR2":mA["rmse"],
                         "MAE_Lin":mL["mae"],"RMSE_Lin":mL["rmse"]})
    df_pred = pd.DataFrame(recs)
    path = os.path.join(outdir, "PREDICTION_perwindow.csv"); df_pred.to_csv(path, index=False); print("[saved]", path)
    sig_pred = paired_table_and_significance(df_pred, "prediction", cols=('MAE_EBM','MAE_AR2','MAE_Lin'))
    path = os.path.join(outdir, "PREDICTION_significance.csv"); sig_pred.to_csv(path, index=False); print("[saved]", path)

    # Winners summary
    winners={}
    if len(df_imp)>0:
        agg=df_imp.groupby(['gap']).agg(MAE_EBM=('MAE_EBM','mean'), MAE_AR2=('MAE_AR2','mean'), MAE_Lin=('MAE_Lin','mean')).reset_index()
        winners['imputation']=[{"gap":int(r['gap']), "best":min([('EBM',r['MAE_EBM']),('AR2',r['MAE_AR2']),('Linear',r['MAE_Lin'])], key=lambda x:x[1])[0]} for _,r in agg.iterrows()]
    if len(df_pred)>0:
        agg=df_pred.groupby(['horizon']).agg(MAE_EBM=('MAE_EBM','mean'), MAE_AR2=('MAE_AR2','mean'), MAE_Lin=('MAE_Lin','mean')).reset_index()
        winners['prediction']=[{"horizon":int(r['horizon']), "best":min([('EBM',r['MAE_EBM']),('AR2',r['MAE_AR2']),('Linear',r['MAE_Lin'])], key=lambda x:x[1])[0]} for _,r in agg.iterrows()]
    savejson(winners, "WINNERS.json")

    print("\nDone. Results saved to:", outdir)

if __name__ == "__main__":
    main()
