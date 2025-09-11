# =================================== ESN+EBM with spike-centered ECG + OVERLAY PLOTS (LOCAL) ==========================
# - No Google Drive/Colab. Saves everything under ./results/<timestamp>/
# - Robust WFDB loader for NumPy 2.x (no streaming; local cache) + compatibility shim
# - Generates overlay figures with uncertainty band; saves per-window CSVs, summary and significance tables
# ======================================================================================================================

# ------------------- Config -------------------
ECG_DB                   = "nsrdb"           # 'nsrdb' or 'mitdb'
FORCE_ECG_GAP_ON_SPIKE   = True

RUN_CHIRP                = True
CHIRP_K                  = 0.2

STATE_DIM                = 256
ADD_HILBERT              = True
EPOCHS                   = 8                  # lower (e.g., 6) to speed up CPU runs
TEST_EVAL_MAX            = 300                # cap #test windows per dataset (lower for speed)

# Overlay controls
SAVE_OVERLAYS            = True
OVERLAYS_PER_GAP         = 12
SAVE_ENERGY_PROFILES     = False              # set True to also save α-profile plots (slower)
PROFILE_POINTS           = 0                  # leave 0 (band from F/B spread); set >1 to sweep energy vs α

SEED_GLOBAL              = 1337

# ------------------- Robust imports/installs -------------------
import sys, subprocess, os, math, json, random, datetime

def _pip(*args): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

# Use a headless backend for local runs
import matplotlib
matplotlib.use("Agg")

try:
    import numpy as np
except Exception:
    _pip("numpy"); import numpy as np

# --- NumPy 2.x shim: make np.fromstring(bytes) behave like frombuffer (binary mode removed in NumPy>=2.0) ---
try:
    _np_major = int(np.__version__.split(".")[0])
except Exception:
    _np_major = 2
if _np_major >= 2:
    _orig_fromstring = np.fromstring
    def _fromstring_compat(s, dtype=float, count=-1, sep=''):
        if sep == '' and isinstance(s, (bytes, bytearray, memoryview)):
            return np.frombuffer(s, dtype=dtype, count=count)
        return _orig_fromstring(s, dtype=dtype, count=count, sep=sep)
    np.fromstring = _fromstring_compat  # monkey-patch

try:
    import torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
except Exception:
    _pip("torch", "matplotlib"); import torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt

try:
    import wfdb
except Exception:
    _pip("wfdb>=4.1.3"); import wfdb

try:
    from scipy.signal import resample_poly, hilbert, find_peaks
except Exception:
    _pip("scipy>=1.9.0"); from scipy.signal import resample_poly, hilbert, find_peaks

try:
    from scipy.stats import wilcoxon
except Exception:
    _pip("scipy>=1.9.0"); from scipy.stats import wilcoxon

try:
    from sklearn.decomposition import PCA
except Exception:
    _pip("scikit-learn>=1.2.0"); from sklearn.decomposition import PCA

try:
    from tqdm.auto import tqdm
except Exception:
    _pip("tqdm"); from tqdm.auto import tqdm

try:
    import pandas as pd
except Exception:
    _pip("pandas>=1.5.0"); import pandas as pd

from dataclasses import dataclass
from typing import Callable

# ------------------- Device, seeds, IO -------------------
def pick_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = pick_device()
torch.set_num_threads(max(1, os.cpu_count() or 1))

def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(SEED_GLOBAL)

TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_OUT = "results"
OUTDIR = os.path.join(BASE_OUT, TS); os.makedirs(OUTDIR, exist_ok=True)

def savefig(fig, name, dpi=150):
    path = os.path.join(OUTDIR, name); fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig); print(f"[fig] {path}")

def savejson(obj, name):
    path = os.path.join(OUTDIR, name); open(path, "w").write(json.dumps(obj, indent=2)); print(f"[json] {path}")

print("Device:", DEVICE, "| OUTDIR:", OUTDIR)

# ------------------- Utility: z-norm -------------------
def zfit(arr, eps=1e-3, floor=None):
    if arr.ndim==2:
        mu=arr.mean(0, keepdims=True).astype(np.float32); sd=arr.std(0, keepdims=True).astype(np.float32)
    else:
        mu=arr.mean((0,1), keepdims=True).astype(np.float32); sd=arr.std((0,1), keepdims=True).astype(np.float32)
    base=float(np.nanmedian(sd)) if np.isfinite(sd).all() else 1.0
    if floor is None: floor=max(eps, 0.05*base)
    sd=np.maximum(sd, floor).astype(np.float32); mu=np.nan_to_num(mu, nan=0., posinf=0., neginf=0.).astype(np.float32)
    return mu, sd
def zapply(arr, mu, sd): return np.nan_to_num((arr-mu)/sd, nan=0., posinf=0., neginf=0.).astype(np.float32)
def zinvert(arr, mu, sd): return np.nan_to_num(arr*sd+mu, nan=0., posinf=0., neginf=0.).astype(np.float32)
def zapply_t(x, mu, sd):  return torch.nan_to_num((x-mu)/sd, nan=0., posinf=0., neginf=0.)
def zinvert_t(x, mu, sd): return torch.nan_to_num(x*sd+mu,    nan=0., posinf=0., neginf=0.)

# ------------------- Data -------------------
def _resample_to_100hz(X, fs_src):
    up, down = 100, int(round(fs_src)); from math import gcd
    g = gcd(up, down); up//=g; down//=g
    return resample_poly(X, up, down, axis=0).astype(np.float32)

def _load_wfdb_series_local_first(db="mitdb", rec="100"):
    """
    Robust loader for NumPy 2.x + WFDB:
    - first try to download to a local cache, then read from disk (avoids streaming path that uses np.fromstring)
    - fall back to remote rdsamp only if local read fails
    """
    base = os.path.join("wfdb_cache", db)
    os.makedirs(base, exist_ok=True)
    try:
        wfdb.dl_record(rec, pn_dir=db, dl_dir=base, keep_subdirs=True, overwrite=False)
    except Exception:
        pass
    # try a few plausible roots
    for root in (os.path.join(base, db), base):
        try:
            path = os.path.join(root, rec)
            sig, info = wfdb.rdsamp(path)
            return sig.astype(np.float32), info
        except Exception:
            continue
    # last resort: remote (may hit fromstring in old WFDB, but our shim should protect)
    sig, info = wfdb.rdsamp(rec, pn_dir=db)
    return sig.astype(np.float32), info

def load_ecg(db="nsrdb", max_steps=200_000):
    if db=="mitdb": recs=["100","101","103","104","107","109"]
    else:           recs=["16265","16272","16273","16420"]
    parts=[]
    for r in recs:
        try:
            sig, info = _load_wfdb_series_local_first(db, r); fs = info["fs"]
            lead = sig[:,0:1]; lead_100 = _resample_to_100hz(lead, fs); parts.append(lead_100)
            print(f"ECG loaded {db}:{r}: {lead_100.shape} @100Hz")
        except Exception as e:
            print(" skip", db, r, "->", e)
    if not parts:
        sig, info = _load_wfdb_series_local_first("mitdb","100")
        lead_100 = _resample_to_100hz(sig[:,0:1], info["fs"]); parts.append(lead_100)
        print("ECG fallback mitdb:100:", lead_100.shape)
    X = np.concatenate(parts, axis=0).astype(np.float32)
    if X.shape[0] > max_steps: X = X[:max_steps]
    print("ECG total:", X.shape); return X

def make_chirp(T=8000, fs=100.0, k=0.2):
    t = np.arange(T)/fs; f0=0.4
    phase = 2*np.pi*(f0*t + 0.5*k*t**2)
    s = np.sin(phase).astype(np.float32)
    am = (1.0 + 0.2*np.sin(2*np.pi*0.1*t)).astype(np.float32)
    y = (am*s).reshape(-1,1).astype(np.float32)
    print(f"Chirp total: {y.shape} (k={k})"); return y

def make_windows(U: np.ndarray, W: int, S: int):
    T = U.shape[0]; starts = list(range(0, T-W+1, S))
    return np.stack([U[s:s+W] for s in starts], axis=0)

def split_tr_va_te(wins: np.ndarray, rtr=0.7, rva=0.15):
    N=len(wins); n_tr=int(N*rtr); n_va=int(N*rva)
    return wins[:n_tr], wins[n_tr:n_tr+n_va], wins[n_tr+n_va:]

# ------------------- Masking: spike-centered for ECG -------------------
from scipy.signal import find_peaks

def _detect_r_peaks(y1d: np.ndarray, fs=100, min_dist=40):
    y = y1d - np.median(y1d)
    mad = np.median(np.abs(y)) + 1e-6
    thr_prom = 2.0 * mad
    peaks, props = find_peaks(y, distance=min_dist, prominence=thr_prom)
    return peaks, props

def _mask_ecg_over_spike(window: np.ndarray, gap_len: int, window_id: int):
    W = window.shape[0]; y = window[:,0].astype(np.float32)
    peaks, props = _detect_r_peaks(y, fs=100, min_dist=40)
    keep = [i for i,p in enumerate(peaks) if (p - gap_len//2 >= 2 and p + (gap_len - gap_len//2) < W-2)]
    if keep:
        prom = props.get("prominences", np.ones(len(peaks)))
        idx = max(keep, key=lambda i: prom[i]); p = int(peaks[idx]); s = int(p - gap_len//2)
    else:
        d = np.abs(np.diff(y, prepend=y[0])); p = int(np.argmax(d))
        s = int(max(2, min(p - gap_len//2, W - gap_len - 2)))
    mask = np.zeros_like(window, dtype=bool); mask[s:s+gap_len,:] = True
    Uc = window.copy(); Uc[mask] = 0.0
    return Uc, mask, s

def _mask_random(window: np.ndarray, gap_len: int, window_id: int):
    W = window.shape[0]
    rng = np.random.RandomState(17 + 1000*window_id + 10*gap_len)
    s = int(rng.randint(2, max(3, W-gap_len-2)))
    mask = np.zeros_like(window, dtype=bool); mask[s:s+gap_len,:] = True
    Uc = window.copy(); Uc[mask]=0.0
    return Uc, mask, s

def mask_one_gap(window: np.ndarray, gap_len:int, dataset_tag:str, window_id:int):
    if dataset_tag.lower()=="ecg" and FORCE_ECG_GAP_ON_SPIKE:
        return _mask_ecg_over_spike(window, gap_len, window_id)
    else:
        return _mask_random(window, gap_len, window_id)

# ------------------- Baselines -------------------
def linear_interp(y: np.ndarray, mask: np.ndarray):
    out=y.copy(); T,D=y.shape; t=np.arange(T)
    for d in range(D):
        m=mask[:,d]
        if m.any():
            obs=~m
            if obs.sum()>=2: out[m,d]=np.interp(t[m], t[obs], y[obs,d])
    return out

def ar2_impute(y: np.ndarray, mask: np.ndarray, ridge=1e-3):
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
            s,e=run[0],run[-1]
            if s-2<0: continue
            for t in range(s,e+1): out[t,d]=a[0]*out[t-1,d]+a[1]*out[t-2,d]
    return out

# ------------------- ESN -------------------
def act_fn(name:str):
    if name=="tanh": return torch.tanh
    if name=="softsign": return F.softsign
    raise ValueError(name)

def _scale_to_radius(W, target=0.9, iters=50):
    with torch.no_grad():
        v=torch.randn(W.shape[1],1, device=W.device)
        for _ in range(iters): v=F.normalize((W.t()@(W@v)), dim=0)
        sigma=torch.norm(W@v)/(torch.norm(v)+1e-12)
        if sigma.item()>1e-8: W *= (target/ sigma.item())
    return W

class ESN(nn.Module):
    def __init__(self, in_dim, state_dim=256, rho=0.9, tau_m=20.0, input_scale=1.0, activation="tanh", seed=123):
        super().__init__()
        g=torch.Generator(device=DEVICE).manual_seed(seed)
        W=torch.randn(state_dim,state_dim, generator=g, device=DEVICE)/math.sqrt(state_dim)
        W=_scale_to_radius(W, target=rho, iters=50)
        self.register_buffer('W', W)
        Win=torch.randn(state_dim,in_dim, generator=g, device=DEVICE)*(input_scale/math.sqrt(in_dim))
        self.register_buffer('Win', Win)
        self.register_buffer('b', torch.zeros(state_dim, device=DEVICE))
        self.leak=1.0/(1.0+tau_m); self.f=act_fn(activation)
    def forward(self, U):  # (B,T,D)
        B,T,_=U.shape; x=torch.zeros(B,self.W.shape[0], device=U.device, dtype=U.dtype); H=[]
        for t in range(T):
            pre=F.linear(x, self.W)+F.linear(U[:,t,:], self.Win)+self.b
            xt=self.f(pre); x=(1-self.leak)*x + self.leak*xt; H.append(x)
        H=torch.stack(H,1); return torch.nan_to_num(H, nan=0., posinf=0., neginf=0.)

# ------------------- Phase-aware features -------------------
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

def make_features_builder(reservoir, mode="raw", pca=None, n_pc=None, add_hilbert=True):
    def f(U_batch_t: torch.Tensor):
        with torch.no_grad():
            H = reservoir(U_batch_t.to(DEVICE).float())
        if mode=="pca":
            assert pca is not None and n_pc is not None
            with torch.no_grad():
                mean = torch.as_tensor(pca.mean_, device=DEVICE, dtype=torch.float32).view(1,1,-1)
                Wp   = torch.as_tensor(pca.components_[:n_pc], device=DEVICE, dtype=torch.float32)
                Hc   = H - mean
                B,T,N = Hc.shape
                PCs = torch.matmul(Hc.view(B*T,N), Wp.t()).view(B,T,-1)
                F_res = torch.nan_to_num(PCs, nan=0., posinf=0., neginf=0.)
        else:
            F_res = torch.nan_to_num(H, nan=0., posinf=0., neginf=0.)
        if add_hilbert:
            F_h = hilbert_feats_batch(U_batch_t)
            mu = F_h.mean(dim=1, keepdim=True); sd = F_h.std(dim=1, keepdim=True) + 1e-3
            F_h = (F_h - mu)/sd
            F = torch.cat([F_res, F_h], dim=-1)
        else:
            F = F_res
        return F.detach()
    return f

def fit_pca_on_states(reservoir, U_tr:np.ndarray, max_wins=120, auto90=True, plot_name="scree.png"):
    idx=np.arange(len(U_tr)); np.random.shuffle(idx); idx=idx[:min(max_wins, len(idx))]
    Hs=[]
    with torch.no_grad():
        for j in idx:
            U=torch.as_tensor(U_tr[j:j+1], device=DEVICE, dtype=torch.float32)
            H=reservoir(U).squeeze(0).detach().cpu().numpy(); Hs.append(H)
    H_all=np.concatenate(Hs, axis=0); Hc = H_all - H_all.mean(0, keepdims=True)
    pca=PCA(n_components=min(Hc.shape[0], Hc.shape[1])).fit(Hc); evr=pca.explained_variance_ratio_
    fig,ax=plt.subplots(figsize=(6,3)); ax.plot(np.arange(1,len(evr)+1), np.cumsum(evr))
    ax.set_xlabel("PCs"); ax.set_ylabel("Cum. explained var"); ax.set_title("Scree (cumulative)"); ax.grid(True)
    savefig(fig, plot_name)
    n_pc=int(np.searchsorted(np.cumsum(evr),0.90)+1) if auto90 else None
    n_pc=max(8, min(n_pc, Hc.shape[1])) if n_pc is not None else None
    return pca, evr, n_pc

# ------------------- EBM head (stable) -------------------
H_CLAMP, YZ_CLAMP, ECLAMP = 8.0, 8.0, 1e3

class TrueEBMHead(nn.Module):
    def __init__(self, in_dim:int, feat_dim:int, rank:int=4, gamma_state:float=None, kappa:float=0.3):
        super().__init__()
        if gamma_state is None: gamma_state=in_dim/float(feat_dim)
        self.gamma_state_val=float(gamma_state); self.kappa=float(kappa)
        self.in_dim=in_dim; self.feat_dim=feat_dim; self.rank=rank
        self.a_diag_raw=nn.Parameter(torch.zeros(feat_dim))
        if rank>0:
            self.U=nn.Parameter(torch.randn(feat_dim, rank)*0.01); self.V=nn.Parameter(torch.randn(feat_dim, rank)*0.01)
        else:
            self.register_parameter('U', None); self.register_parameter('V', None)
        self.B=nn.Parameter(torch.randn(feat_dim,in_dim)*0.01)
        self.C=nn.Parameter(torch.randn(in_dim,feat_dim)*0.01)
        self.log_sigma_y=nn.Parameter(torch.zeros(in_dim)); self.log_sigma_h=nn.Parameter(torch.zeros(feat_dim))
    def _D(self): return 0.99*torch.tanh(self.a_diag_raw)
    def _A(self):
        D=torch.diag(self._D()); 
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
        E=torch.zeros(B, device=Yseq.device, dtype=Yseq.dtype)
        for t in range(1,T):
            h, y_pred=self.step(h, u_prev)
            Ey=((Yseq[:,t,:]-y_pred)/sy).pow(2).mean(1).mul_(0.5).clamp_(max=ECLAMP)
            Eh=((Hseq[:,t,:]-h)/sh).pow(2).mean(1).mul_(0.5).clamp_(max=ECLAMP)
            Ey=torch.nan_to_num(Ey, nan=ECLAMP); Eh=torch.nan_to_num(Eh, nan=ECLAMP)
            E=E + Ey + self.gamma_state_val*Eh
            u_prev=Yseq[:,t,:] if teacher else y_pred
        return torch.nan_to_num(E/max(1,(T-1)), nan=ECLAMP)
    def per_timestep(self, Hseq, Yseq, teacher=True):
        B,T,Dy = Yseq.shape
        h = Hseq[:,0,:]; u_prev = Yseq[:,0,:]
        sy=torch.exp(self.log_sigma_y); sh=torch.exp(self.log_sigma_h)
        Ey_t = torch.zeros(B,T, device=Yseq.device, dtype=Yseq.dtype)
        Eh_t = torch.zeros(B,T, device=Yseq.device, dtype=Yseq.dtype)
        Yp   = torch.zeros(B,T,Dy, device=Yseq.device, dtype=Yseq.dtype)
        Yp[:,0,:] = Yseq[:,0,:]
        for t in range(1,T):
            h, y_pred = self.step(h, u_prev)
            Ey_t[:,t] = torch.nan_to_num(((Yseq[:,t,:]-y_pred)/sy).pow(2).mean(dim=1).mul(0.5).clamp(max=ECLAMP), nan=ECLAMP)
            Eh_t[:,t] = torch.nan_to_num(((Hseq[:,t,:]-h)/sh).pow(2).mean(dim=1).mul(0.5).clamp(max=ECLAMP), nan=ECLAMP)
            Yp[:,t,:] = y_pred
            u_prev = Yseq[:,t,:] if teacher else y_pred
        return Ey_t, Eh_t, Yp

def ebm_reg(head,wA=1e-4,wUV=1e-4,w_sy=5e-4,w_sh=5e-5):
    reg=wA*(head.a_diag_raw**2).mean()
    if head.U is not None: reg += wUV*((head.U**2).mean()+(head.V**2).mean())
    reg += w_sy*(head.log_sigma_y**2).mean() + w_sh*(head.log_sigma_h**2).mean()
    return reg

def _svd_top(W):
    try: return float(torch.linalg.svdvals(W)[0].item())
    except Exception: return float(torch.linalg.norm(W,2).item())

def stabilize_head(head, rho=0.95, b_max=0.8, c_max=2.0):
    with torch.no_grad():
        D=head._D(); s0=float(torch.max(torch.abs(D)).item())
        if head.U is not None and head.V is not None:
            sU=_svd_top(head.U); sV=_svd_top(head.V); s_lr=sU*sV; margin=max(rho - s0, 0.)
            if s_lr>margin and s_lr>0.:
                scale=math.sqrt(max(min(margin/(s_lr+1e-8),1.0), 1e-6)); head.U.mul_(scale); head.V.mul_(scale)
        sB=_svd_top(head.B);  head.B.mul_(min(1.0, b_max/(sB+1e-8)))
        sC=_svd_top(head.C);  head.C.mul_(min(1.0, c_max/(sC+1e-8)))
        head.log_sigma_y.clamp_(-2.5, 2.5); head.log_sigma_h.clamp_(-2.5, 2.5)

# ------------------- Prefit & training -------------------
def _ridge(X, Y, lam=1e-2):
    X=np.asarray(X, dtype=np.float64); Y=np.asarray(Y, dtype=np.float64)
    F=X.shape[1]; W=np.linalg.solve(X.T@X + lam*np.eye(F), X.T@Y)
    return np.nan_to_num(W, nan=0., posinf=0., neginf=0.)

def prefit_decoder_C(head, Htr, Ytr, lam=1e-1):
    H2=Htr.reshape(-1,Htr.shape[-1]); Y2=Ytr.reshape(-1,Ytr.shape[-1])
    C = _ridge(H2, Y2, lam=lam).T.astype(np.float32)
    with torch.no_grad(): head.C.copy_(torch.from_numpy(C).to(DEVICE))
    return C

def prefit_AB(head, Htr, Ytr, lam=1e-2, rank=4, max_wins=200):
    idx=np.arange(Htr.shape[0]); np.random.shuffle(idx); idx=idx[:min(max_wins,len(idx))]
    Hs=Htr[idx]; Ys=Ytr[idx]
    h_prev=Hs[:,:-1,:].reshape(-1,Htr.shape[-1]); h_next=Hs[:,1:,:].reshape(-1,Htr.shape[-1]); u_prev=Ys[:,:-1,:].reshape(-1,Ytr.shape[-1])
    Wab=_ridge(np.concatenate([h_prev,u_prev],1), h_next, lam=lam)
    d=Htr.shape[-1]; A_target=Wab[:d,:].T; B_target=Wab[d:,:].T
    diagA=np.clip(np.diag(A_target), -0.98, 0.98)/0.99; a_raw=np.arctanh(diagA).astype(np.float32)
    R=A_target - np.diag(np.diag(A_target)); U_s,S_s,Vh_s = np.linalg.svd(R, full_matrices=False)
    r=min(rank, len(S_s)); U_lr=(U_s[:,:r]*np.sqrt(np.maximum(S_s[:r],0.)+1e-8)).astype(np.float32)
    V_lr=(Vh_s[:r,:].T*np.sqrt(np.maximum(S_s[:r],0.)+1e-8)).astype(np.float32)
    with torch.no_grad():
        head.a_diag_raw.copy_(torch.from_numpy(a_raw).to(DEVICE))
        if head.U is not None: head.U.copy_(torch.from_numpy(U_lr).to(DEVICE)); head.V.copy_(torch.from_numpy(V_lr).to(DEVICE))
        head.B.copy_(torch.from_numpy(B_target.astype(np.float32)).to(DEVICE)); head.log_sigma_y.fill_(0.); head.log_sigma_h.fill_(0.)
    stabilize_head(head); return A_target, B_target

def make_negatives(Y, K=2, noise_std=0.2, max_shift=None):
    B,T,D=Y.shape; outs=[]
    if max_shift is None: max_shift=max(2,int(T//6))
    for _ in range(K):
        y=Y.clone(); s=np.random.randint(-max_shift, max_shift+1)
        if s!=0: y=torch.roll(y, shifts=s, dims=1)
        if noise_std>0: y=y+noise_std*torch.randn_like(y)
        outs.append(y)
    return torch.cat(outs,0)

def contrastive_loss(head,H,Y,neg_K=2,margin=2.0,teacher=True):
    E_pos=head.energy(H,Y,teacher=teacher); Yneg=make_negatives(Y,neg_K); Hrep=H.repeat_interleave(neg_K,0)
    E_neg=head.energy(Hrep,Yneg,teacher=teacher).view(neg_K,-1).transpose(0,1)
    return torch.nan_to_num(E_pos.mean() + F.relu(margin + E_pos.unsqueeze(1) - E_neg).mean(), nan=1e6)

def head_supervised_losses(head,H,Y,tf_k=3):
    Y_hat=F.linear(H, head.C).clamp(-YZ_CLAMP, YZ_CLAMP); L_dec=0.5*((Y_hat-Y)**2).mean()
    B,T,D=Y.shape; h=H[:,0,:]; u_prev=Y[:,0,:]; errs=[]
    for t in range(1,T):
        h,y_pred=head.step(h,u_prev); errs.append(((y_pred-Y[:,t,:])**2).mean())
        h_k=0.9*h+0.1*H[:,t,:]; u_k=y_pred
        for j in range(1,tf_k):
            tt=t+j; 
            if tt>=T: break
            h_k,y_k=head.step(h_k,u_k); h_k=0.9*h_k+0.1*H[:,tt,:]; errs.append(((y_k-Y[:,tt,:])**2).mean()); u_k=y_k
        u_prev=Y[:,t,:]; h=0.9*h+0.1*H[:,t,:]
    L_tf=torch.stack(errs).mean() if errs else torch.tensor(0., device=Y.device, dtype=Y.dtype)
    return torch.nan_to_num(L_dec, nan=1e6), torch.nan_to_num(L_tf, nan=1e6)

@dataclass
class TrainCfg:
    rank:int=4; gamma_state:float=None
    lr:float=3e-4; wd:float=1e-5; epochs:int=EPOCHS
    batch_size:int=64; neg_K:int=2; margin:float=2.0
    sup_w:float=0.4; tf_w:float=0.4; tf_k:int=3
    ridge_lam:float=1e-2; prefit_wins:int=200

def train_head(features_fn:Callable, U_tr:np.ndarray, U_va:np.ndarray, cfg:TrainCfg):
    in_dim=U_tr.shape[-1]
    with torch.no_grad():
        H_tr=features_fn(torch.as_tensor(U_tr, device=DEVICE, dtype=torch.float32)).cpu().numpy()
        H_va=features_fn(torch.as_tensor(U_va, device=DEVICE, dtype=torch.float32)).cpu().numpy()
    muY,sdY=zfit(U_tr); muH,sdH=zfit(H_tr)
    Ytr=zapply(U_tr,muY,sdY); Yva=zapply(U_va,muY,sdY); Htr=zapply(H_tr,muH,sdH); Hva=zapply(H_va,muH,sdH)
    feat_dim=Htr.shape[-1]; head=TrueEBMHead(in_dim, feat_dim, rank=cfg.rank, gamma_state=cfg.gamma_state, kappa=0.3).to(DEVICE)
    prefit_decoder_C(head, Htr, Ytr, lam=1e-1); prefit_AB(head, Htr, Ytr, lam=cfg.ridge_lam, rank=cfg.rank, max_wins=cfg.prefit_wins); stabilize_head(head)
    opt=torch.optim.Adam(head.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    best={"val":float("inf"), "state":None}
    def batches(X,Y,bs):
        N=len(X); idx=np.arange(N); np.random.shuffle(idx)
        for i in range(0,N,bs):
            j=idx[i:i+bs]
            yield torch.as_tensor(X[j], device=DEVICE, dtype=torch.float32), torch.as_tensor(Y[j], device=DEVICE, dtype=torch.float32)
    for ep in range(1,cfg.epochs+1):
        head.train(); losses=[]
        for Hb,Yb in batches(Htr,Ytr,cfg.batch_size):
            loss=contrastive_loss(head,Hb,Yb,cfg.neg_K,cfg.margin,teacher=True)
            L_dec,L_tf=head_supervised_losses(head,Hb,Yb,tf_k=cfg.tf_k)
            loss=loss + cfg.sup_w*L_dec + cfg.tf_w*L_tf + ebm_reg(head)
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 0.2); opt.step()
            with torch.no_grad(): stabilize_head(head)
            losses.append(float(loss.detach().cpu()))
        head.eval(); vs=[]
        with torch.no_grad():
            for Hb,Yb in batches(Hva,Yva,cfg.batch_size):
                v=contrastive_loss(head,Hb,Yb,cfg.neg_K,cfg.margin,teacher=True)
                L_dec,L_tf=head_supervised_losses(head,Hb,Yb,tf_k=cfg.tf_k)
                vs.append(float((v + cfg.sup_w*L_dec + cfg.tf_w*L_tf + ebm_reg(head)).detach().cpu()))
        vmean=float(np.mean(vs)) if len(vs)>0 else float("inf")
        print(f"[ep {ep:02d}] val {vmean:.4f}")
        if vmean<best["val"]: best["val"]=vmean; best["state"]={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    if best["state"] is not None: head.load_state_dict(best["state"])
    return head, {"muY":muY,"sdY":sdY,"muH":muH,"sdH":sdH}

# ------------------- Rollout + overlays -------------------
@dataclass
class RolloutCfg: beta_min:float=0.2; beta_max:float=0.4

def _beta_schedule(s,e, beta_min, beta_max):
    L=e-s+1
    if L<=1: return np.array([beta_max],dtype=np.float32)
    idx=np.arange(L); r=1.0 - np.abs((idx-0.5*(L-1))/(0.5*(L-1)))
    return (beta_min + (beta_max-beta_min)*r).astype(np.float32)

def seed_ar2_or_phasecopy(U_win, mask, dataset_tag):
    if dataset_tag.lower()=="ecg":
        y=U_win[:,0]; idx=np.where(mask[:,0])[0]
        if idx.size==0: return U_win.copy()
        s,e=int(idx[0]), int(idx[-1])
        left=y[max(0,s-220):s]
        if len(left)<40: return ar2_impute(U_win, mask)
        thr=left.mean()+1.0*left.std()
        peaks,_=find_peaks(left, height=thr, distance=30)
        if len(peaks)>=2: P=peaks[-1]-peaks[-2]
        else:
            x=left-left.mean(); ac=np.correlate(x,x,mode='full')[len(x)-1:]; lmin=20; lmax=min(len(ac)-1,160)
            P=lmin+int(np.argmax(ac[lmin:lmax+1])) if lmax>lmin else 50
        P=int(max(20, min(P,160)))
        if s-P<0: return ar2_impute(U_win, mask)
        template=y[s-P:s].copy(); baseL=np.median(y[max(0,s-10):s])
        template=template-template.mean()+baseL
        out=U_win.copy()
        for t in range(s, e+1): out[t,0]=template[(t-s)%P]
        if e+1 < len(y):
            baseR=np.median(y[e+1:min(len(y), e+11)])
            if baseR!=baseL:
                L=e-s+1
                for i,t in enumerate(range(s,e+1)):
                    w=(i+1)/L; out[t,0]=(1-w)*out[t,0]+w*(out[t,0]-baseL+baseR)
        return out
    else:
        return ar2_impute(U_win, mask)

def _rollout_one_side(head, Hn, Yz, s, e, beta_vec):
    with torch.no_grad():
        A=head._A(); Yout=Yz.clone()
        if s>0: h=Hn[:,s-1,:].clone(); u_prev=Yz[:,s-1,:].clone()
        else:   h=Hn[:,s,:].clone();    u_prev=F.linear(Hn[:,s,:], head.C).unsqueeze(1)[:,0,:]
        for i,t in enumerate(range(s,e+1)):
            pre=F.linear(h,A)+F.linear(u_prev,head.B); h=(1-head.kappa)*h + head.kappa*pre
            b=float(beta_vec[i]); h=(1.0-b)*h + b*Hn[:,t,:]
            y_pred=F.linear(h, head.C); h=torch.clamp(h,-H_CLAMP,H_CLAMP); y_pred=torch.clamp(y_pred,-YZ_CLAMP,YZ_CLAMP)
            Yout[:,t,:]=y_pred; u_prev=y_pred
    return torch.nan_to_num(Yout, nan=0., posinf=0., neginf=0.)

def impute_bidir_with_unc(head, features_fn, U_win, mask, norms, dataset_tag, rcfg:RolloutCfg, profile_points:int=0):
    U_seed=seed_ar2_or_phasecopy(U_win, mask, dataset_tag)
    with torch.no_grad():
        H_f   = features_fn(torch.as_tensor(U_seed[None,...], device=DEVICE, dtype=torch.float32))
        U_rev = U_seed[::-1].copy()
        H_b_r = features_fn(torch.as_tensor(U_rev[None,...], device=DEVICE, dtype=torch.float32))
    muY=torch.as_tensor(norms["muY"],device=DEVICE,dtype=torch.float32); sdY=torch.as_tensor(norms["sdY"],device=DEVICE,dtype=torch.float32)
    muH=torch.as_tensor(norms["muH"],device=DEVICE,dtype=torch.float32); sdH=torch.as_tensor(norms["sdH"],device=DEVICE,dtype=torch.float32)
    Hn_f=zapply_t(H_f,muH,sdH); Hn_b=torch.flip(zapply_t(H_b_r,muH,sdH), dims=[1]); Yz=torch.as_tensor(zapply(U_win[None,...],norms["muY"],norms["sdY"]), device=DEVICE, dtype=torch.float32)
    idx=np.where(mask[:,0])[0]
    if idx.size==0:
        with torch.no_grad():
            Ey_t=torch.zeros(1,U_win.shape[0], device=DEVICE); Eh_t=torch.zeros(1,U_win.shape[0], device=DEVICE)
        return U_win.copy(), None, None, (0, -1), None, None
    s,e=int(idx[0]), int(idx[-1]); beta_vec=_beta_schedule(s,e, rcfg.beta_min, rcfg.beta_max)
    Y_f=_rollout_one_side(head,Hn_f,Yz.clone(),s,e,beta_vec)
    T=Yz.shape[1]; s_r,e_r=T-1-e, T-1-s
    Y_b_r=_rollout_one_side(head, zapply_t(H_b_r,muH,sdH), torch.flip(Yz.clone(), dims=[1]), s_r,e_r, beta_vec); Y_b=torch.flip(Y_b_r,dims=[1])
    w = torch.linspace(0,1,steps=(e - s + 1), device=DEVICE).view(1,-1,1) if e>s else torch.ones(1,1,1, device=DEVICE)*0.5
    Y_blend=Yz.clone(); Y_blend[:,s:e+1,:]=(1-w)*Y_f[:,s:e+1,:] + w*Y_b[:,s:e+1,:]
    Y_imp = zinvert_t(Y_blend, muY,sdY)[0].detach().cpu().numpy()
    Y_f_dn= zinvert_t(Y_f,     muY,sdY)[0].detach().cpu().numpy()
    Y_b_dn= zinvert_t(Y_b,     muY,sdY)[0].detach().cpu().numpy()
    with torch.no_grad():
        Ey_t,Eh_t,_=head.per_timestep(Hn_f, Y_blend, teacher=True)
        unc_t=(Ey_t + head.gamma_state_val*Eh_t).detach().cpu().numpy()[0]
    if profile_points and profile_points>1:
        alphas=np.linspace(0.,1.,num=profile_points).astype(np.float32); E_vals=[]
        with torch.no_grad():
            for a in alphas:
                Ya=Yz.clone()
                if e> s: Ya[:,s:e+1,:]=(1-a)*Y_f[:,s:e+1,:] + a*Y_b[:,s:e+1,:]
                else:    Ya[:,s:e+1,:]=0.5*(Y_f[:,s:e+1,:] + Y_b[:,s:e+1,:])
                E=head.energy(Hn_f, Ya, teacher=True).mean(); E_vals.append(float(E.detach().cpu()))
        return Y_imp, unc_t, (alphas, np.array(E_vals, dtype=np.float32)), (s,e), Y_f_dn, Y_b_dn
    else:
        return Y_imp, unc_t, None, (s,e), Y_f_dn, Y_b_dn

# ------------------- Metrics -------------------
def metrics_on_mask(y_true, y_hat, mask):
    m=mask.astype(bool)
    if not m.any(): return {"mae":np.nan,"rmse":np.nan,"r2":np.nan}
    diff=y_hat[m]-y_true[m]; mae=float(np.mean(np.abs(diff))); rmse=float(np.sqrt(np.mean(diff**2)))
    yt=y_true[m]; mu=yt.mean(); ss_tot=np.sum((yt-mu)**2)+1e-12; ss_res=np.sum(diff**2); r2=float(1.0-ss_res/ss_tot)
    return {"mae":mae,"rmse":rmse,"r2":r2}

# ------------------- Arch specs -------------------
@dataclass
class ESNSpec: activation:str; state_dim:int=STATE_DIM; rho:float=0.9; tau_m:float=20.0; input_scale:float=1.0; seed:int=123
@dataclass
class ArchSpec: name:str; esn:ESNSpec; feature_type:str; pca_auto90:bool=True

# ------------------- Overlay plotting -------------------
def _std_alpha_blend(yf: np.ndarray, yb: np.ndarray):
    return (np.sqrt(1.0/12.0) * np.abs(yb - yf)).astype(np.float32)

def plot_overlay(y_true:np.ndarray, y_lin:np.ndarray, y_ar2:np.ndarray, y_ebm:np.ndarray,
                 mask:np.ndarray, dataset:str, arch:str, gap_len:int, win_id:int,
                 s:int, e:int, y_f:np.ndarray, y_b:np.ndarray, unc_t:np.ndarray):
    T = y_true.shape[0]; t = np.arange(T)
    yT = y_true[:,0]; yL=y_lin[:,0]; yA=y_ar2[:,0]; yE=y_ebm[:,0]
    band_std = np.zeros_like(yE)
    if y_f is not None and y_b is not None:
        band_std[s:e+1] = _std_alpha_blend(y_f[s:e+1,0], y_b[s:e+1,0])
    fig, ax = plt.subplots(figsize=(8,3.2))
    ax.plot(t, yT, label="Ground truth", linewidth=2)
    ax.plot(t, yL, "--", label="Linear")
    ax.plot(t, yA, "-.", label="AR(2)")
    ax.plot(t, yE, "r:", linewidth=2, label="EBM (ours)")
    if e >= s:
        ax.fill_between(t[s:e+1], yE[s:e+1]-band_std[s:e+1], yE[s:e+1]+band_std[s:e+1],
                        color="C0", alpha=0.15, label="Unc. band")
    ax.axvspan(s-0.5, e+0.5, color="k", alpha=0.08, label="Gap")
    ax.set_title(f"{dataset} | {arch} | gap={gap_len} | win={win_id}")
    ax.set_xlabel("t"); ax.set_ylabel("signal"); ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=4, fontsize=9)
    savefig(fig, f"{dataset}_{arch}_overlay_gap{gap_len}_win{win_id:04d}.png")

def maybe_save_overlay(counter_dict:dict, dataset:str, arch:str, gap_len:int, win_id:int, total_wins:int)->bool:
    if not SAVE_OVERLAYS: return False
    key=(dataset, arch, gap_len)
    if key not in counter_dict:
        counter_dict[key]={"count":0, "stride":max(1, total_wins//OVERLAYS_PER_GAP)}
    if counter_dict[key]["count"]>=OVERLAYS_PER_GAP: return False
    stride=counter_dict[key]["stride"]
    if (win_id % stride)==0 or counter_dict[key]["count"]==0:
        counter_dict[key]["count"] += 1
        return True
    return False

# ------------------- Evaluation (per-window + overlays) -------------------
def evaluate_gaps_detailed(head, features_fn, norms, U_te:np.ndarray, dataset_tag:str, gaps=(10,20,50),
                           rcfg:RolloutCfg=RolloutCfg(), profile_points:int=0, max_windows:int=None,
                           arch_name:str="arch", overlay_counter:dict=None):
    records=[]
    WN = len(U_te) if max_windows is None else min(len(U_te), max_windows)
    for L in gaps:
        pbar = tqdm(range(WN), desc=f"[test {dataset_tag}|gap={L}]")
        for i in pbar:
            y = U_te[i]
            Uc, mask, start = mask_one_gap(y, L, dataset_tag, i)
            y_lin = linear_interp(y, mask)
            y_ar2 = ar2_impute(y, mask)
            y_ebm, unc_t, profile, (s,e), y_f, y_b = impute_bidir_with_unc(head, features_fn, y, mask, norms, dataset_tag, rcfg, profile_points=profile_points)
            mE = metrics_on_mask(y, y_ebm, mask); mA = metrics_on_mask(y, y_ar2, mask); mL = metrics_on_mask(y, y_lin, mask)
            rec = {"dataset":dataset_tag, "gap":L, "window_id":i, "start":int(start),
                   "MAE_EBM":mE["mae"], "RMSE_EBM":mE["rmse"], "R2_EBM":mE["r2"],
                   "MAE_AR2":mA["mae"], "MAE_Lin":mL["mae"]}
            if unc_t is not None:
                mm = mask[:,0].astype(bool); rec["UNC_mean"] = float(np.mean(unc_t[mm])) if mm.any() else np.nan
            records.append(rec)
            if overlay_counter is not None and maybe_save_overlay(overlay_counter, dataset_tag, arch_name, L, i, WN):
                plot_overlay(y, y_lin, y_ar2, y_ebm, mask, dataset_tag, arch_name, L, i, s, e, y_f, y_b, unc_t)
                if SAVE_ENERGY_PROFILES and profile is not None:
                    alphas, E_vals = profile
                    fig, ax = plt.subplots(figsize=(4.5,3))
                    ax.plot(alphas, E_vals, marker='o'); ax.set_xlabel(r'blend $\alpha$'); ax.set_ylabel('Energy')
                    ax.set_title(f"{dataset_tag} | {arch_name} | gap={L} | win={i}")
                    ax.grid(True, alpha=0.3)
                    savefig(fig, f"{dataset_tag}_{arch_name}_energy_profile_gap{L}_win{i:04d}.png")
    return records

# ------------------- Stats & recommendations -------------------
def paired_effects(x, y):
    d = np.array(x)-np.array(y); dz = float(np.mean(d)/(np.std(d, ddof=1)+1e-12)) if len(d)>1 else np.nan
    if len(d)==0: return dz, np.nan
    n_pos = np.sum(d>0); n_neg=np.sum(d<0); delta = float((n_pos - n_neg)/len(d)); return dz, delta

def holm_bonferroni(pvals, labels):
    order = np.argsort(pvals); m = len(pvals); adj = [None]*m
    for rank,k in enumerate(order, start=1):
        adj_p = pvals[k] * (m - rank + 1); adj[k] = min(1.0, adj_p)
    return np.array(adj), order

def ebm_significance_table(df_all: pd.DataFrame, dataset_tags=("chirp","ecg")):
    out_rows=[]; pair_rows=[]
    ebm_archs = sorted(df_all['arch'].unique())
    for ds in dataset_tags:
        for gap in sorted(df_all['gap'].unique()):
            sub = df_all[(df_all['dataset']==ds) & (df_all['gap']==gap)]
            if len(sub)==0: continue
            pivot = sub.pivot_table(index='window_id', columns='arch', values='MAE_EBM', aggfunc='first').dropna()
            n = len(pivot)
            for a in ebm_archs:
                if a in pivot.columns:
                    mu=float(pivot[a].mean()); sd=float(pivot[a].std(ddof=1)) if n>1 else 0.0
                    out_rows.append({"dataset":ds, "gap":gap, "arch":a, "N":n, "MAE_mean":mu, "MAE_std":sd})
            pairs=[]
            for i,a in enumerate(ebm_archs):
                for b in ebm_archs[i+1:]:
                    if a not in pivot.columns or b not in pivot.columns: continue
                    xa = pivot[a].values; xb=pivot[b].values
                    try: _, p = wilcoxon(xa, xb, alternative='two-sided', zero_method='wilcox', mode='auto')
                    except Exception: p = np.nan
                    dz, delta = paired_effects(xa, xb); pairs.append(((a,b), p, dz, delta))
            if pairs:
                pvals = np.array([p if np.isfinite(p) else 1.0 for (_,p,_,_) in pairs])
                adj, order = holm_bonferroni(pvals, labels=[f"{a} vs {b}" for (a,b),_,_,_ in pairs])
                for idx, ((a,b), p, dz, delta) in enumerate(pairs):
                    pair_rows.append({"dataset":ds, "gap":gap, "A":a, "B":b, "p_raw":float(p) if np.isfinite(p) else np.nan,
                                      "p_holm":float(adj[idx]), "cohen_dz":dz, "cliffs_delta":delta, "N":n})
    tbl = pd.DataFrame(out_rows).sort_values(["dataset","gap","MAE_mean"])
    pairs_df = pd.DataFrame(pair_rows).sort_values(["dataset","gap","p_holm"])
    return tbl, pairs_df

def recommend_design(tbl: pd.DataFrame):
    recs={}
    for ds in tbl['dataset'].unique():
        sub = tbl[tbl['dataset']==ds]
        if len(sub)==0: continue
        winners = sub.sort_values(["gap","MAE_mean"]).groupby("gap").first().reset_index()
        overall = sub.groupby("arch")["MAE_mean"].mean().sort_values().index.tolist()
        recs[ds] = {"overall_rank": overall,
                    "per_gap_best": [{"gap":int(r["gap"]), "arch":r["arch"], "MAE_mean":float(r["MAE_mean"])} for _,r in winners.iterrows()]}
    if len(recs)>0:
        archs = list(tbl['arch'].unique()); scores = {a:0.0 for a in archs}
        for ds in recs:
            ranks = {a:i for i,a in enumerate(recs[ds]["overall_rank"])}
            for a in archs: scores[a] += ranks.get(a, len(archs))
        recs["combined_overall_rank"] = sorted(archs, key=lambda a: scores[a])
    return recs

# ------------------- Runner per-arch -------------------
@dataclass
class ESNSpec: activation:str; state_dim:int=STATE_DIM; rho:float=0.9; tau_m:float=20.0; input_scale:float=1.0; seed:int=123
@dataclass
class ArchSpec: name:str; esn:ESNSpec; feature_type:str; pca_auto90:bool=True

def run_architecture(arch:ArchSpec, U_tr, U_va, U_te, dataset_tag:str):
    print(f"\n=== Architecture: {arch.name} on {dataset_tag} ===")
    esn=ESN(in_dim=1, state_dim=arch.esn.state_dim, rho=arch.esn.rho, tau_m=arch.esn.tau_m,
            input_scale=arch.esn.input_scale, activation=arch.esn.activation, seed=arch.esn.seed+7).to(DEVICE).eval()
    if arch.feature_type=="pca":
        p_scree=f"{dataset_tag}_{arch.name}_scree.png"
        pca,_,k_auto=fit_pca_on_states(esn, U_tr, auto90=arch.pca_auto90, plot_name=p_scree); n_pc=k_auto
        features_fn = make_features_builder(esn, mode="pca", pca=pca, n_pc=n_pc, add_hilbert=ADD_HILBERT)
    else:
        features_fn = make_features_builder(esn, mode="raw", add_hilbert=ADD_HILBERT)
    head, norms = train_head(features_fn, U_tr, U_va, TrainCfg())
    beta_cands=[(0.2,0.4),(0.25,0.45)] if dataset_tag!="ecg" else [(0.3,0.5),(0.25,0.45)]
    best=(1e9,None)
    for bmin,bmax in beta_cands:
        rows = evaluate_gaps_detailed(head, features_fn, norms, U_va[:min(32,len(U_va))], dataset_tag, gaps=(10,20,50),
                                      rcfg=RolloutCfg(bmin,bmax), profile_points=0, max_windows=min(32,len(U_va)),
                                      arch_name=arch.name, overlay_counter=None)
        v = np.nanmean([r["MAE_EBM"] for r in rows])
        if v<best[0]: best=(v,(bmin,bmax))
    print(f"  selected β=({best[1][0]},{best[1][1]})")
    overlay_counter = {} if SAVE_OVERLAYS else None
    recs = evaluate_gaps_detailed(head, features_fn, norms, U_te, dataset_tag, gaps=(10,20,50),
                                  rcfg=RolloutCfg(*best[1]), profile_points=PROFILE_POINTS, max_windows=TEST_EVAL_MAX,
                                  arch_name=arch.name, overlay_counter=overlay_counter)
    df = pd.DataFrame(recs); df["arch"]=arch.name
    csv_path = os.path.join(OUTDIR, f"{dataset_tag}_{arch.name}_PERWINDOW.csv"); df.to_csv(csv_path, index=False); print("[per-window saved]", csv_path)
    return df

# ------------------- Main -------------------
if __name__=="__main__":
    print("Preparing data...")
    W=256; S=128
    # Chirp
    if RUN_CHIRP:
        U_chirp = make_chirp(T=8000, fs=100.0, k=CHIRP_K).astype(np.float32)
        wins_chirp = make_windows(U_chirp, W=W, S=S); Utr_c, Uva_c, Ute_c = split_tr_va_te(wins_chirp, 0.7, 0.15)
    # ECG
    U_ecg = load_ecg(db=ECG_DB, max_steps=200_000).astype(np.float32)
    wins_ecg = make_windows(U_ecg, W=W, S=S); Utr_e, Uva_e, Ute_e = split_tr_va_te(wins_ecg, 0.7, 0.15)

    # Four EBM variants
    archs = [
        ArchSpec(name="tanh_raw",     esn=ESNSpec(activation="tanh",     state_dim=STATE_DIM), feature_type="raw"),
        ArchSpec(name="tanh_pca",     esn=ESNSpec(activation="tanh",     state_dim=STATE_DIM), feature_type="pca"),
        ArchSpec(name="softsign_raw", esn=ESNSpec(activation="softsign", state_dim=STATE_DIM), feature_type="raw"),
        ArchSpec(name="softsign_pca", esn=ESNSpec(activation="softsign", state_dim=STATE_DIM), feature_type="pca"),
    ]

    all_perwin=[]
    if RUN_CHIRP:
        for arch in archs:
            df_c = run_architecture(arch, Utr_c, Uva_c, Ute_c, dataset_tag="chirp")
            all_perwin.append(df_c)
    for arch in archs:
        df_e = run_architecture(arch, Utr_e, Uva_e, Ute_e, dataset_tag="ecg")
        all_perwin.append(df_e)

    df_all = pd.concat(all_perwin, ignore_index=True)
    perwin_csv = os.path.join(OUTDIR, "PER_WINDOW_ALL.csv"); df_all.to_csv(perwin_csv, index=False); print("[ALL per-window saved]", perwin_csv)

    # Summary tables (means by dataset × gap × arch)
    agg = (df_all
        .groupby(['dataset','gap','arch'])
        .agg(MAE_EBM=('MAE_EBM','mean'),
             RMSE_EBM=('RMSE_EBM','mean'),
             R2_EBM=('R2_EBM','mean'),
             MAE_AR2=('MAE_AR2','mean'),
             MAE_Lin=('MAE_Lin','mean'),
             UNC_mean=('UNC_mean','mean'),
             N=('MAE_EBM','count'))
        .reset_index())
    agg_csv = os.path.join(OUTDIR, "SUMMARY_by_dataset_gap_arch.csv"); agg.to_csv(agg_csv, index=False); print("[summary saved]", agg_csv)

    # EBM-only significance (paired)
    tbl, pairs = ebm_significance_table(df_all, dataset_tags=["chirp","ecg"])
    mae_tbl_csv = os.path.join(OUTDIR, "EBM_variant_mae_table.csv"); tbl.to_csv(mae_tbl_csv, index=False); print("[EBM MAE table]", mae_tbl_csv)
    sig_csv = os.path.join(OUTDIR, "EBM_variant_significance.csv"); pairs.to_csv(sig_csv, index=False); print("[EBM significance]", sig_csv)

    # Design recommendation
    recs = recommend_design(tbl); savejson(recs, "EBM_variant_recommendations.json")

    print("\nDone. Results folder:", OUTDIR)
