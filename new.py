# =================================== ESN/LSM -> EBM (ECG-only, metrics-only) ===================================
# - Evaluates ESN / LSM reservoirs feeding an EBM decoder on ECG data only
# - NO chirp generation, NO plots, NO stats tests, NO per-window CSVs, NO JSON
# - Outputs exactly three CSVs (repo root): RMSE, R2, CORR aggregated by gap × reservoir arch
# ===============================================================================================================

# ------------------- Config -------------------
ECG_DB                 = "nsrdb"            # 'nsrdb' or 'mitdb'
FORCE_ECG_GAP_ON_SPIKE = True

STATE_DIM              = 256
RES_CONN_PROB          = 0.10               # sparsity for ESN+LSM graph
ADD_HILBERT            = True
EPOCHS                 = 8
TEST_EVAL_MAX          = 300                # cap #test windows per dataset (lower for speed)
GAPS                   = (10, 20, 50)

SEED_GLOBAL            = 1337

# ------------------- Robust imports/installs -------------------
import sys, subprocess, os, math, random
def _pip(*args): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

try:
    import numpy as np
except Exception:
    _pip("numpy"); import numpy as np

# NumPy 2.x shim for WFDB old np.fromstring usage
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
    np.fromstring = _fromstring_compat

try:
    import torch, torch.nn as nn, torch.nn.functional as F
except Exception:
    _pip("torch"); import torch, torch.nn as nn, torch.nn.functional as F

try:
    import wfdb
except Exception:
    _pip("wfdb>=4.1.3"); import wfdb

try:
    from scipy.signal import resample_poly, hilbert, find_peaks
except Exception:
    _pip("scipy>=1.9.0"); from scipy.signal import resample_poly, hilbert, find_peaks

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

# ------------------- Device, seeds -------------------
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
print("Device:", DEVICE)

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
def zinvert_t(x, mu, sd): return torch.nan_to_num(x*sd+mu, nan=0., posinf=0., neginf=0.)

# ------------------- Data -------------------
def _resample_to_100hz(X, fs_src):
    up, down = 100, int(round(fs_src)); from math import gcd
    g = gcd(up, down); up//=g; down//=g
    return resample_poly(X, up, down, axis=0).astype(np.float32)

def _load_wfdb_series_local_first(db="mitdb", rec="100"):
    base = os.path.join("wfdb_cache", db)
    os.makedirs(base, exist_ok=True)
    try:
        wfdb.dl_record(rec, pn_dir=db, dl_dir=base, keep_subdirs=True, overwrite=False)
    except Exception:
        pass
    for root in (os.path.join(base, db), base):
        try:
            path = os.path.join(root, rec)
            sig, info = wfdb.rdsamp(path)
            return sig.astype(np.float32), info
        except Exception:
            continue
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

def make_windows(U: np.ndarray, W: int, S: int):
    T = U.shape[0]; starts = list(range(0, T-W+1, S))
    return np.stack([U[s:s+W] for s in starts], axis=0)

def split_tr_va_te(wins: np.ndarray, rtr=0.7, rva=0.15):
    N=len(wins); n_tr=int(N*rtr); n_va=int(N*rva)
    return wins[:n_tr], wins[n_tr:n_tr+n_va], wins[n_tr+n_va:]

# ------------------- Masking: spike-centered for ECG -------------------
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

# ------------------- Baselines (used for seeding) -------------------
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

# ------------------- Standardised reservoir graph -------------------
def build_reservoir_graph(N:int, in_dim:int, conn_prob:float, seed:int, input_scale:float=1.0):
    rng = np.random.RandomState(seed)
    mask = (rng.rand(N,N) < conn_prob).astype(np.float32)
    np.fill_diagonal(mask, 0.0)
    W_base = (rng.randn(N,N).astype(np.float32) / np.sqrt(max(1e-6, conn_prob*N))) * mask
    Win_base = (rng.randn(N,in_dim).astype(np.float32) * (input_scale/np.sqrt(in_dim)))
    return W_base, mask, Win_base

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
    def __init__(self, in_dim, state_dim=256, rho=0.9, tau_m=20.0, activation="tanh",
                 W_init:np.ndarray=None, Win_init:np.ndarray=None):
        super().__init__()
        if W_init is None or Win_init is None:
            raise ValueError("ESN requires W_init and Win_init from the standardised graph.")
        W=torch.tensor(W_init, device=DEVICE, dtype=torch.float32)
        W=_scale_to_radius(W, target=rho, iters=50)
        self.register_buffer('W', W)
        self.register_buffer('Win', torch.tensor(Win_init, device=DEVICE, dtype=torch.float32))
        self.register_buffer('b', torch.zeros(state_dim, device=DEVICE))
        self.leak=1.0/(1.0+tau_m); self.f=act_fn(activation)
    def forward(self, U):  # (B,T,D)
        B,T,_=U.shape; x=torch.zeros(B,self.W.shape[0], device=U.device, dtype=U.dtype); H=[]
        for t in range(T):
            pre=F.linear(x, self.W)+F.linear(U[:,t,:], self.Win)+self.b
            xt=self.f(pre); x=(1-self.leak)*x + self.leak*xt; H.append(x)
        H=torch.stack(H,1); return torch.nan_to_num(H, nan=0., posinf=0., neginf=0.)

# ------------------- LSM (LIF spiking) -------------------
class LSM(nn.Module):
    def __init__(self, W_base:np.ndarray, Win_base_1d:np.ndarray, n_in:int,
                 tau_m:float=20.0, v_th:float=1.0, t_ref:int=2, tau_syn:float=5.0,
                 exc_frac:float=0.8, w_scale:float=0.7, seed:int=123):
        super().__init__()
        N=W_base.shape[0]
        rng=np.random.RandomState(seed)
        n_exc=int(exc_frac*N)
        sign=np.ones(N, dtype=np.float32); sign[n_exc:]=-1.0
        W=np.abs(W_base).astype(np.float32)
        W = W * sign[:,None]
        W = W * (w_scale/ max(1e-6, np.max(np.abs(W))))
        self.register_buffer('W_syn', torch.tensor(W, device=DEVICE, dtype=torch.float32))
        Win = np.tile(Win_base_1d, (1,n_in)).astype(np.float32)
        self.register_buffer('Win', torch.tensor(Win, device=DEVICE, dtype=torch.float32))
        self.tau_m=float(tau_m); self.t_ref=int(t_ref); self.tau_syn=float(tau_syn)
        self.v_th=float(v_th); self.v_reset=0.0
        self.N=N
    def forward(self, S_in):  # (B,T,M) spike inputs
        B,T,M=S_in.shape; N=self.N
        v = torch.zeros(B,N, device=S_in.device, dtype=torch.float32)
        ref = torch.zeros(B,N, device=S_in.device, dtype=torch.int32)
        s_prev = torch.zeros(B,N, device=S_in.device, dtype=torch.float32)
        s_all=[]; r_all=[]
        alpha = 1.0/self.tau_syn
        rate = torch.zeros(B,N, device=S_in.device, dtype=torch.float32)
        for t in range(T):
            I_in  = F.linear(S_in[:,t,:], self.Win)
            I_syn = F.linear(s_prev, self.W_syn)
            dv = (-(v - 0.0) + I_in + I_syn)/self.tau_m
            v = v + dv
            can_spike = (ref<=0)
            s = (v >= self.v_th) & can_spike
            s_f = s.float()
            v = torch.where(s, torch.full_like(v, self.v_reset), v)
            ref = torch.where(s, torch.full_like(ref, self.t_ref), ref)
            ref = torch.clamp(ref-1, min=0)
            rate = (1.0 - alpha)*rate + alpha*s_f
            s_all.append(s_f); r_all.append(rate)
            s_prev = s_f
        S = torch.stack(s_all, 1)
        R = torch.stack(r_all, 1)
        return S, R

# ------------------- Level-crossing & Hilbert -------------------
LC_LEVELS = (0.5, 1.0, 2.0, 4.0)

def level_cross_encode_batch(U_bt: torch.Tensor, levels=LC_LEVELS):
    U = U_bt.detach().cpu().numpy()  # B,T,1
    B,T,_ = U.shape
    K = len(levels)
    S = np.zeros((B,T,2*K), dtype=np.float32)
    for b in range(B):
        x = U[b,:,0]
        std = np.std(x) + 1e-6
        for k, mul in enumerate(levels):
            delta = float(mul*std)
            acc = 0.0
            for t in range(1,T):
                acc += (x[t] - x[t-1])
                if acc >= delta:
                    S[b,t,2*k] = 1.0; acc -= delta
                elif acc <= -delta:
                    S[b,t,2*k+1] = 1.0; acc += delta
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

# ------------------- Feature builders -------------------
class ESNFeatures:
    def __init__(self, esn:ESN, mode="raw", pca=None, n_pc=None, add_hilbert=True):
        self.esn=esn; self.mode=mode; self.pca=pca; self.n_pc=n_pc; self.add_hilbert=add_hilbert
    def __call__(self, U_batch_t: torch.Tensor):
        with torch.no_grad():
            H = self.esn(U_batch_t.to(DEVICE).float())
        if self.mode=="pca":
            assert self.pca is not None and self.n_pc is not None
            mean = torch.as_tensor(self.pca.mean_, device=DEVICE, dtype=torch.float32).view(1,1,-1)
            Wp   = torch.as_tensor(self.pca.components_[:self.n_pc], device=DEVICE, dtype=torch.float32)
            Hc   = H - mean
            B,T,N = Hc.shape
            PCs = torch.matmul(Hc.view(B*T,N), Wp.t()).view(B,T,-1)
            F_res = torch.nan_to_num(PCs, nan=0., posinf=0., neginf=0.)
        else:
            F_res = torch.nan_to_num(H, nan=0., posinf=0., neginf=0.)
        if self.add_hilbert:
            F_h = hilbert_feats_batch(U_batch_t)
            mu = F_h.mean(dim=1, keepdim=True); sd = F_h.std(dim=1, keepdim=True) + 1e-3
            F_h = (F_h - mu)/sd
            F = torch.cat([F_res, F_h], dim=-1)
        else:
            F = F_res
        return F.detach()

class LSMFeatures:
    def __init__(self, lsm:LSM, mode="spikes", add_hilbert=True):
        self.lsm=lsm; self.mode=mode; self.add_hilbert=add_hilbert
    def __call__(self, U_batch_t: torch.Tensor):
        S_in = level_cross_encode_batch(U_batch_t)
        with torch.no_grad():
            S,R = self.lsm(S_in)
        F_res = S if (self.mode=="spikes") else R
        if self.add_hilbert:
            F_h = hilbert_feats_batch(U_batch_t)
            mu = F_h.mean(dim=1, keepdim=True); sd = F_h.std(dim=1, keepdim=True) + 1e-3
            F_h = (F_h - mu)/sd
            F = torch.cat([F_res, F_h], dim=-1)
        else:
            F = F_res
        return F.detach()

def fit_pca_on_states_esn(features:ESNFeatures, U_tr:np.ndarray, max_wins=120, auto90=True):
    idx=np.arange(len(U_tr)); np.random.shuffle(idx); idx=idx[:min(max_wins, len(U_tr))]
    Hs=[]
    with torch.no_grad():
        for j in idx:
            U=torch.as_tensor(U_tr[j:j+1], device=DEVICE, dtype=torch.float32)
            H=features.esn(U).squeeze(0).detach().cpu().numpy(); Hs.append(H)
    H_all=np.concatenate(Hs, axis=0); Hc = H_all - H_all.mean(0, keepdims=True)
    pca=PCA(n_components=min(Hc.shape[0], Hc.shape[1])).fit(Hc)
    if auto90:
        evr=pca.explained_variance_ratio_
        n_pc=int(np.searchsorted(np.cumsum(evr),0.90)+1)
        n_pc=max(8, min(n_pc, Hc.shape[1]))
    else:
        n_pc=None
    return pca, n_pc

# ------------------- EBM head -------------------
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
        pre=F.linear(h, A)+F.linear(u_prev, self.B)
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

# ------------------- Rollout -------------------
@dataclass
class RolloutCfg: beta_min:float=0.2; beta_max:float=0.4

def _beta_schedule(s,e, beta_min, beta_max):
    L=e-s+1
    if L<=1: return np.array([beta_max],dtype=np.float32)
    idx=np.arange(L); r=1.0 - np.abs((idx-0.5*(L-1))/(0.5*(L-1)))
    return (beta_min + (beta_max-beta_max)*r + (beta_max-beta_min)*r).astype(np.float32)

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
            baseR=np.median(y[e+1:min(len(y), e+11)]); 
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

def impute_bidir(head, features_fn, U_win, mask, norms, dataset_tag, rcfg:RolloutCfg):
    U_seed=seed_ar2_or_phasecopy(U_win, mask, dataset_tag)
    with torch.no_grad():
        H_f   = features_fn(torch.as_tensor(U_seed[None,...], device=DEVICE, dtype=torch.float32))
        U_rev = U_seed[::-1].copy()
        H_b_r = features_fn(torch.as_tensor(U_rev[None,...], device=DEVICE, dtype=torch.float32))
    muY=torch.as_tensor(norms["muY"],device=DEVICE,dtype=torch.float32); sdY=torch.as_tensor(norms["sdY"],device=DEVICE,dtype=torch.float32)
    muH=torch.as_tensor(norms["muH"],device=DEVICE,dtype=torch.float32); sdH=torch.as_tensor(norms["sdH"],device=DEVICE,dtype=torch.float32)
    Hn_f=(H_f - muH)/sdH; Hn_b_r=(H_b_r - muH)/sdH
    Hn_b=torch.flip(Hn_b_r, dims=[1])
    Yz=torch.as_tensor(zapply(U_win[None,...],norms["muY"],norms["sdY"]), device=DEVICE, dtype=torch.float32)
    idx=np.where(mask[:,0])[0]
    if idx.size==0:
        return U_win.copy()
    s,e=int(idx[0]), int(idx[-1]); beta_vec=_beta_schedule(s,e, rcfg.beta_min, rcfg.beta_max)
    Y_f=_rollout_one_side(head,Hn_f,Yz.clone(),s,e,beta_vec)
    T=Yz.shape[1]; s_r,e_r=T-1-e, T-1-s
    Y_b_r=_rollout_one_side(head, Hn_b_r, torch.flip(Yz.clone(), dims=[1]), s_r,e_r, beta_vec); Y_b=torch.flip(Y_b_r,dims=[1])
    w = torch.linspace(0,1,steps=(e - s + 1), device=DEVICE).view(1,-1,1) if e>s else torch.ones(1,1,1, device=DEVICE)*0.5
    Y_blend=Yz.clone(); Y_blend[:,s:e+1,:]=(1-w)*Y_f[:,s:e+1,:] + w*Y_b[:,s:e+1,:]
    Y_imp = zinvert_t(Y_blend, muY,sdY)[0].detach().cpu().numpy()
    return Y_imp

# ------------------- Metrics -------------------
def corr_on_mask(y_true, y_hat, mask):
    idx = np.where(mask[:,0])[0]
    if idx.size < 2: return np.nan
    a = y_true[idx,0]; b = y_hat[idx,0]
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a*a).sum() * (b*b).sum()) + 1e-12
    return float((a*b).sum() / denom)

def metrics_on_mask(y_true, y_hat, mask):
    m=mask.astype(bool)
    if not m.any(): return {"rmse":np.nan,"r2":np.nan,"corr":np.nan}
    diff=y_hat[m]-y_true[m]
    rmse=float(np.sqrt(np.mean(diff**2)))
    yt=y_true[m]; mu=yt.mean(); ss_tot=np.sum((yt-mu)**2)+1e-12; ss_res=np.sum(diff**2); r2=float(1.0-ss_res/ss_tot)
    corr=corr_on_mask(y_true, y_hat, mask)
    return {"rmse":rmse,"r2":r2,"corr":corr}

# ------------------- Arch specs -------------------
@dataclass
class ESNSpec: activation:str; state_dim:int=STATE_DIM; rho:float=0.9; tau_m:float=20.0; seed:int=123
@dataclass
class LSMSpec: state_dim:int=STATE_DIM; tau_m:float=20.0; v_th:float=1.0; t_ref:int=2; tau_syn:float=5.0; seed:int=123
@dataclass
class ArchSpec: name:str; kind:str; esn:ESNSpec=None; lsm:LSMSpec=None; feature_type:str="raw"; pca_auto90:bool=True

# ------------------- Evaluation -------------------
def evaluate_gaps(head, features_fn, norms, U_te:np.ndarray, dataset_tag:str, gaps=GAPS, max_windows:int=None, arch_name:str="arch"):
    records=[]
    WN = len(U_te) if max_windows is None else min(len(U_te), max_windows)
    for L in gaps:
        pbar = tqdm(range(WN), desc=f"[test {dataset_tag}|gap={L}]")
        for i in pbar:
            y = U_te[i]
            _, mask, start = _mask_ecg_over_spike(y, L, i) if FORCE_ECG_GAP_ON_SPIKE else _mask_random(y, L, i)
            y_ebm = impute_bidir(head, features_fn, y, mask, {"muY":norms["muY"],"sdY":norms["sdY"],"muH":norms["muH"],"sdH":norms["sdH"]}, dataset_tag, RolloutCfg(0.3,0.5))
            mE = metrics_on_mask(y, y_ebm, mask)
            rec = {"dataset":dataset_tag, "gap":L, "window_id":i, "start":int(start),
                   "RMSE_EBM":mE["rmse"], "R2_EBM":mE["r2"], "CORR_EBM":mE["corr"]}
            records.append(rec)
    df = pd.DataFrame(records); df["arch"]=arch_name
    return df

def evaluate_gaps_with_beta(head, features_fn, norms, U_set:np.ndarray, dataset_tag:str, beta_pair, gaps=GAPS, max_windows:int=None, arch_name:str="arch"):
    # Overridden rollout schedule during eval
    records=[]
    WN = len(U_set) if max_windows is None else min(len(U_set), max_windows)
    bmin,bmax = beta_pair
    for L in gaps:
        pbar = tqdm(range(WN), desc=f"[val {dataset_tag}|gap={L}|β=({bmin},{bmax})]")
        for i in pbar:
            y = U_set[i]
            Uc, mask, start = mask_one_gap(y, L, dataset_tag, i)
            # local rollout with chosen betas
            U_seed=seed_ar2_or_phasecopy(y, mask, dataset_tag)
            with torch.no_grad():
                H_f   = features_fn(torch.as_tensor(U_seed[None,...], device=DEVICE, dtype=torch.float32))
                U_rev = U_seed[::-1].copy()
                H_b_r = features_fn(torch.as_tensor(U_rev[None,...], device=DEVICE, dtype=torch.float32))
            muY=torch.as_tensor(norms["muY"],device=DEVICE,dtype=torch.float32); sdY=torch.as_tensor(norms["sdY"],device=DEVICE,dtype=torch.float32)
            muH=torch.as_tensor(norms["muH"],device=DEVICE,dtype=torch.float32); sdH=torch.as_tensor(norms["sdH"],device=DEVICE,dtype=torch.float32)
            Hn_f=(H_f - muH)/sdH; Hn_b_r=(H_b_r - muH)/sdH
            Hn_b=torch.flip(Hn_b_r, dims=[1])
            Yz=torch.as_tensor(zapply(y[None,...],norms["muY"],norms["sdY"]), device=DEVICE, dtype=torch.float32)
            idx=np.where(mask[:,0])[0]
            if idx.size==0:
                y_imp = y.copy()
            else:
                s,e=int(idx[0]), int(idx[-1])
                Lgap = e-s+1
                if Lgap<=1:
                    beta_vec = np.array([bmax], dtype=np.float32)
                else:
                    idxv=np.arange(Lgap); r=1.0 - np.abs((idxv-0.5*(Lgap-1))/(0.5*(Lgap-1)))
                    beta_vec = (bmin + (bmax-bmin)*r).astype(np.float32)
                Y_f=_rollout_one_side(head,Hn_f,Yz.clone(),s,e,beta_vec)
                T=Yz.shape[1]; s_r,e_r=T-1-e, T-1-s
                Y_b_r=_rollout_one_side(head, Hn_b_r, torch.flip(Yz.clone(), dims=[1]), s_r,e_r, beta_vec); Y_b=torch.flip(Y_b_r,dims=[1])
                w = torch.linspace(0,1,steps=(e - s + 1), device=DEVICE).view(1,-1,1) if e>s else torch.ones(1,1,1, device=DEVICE)*0.5
                Y_blend=Yz.clone(); Y_blend[:,s:e+1,:]=(1-w)*Y_f[:,s:e+1,:] + w*Y_b[:,s:e+1,:]
                y_imp = zinvert_t(Y_blend, muY,sdY)[0].detach().cpu().numpy()
            mE = metrics_on_mask(y, y_imp, mask)
            rec = {"dataset":dataset_tag, "gap":L, "window_id":i, "start":int(start),
                   "RMSE_EBM":mE["rmse"], "R2_EBM":mE["r2"], "CORR_EBM":mE["corr"]}
            records.append(rec)
    df = pd.DataFrame(records); df["arch"]=arch_name
    return df

# ------------------- Runner per-arch -------------------
def build_esn_features(spec:ESNSpec, W_base, Win_base):
    esn=ESN(in_dim=1, state_dim=spec.state_dim, rho=spec.rho, tau_m=spec.tau_m,
            activation=spec.activation, W_init=W_base, Win_init=Win_base).to(DEVICE).eval()
    return esn

def build_lsm_features(spec:LSMSpec, W_base, Win_base, n_enc:int, mode:str):
    lsm = LSM(W_base=W_base, Win_base_1d=Win_base, n_in=n_enc,
              tau_m=spec.tau_m, v_th=1.0, t_ref=spec.t_ref, tau_syn=spec.tau_syn, seed=spec.seed+11).to(DEVICE).eval()
    return LSMFeatures(lsm, mode=("spikes" if mode=="spikes" else "rates"), add_hilbert=ADD_HILBERT)

def run_architecture(arch:ArchSpec, U_tr, U_va, U_te, dataset_tag:str, W_base, Win_base):
    print(f"\n=== Architecture: {arch.name} on {dataset_tag} ===")
    # Feature builder
    if arch.kind=="esn":
        esn = build_esn_features(arch.esn, W_base, Win_base)
        fb = ESNFeatures(esn, mode=arch.feature_type, add_hilbert=ADD_HILBERT)
        if arch.feature_type=="pca":
            pca, k_auto = fit_pca_on_states_esn(fb, U_tr, auto90=arch.pca_auto90); n_pc=k_auto
            fb = ESNFeatures(esn, mode="pca", pca=pca, n_pc=n_pc, add_hilbert=ADD_HILBERT)
    elif arch.kind=="lsm":
        fb = build_lsm_features(arch.lsm, W_base, Win_base, n_enc=2*len(LC_LEVELS), mode=arch.feature_type)
    else:
        raise ValueError(arch.kind)

    head, norms = train_head(fb, U_tr, U_va, TrainCfg())

    # Select β by validation RMSE (no MAE anywhere)
    beta_cands=[(0.3,0.5),(0.25,0.45)]  # ECG-friendly candidates
    best=(1e9,None)
    for pair in beta_cands:
        df_val = evaluate_gaps_with_beta(head, fb, norms, U_va[:min(32,len(U_va))], dataset_tag, pair, gaps=GAPS,
                                         max_windows=min(32,len(U_va)), arch_name=arch.name)
        v = np.nanmean(df_val["RMSE_EBM"].values)
        if v<best[0]: best=(v,pair)
    print(f"  selected β=({best[1][0]},{best[1][1]})")

    # Final test evaluation with chosen β (impl via evaluate_gaps uses default ECG β=0.3..0.5)
    # To respect the chosen β exactly, reuse evaluate_gaps_with_beta:
    df_test = evaluate_gaps_with_beta(head, fb, norms, U_te, dataset_tag, best[1], gaps=GAPS,
                                      max_windows=TEST_EVAL_MAX, arch_name=arch.name)
    return df_test

# ------------------- Main -------------------
if __name__=="__main__":
    print("Preparing ECG data only...")
    # Shared reservoir graph
    W_base_np, mask_np, Win_base_np = build_reservoir_graph(STATE_DIM, in_dim=1, conn_prob=RES_CONN_PROB, seed=SEED_GLOBAL, input_scale=1.0)

    # ECG windows
    W=256; S=128
    U_ecg = load_ecg(db=ECG_DB, max_steps=200_000).astype(np.float32)
    wins_ecg = make_windows(U_ecg, W=W, S=S); Utr_e, Uva_e, Ute_e = split_tr_va_te(wins_ecg, 0.7, 0.15)

    # ESN & LSM variants
    archs = [
        ArchSpec(name="tanh_raw",     kind="esn", esn=ESNSpec(activation="tanh",     state_dim=STATE_DIM), feature_type="raw"),
        ArchSpec(name="tanh_pca",     kind="esn", esn=ESNSpec(activation="tanh",     state_dim=STATE_DIM), feature_type="pca"),
        ArchSpec(name="softsign_raw", kind="esn", esn=ESNSpec(activation="softsign", state_dim=STATE_DIM), feature_type="raw"),
        ArchSpec(name="softsign_pca", kind="esn", esn=ESNSpec(activation="softsign", state_dim=STATE_DIM), feature_type="pca"),
        ArchSpec(name="lsm_spikes",   kind="lsm", lsm=LSMSpec(state_dim=STATE_DIM),   feature_type="spikes"),
        ArchSpec(name="lsm_rates",    kind="lsm", lsm=LSMSpec(state_dim=STATE_DIM),   feature_type="rates"),
    ]

    all_test=[]
    for arch in archs:
        df_e = run_architecture(arch, Utr_e, Uva_e, Ute_e, dataset_tag="ecg", W_base=W_base_np, Win_base=Win_base_np)
        all_test.append(df_e)

    df_all = pd.concat(all_test, ignore_index=True)

    # Aggregate by gap × arch: mean/std/N for the three metrics (no MAE anywhere)
    agg = (df_all
        .groupby(['gap','arch'])
        .agg(RMSE_EBM_mean=('RMSE_EBM','mean'),
             RMSE_EBM_std=('RMSE_EBM','std'),
             R2_EBM_mean=('R2_EBM','mean'),
             R2_EBM_std=('R2_EBM','std'),
             CORR_EBM_mean=('CORR_EBM','mean'),
             CORR_EBM_std=('CORR_EBM','std'),
             N=('RMSE_EBM','count'))
        .reset_index())

    # Write exactly three CSVs at repo root
    rmse_df = agg[['gap','arch','RMSE_EBM_mean','RMSE_EBM_std','N']].rename(columns={'RMSE_EBM_mean':'RMSE_mean','RMSE_EBM_std':'RMSE_std'})
    r2_df   = agg[['gap','arch','R2_EBM_mean','R2_EBM_std','N']].rename(columns={'R2_EBM_mean':'R2_mean','R2_EBM_std':'R2_std'})
    corr_df = agg[['gap','arch','CORR_EBM_mean','CORR_EBM_std','N']].rename(columns={'CORR_EBM_mean':'CORR_mean','CORR_EBM_std':'CORR_std'})

    rmse_df.to_csv("ECG_EBM_RMSE_by_gap_arch.csv", index=False)
    r2_df.to_csv("ECG_EBM_R2_by_gap_arch.csv", index=False)
    corr_df.to_csv("ECG_EBM_CORR_by_gap_arch.csv", index=False)

    print("\nSaved:")
    print(" - ECG_EBM_RMSE_by_gap_arch.csv")
    print(" - ECG_EBM_R2_by_gap_arch.csv")
    print(" - ECG_EBM_CORR_by_gap_arch.csv")
    print("Done.")
