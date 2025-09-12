# make_analysis_here.py
# Rebuild analysis artifacts from saved per-window results (CSV/JSON only).
# Place this script INSIDE a single results\<timestamp>\ folder and run it.
# Outputs are written to .\analysis\

import os, json, sys
import numpy as np
import pandas as pd

# ---- Optional deps with fallbacks (no hard errors) ----
_HAVE_SCIPY = False
_HAVE_SKLEARN = False
try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:
    def wilcoxon(x, y, **kwargs):  # fallback => NaN p
        return (np.nan, np.nan)

try:
    from sklearn.metrics import roc_auc_score
    _HAVE_SKLEARN = True
except Exception:
    def roc_auc_score(y_true, y_score):  # not used when fallback path
        raise RuntimeError("sklearn not available")

ROOT = os.path.abspath(os.path.dirname(__file__))
OUT = os.path.join(ROOT, "analysis")
os.makedirs(OUT, exist_ok=True)

# ----------------- Helpers -----------------
def _json_safe(obj, path):
    def _safe(o):
        if isinstance(o, dict):                 return {str(k): _safe(v) for k,v in o.items()}
        if isinstance(o, (list, tuple, set)):   return [_safe(x) for x in o]
        if isinstance(o, (np.integer,)):        return int(o)
        if isinstance(o, (np.floating,)):       return float(o)
        if isinstance(o, np.ndarray):           return o.tolist()
        return o
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_safe(obj), f, indent=2)

def _holm_bonferroni(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(np.where(np.isfinite(p), p, 1.0))
    m = len(p)
    adj = np.ones_like(p, dtype=float)
    for rank, k in enumerate(order, start=1):
        pv = p[k] if np.isfinite(p[k]) else 1.0
        adj[k] = min(1.0, pv * (m - rank + 1))
    return adj

def _cohens_dz(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    d = x - y
    if d.size <= 1: return np.nan
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / (sd + 1e-12))

def _cliffs_delta(x, y):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(1, -1)
    comp = (x > y).astype(int) - (x < y).astype(int)
    return float(np.sum(comp) / (x.size * y.size))

def _rankdata(a):
    a = np.asarray(a, dtype=float)
    idx = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(idx, dtype=float)
    ranks[idx] = np.arange(len(a), dtype=float)
    s = a[idx]
    start = 0
    for i in range(1, len(a) + 1):
        if i == len(a) or s[i] != s[i-1]:
            r = (start + i - 1) / 2.0
            ranks[idx[start:i]] = r
            start = i
    return ranks + 1.0

def _spearman(u, e):
    # Spearman rho (handles NaNs by pairwise finite masking) — fallback w/o SciPy
    u = np.asarray(u, dtype=float); e = np.asarray(e, dtype=float)
    m = np.isfinite(u) & np.isfinite(e)
    if m.sum() < 2: return np.nan, np.nan
    ru = _rankdata(u[m]); re = _rankdata(e[m])
    cu = ru - ru.mean(); ce = re - re.mean()
    num = float(np.dot(cu, ce))
    den = float(np.sqrt(np.dot(cu, cu) * np.dot(ce, ce)) + 1e-12)
    return num / den, np.nan

def _auroc_topquart_err(u, e):
    u = np.asarray(u, dtype=float); e = np.asarray(e, dtype=float)
    m = np.isfinite(u) & np.isfinite(e)
    if m.sum() < 2: return np.nan
    thr = np.nanquantile(e[m], 0.75)
    y_true = (e >= thr).astype(int)
    if len(np.unique(y_true[m])) < 2: return np.nan
    if _HAVE_SKLEARN:
        try:
            return float(roc_auc_score(y_true[m], u[m]))
        except Exception:
            return np.nan
    # rank-based fallback (Mann–Whitney)
    ranks = _rankdata(u[m]); pos = y_true[m] == 1
    n1 = int(np.sum(pos)); n0 = int(np.sum(~pos))
    if n1 == 0 or n0 == 0: return np.nan
    R1 = float(np.sum(ranks[pos]))
    return float((R1 - n1*(n1+1)/2.0) / (n0*n1 + 1e-12))

# ----------------- Load -----------------
csv_path = os.path.join(ROOT, "PER_WINDOW_ALL.csv")
if not os.path.exists(csv_path):
    print(f"[ERROR] PER_WINDOW_ALL.csv not found in: {ROOT}")
    sys.exit(0)

df = pd.read_csv(csv_path)

need = {"dataset","gap","window_id","arch","MAE_EBM","MAE_AR2","MAE_Lin"}
missing = sorted(list(need - set(df.columns)))
if missing:
    print(f"[ERROR] Missing columns in PER_WINDOW_ALL.csv: {missing}")
    sys.exit(0)

df["dataset"]   = df["dataset"].astype(str)
df["gap"]       = df["gap"].astype(int)
df["window_id"] = df["window_id"].astype(int)
df["arch"]      = df["arch"].astype(str)

# ----------------- FULL RESULTS (means; de-dup per window for baselines) -----------------
rows = []
for (ds, gap), sub in df.groupby(["dataset","gap"]):
    # baselines: 1 value per window_id
    base = (sub.groupby("window_id")
              .agg(MAE_Lin=("MAE_Lin","mean"), MAE_AR2=("MAE_AR2","mean"))
              .reset_index())
    rows.append({"dataset":ds,"gap":gap,"variant":"Linear",
                 "MAE":base["MAE_Lin"].mean(), "SD":base["MAE_Lin"].std(ddof=1), "N":len(base)})
    rows.append({"dataset":ds,"gap":gap,"variant":"AR(2)",
                 "MAE":base["MAE_AR2"].mean(), "SD":base["MAE_AR2"].std(ddof=1), "N":len(base)})
    # EBM variants: de-dup per window_id as well
    for arch, s in sub.groupby("arch"):
        perwin = s.groupby("window_id")["MAE_EBM"].mean()
        rows.append({"dataset":ds,"gap":gap,"variant":arch,
                     "MAE":perwin.mean(), "SD":perwin.std(ddof=1), "N":len(perwin)})
full_tbl = pd.DataFrame(rows).sort_values(["dataset","gap","variant"])
full_tbl.to_csv(os.path.join(OUT, "FULL_RESULTS.csv"), index=False)

# ----------------- Pairwise significance among EBM variants (paired; Holm) -----------------
pairs = []
for (ds, gap), sub in df.groupby(["dataset","gap"]):
    piv = (sub.groupby(["window_id","arch"])["MAE_EBM"]
             .mean().unstack())  # windows × arch
    piv = piv.dropna(axis=0, how="any")
    archs = list(piv.columns)
    for i, a in enumerate(archs):
        for b in archs[i+1:]:
            xa = piv[a].values; xb = piv[b].values
            if len(xa) < 5: continue
            try:
                _, p = wilcoxon(xa, xb, alternative="two-sided", zero_method="wilcox", mode="auto") if _HAVE_SCIPY else (np.nan, np.nan)
            except Exception:
                p = np.nan
            pairs.append({"dataset":ds,"gap":gap,"A":a,"B":b,"N":len(xa),
                          "p_raw":float(p) if np.isfinite(p) else np.nan,
                          "dz":_cohens_dz(xa, xb),
                          "cliffs_delta":_cliffs_delta(xa, xb)})
pairs_df = pd.DataFrame(pairs)
if len(pairs_df):
    for (ds, gap), sub in pairs_df.groupby(["dataset","gap"]):
        idx = sub.index
        pairs_df.loc[idx, "p_holm"] = _holm_bonferroni(sub["p_raw"].fillna(1.0).values)
pairs_df = pairs_df.sort_values(["dataset","gap","p_holm"]) if len(pairs_df) else pairs_df
pairs_df.to_csv(os.path.join(OUT, "EBM_variant_significance.csv"), index=False)

# ----------------- Best EBM vs baselines (paired; Holm) -----------------
vb_rows = []
for (ds, gap), sub in df.groupby(["dataset","gap"]):
    arch_means = (sub.groupby(["arch","window_id"])["MAE_EBM"]
                    .mean().groupby("arch").mean().sort_values())
    if len(arch_means) == 0: continue
    best_arch = str(arch_means.index[0])

    s_perwin  = (sub[sub["arch"]==best_arch].groupby("window_id")["MAE_EBM"].mean())
    base_ar2  = sub.groupby("window_id")["MAE_AR2"].mean()
    base_lin  = sub.groupby("window_id")["MAE_Lin"].mean()

    for base_name, base_series in [("AR(2)", base_ar2), ("Linear", base_lin)]:
        common = s_perwin.dropna().index.intersection(base_series.dropna().index)
        if len(common) < 5: continue
        xa = s_perwin.loc[common].values
        xb = base_series.loc[common].values
        try:
            _, p = wilcoxon(xa, xb, alternative="two-sided", zero_method="wilcox", mode="auto") if _HAVE_SCIPY else (np.nan, np.nan)
        except Exception:
            p = np.nan
        vb_rows.append({"dataset":ds,"gap":gap,"arch":best_arch,"baseline":base_name,
                        "N":len(common),
                        "MAE_EBM_mean":float(np.mean(xa)),
                        "MAE_base_mean":float(np.mean(xb)),
                        "p_raw":float(p) if np.isfinite(p) else np.nan,
                        "dz":_cohens_dz(xa, xb),
                        "cliffs_delta":_cliffs_delta(xa, xb)})
vb = pd.DataFrame(vb_rows)
if len(vb):
    for (ds, gap), sub in vb.groupby(["dataset","gap"]):
        idx = sub.index
        vb.loc[idx, "p_holm"] = _holm_bonferroni(sub["p_raw"].fillna(1.0).values)
vb = vb.sort_values(["dataset","gap","baseline","p_holm"]) if len(vb) else vb
vb.to_csv(os.path.join(OUT, "SIG_BEST_VS_BASELINES.csv"), index=False)

# ----------------- Uncertainty reliability (Spearman & AUROC) -----------------
unc_rows = []
if "UNC_mean" in df.columns:
    for (ds, gap, arch), sub in df.groupby(["dataset","gap","arch"]):
        perwin = (sub.groupby("window_id")
                    .agg(MAE_EBM=("MAE_EBM","mean"), UNC_mean=("UNC_mean","mean"))
                    .dropna())
        if len(perwin) < 5: continue
        rho, pr = _spearman(perwin["UNC_mean"].values, perwin["MAE_EBM"].values)
        try:
            au = _auroc_topquart_err(perwin["UNC_mean"].values, perwin["MAE_EBM"].values)
        except Exception:
            au = np.nan
        unc_rows.append({"dataset":ds,"gap":gap,"arch":arch,"N":len(perwin),
                         "spearman_rho":rho, "spearman_p":pr, "auroc_hardwin":au})
unc_df = pd.DataFrame(unc_rows)
unc_df.to_csv(os.path.join(OUT, "UNC_RELIABILITY.csv"), index=False)

# ----------------- Recommendations and significant winners (JSON-safe) -----------------
mae_tbl = (df.groupby(["dataset","gap","arch"])["MAE_EBM"]
             .mean().reset_index().rename(columns={"MAE_EBM":"MAE_mean"}))
recs = {}
for ds, sub in mae_tbl.groupby("dataset"):
    winners = sub.sort_values(["gap","MAE_mean"]).groupby("gap").first().reset_index()
    overall = sub.groupby("arch")["MAE_mean"].mean().sort_values().index.tolist()
    recs[str(ds)] = {"overall_rank": list(map(str, overall)),
                     "per_gap_best": [{"gap":int(r["gap"]), "arch":str(r["arch"]), "MAE_mean":float(r["MAE_mean"])}
                                      for _, r in winners.iterrows()]}
if len(mae_tbl):
    archs = sorted(mae_tbl["arch"].unique())
    scores = {a:0.0 for a in archs}
    for ds in recs:
        ranks = {a:i for i,a in enumerate(recs[ds]["overall_rank"])}
        for a in archs: scores[a] += ranks.get(a, len(archs))
    recs["combined_overall_rank"] = [str(a) for a in sorted(archs, key=lambda a: scores[a])]
_json_safe(recs, os.path.join(OUT, "EBM_variant_recommendations.json"))

sig_map = {}
if len(pairs_df):
    for (ds, gap), sub in pairs_df.groupby(["dataset","gap"]):
        better = {}
        for _, r in sub.iterrows():
            p = r.get("p_holm", np.nan)
            dz = r.get("dz", np.nan)
            if not np.isfinite(p) or not np.isfinite(dz) or p >= 0.05:
                continue
            A, B = str(r["A"]), str(r["B"])
            if dz < 0:   better.setdefault(A, set()).add(B)
            elif dz > 0: better.setdefault(B, set()).add(A)
        sig_map.setdefault(str(ds), {})[str(int(gap))] = {a: sorted(list(v)) for a, v in better.items()}
_json_safe(sig_map, os.path.join(OUT, "SIGNIFICANT_MODELS.json"))

# Also export per-arch means ± std (de-duplicated per window)
mae_detail = []
for (ds, gap, arch), sub in df.groupby(["dataset","gap","arch"]):
    perwin = sub.groupby("window_id")["MAE_EBM"].mean()
    mae_detail.append({"dataset":ds,"gap":gap,"arch":arch,
                       "MAE_mean":perwin.mean(),
                       "MAE_std":perwin.std(ddof=1),
                       "N":len(perwin)})
pd.DataFrame(mae_detail).sort_values(["dataset","gap","MAE_mean"])\
  .to_csv(os.path.join(OUT, "EBM_variant_mae_table.csv"), index=False)

print("\n[OK] Analysis written to:", OUT)
if not _HAVE_SCIPY:
    print("[Note] SciPy not found; p-values are NaN. Install: pip install scipy")
if not _HAVE_SKLEARN:
    print("[Note] scikit-learn not found; AUROC uses a rank-based fallback or NaN. Install: pip install scikit-learn")
