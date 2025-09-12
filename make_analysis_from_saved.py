# make_analysis_from_saved.py
import os, json, math, argparse
import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon, spearmanr
except Exception:
    raise SystemExit("Please install SciPy: pip install scipy")
try:
    from sklearn.metrics import roc_auc_score
except Exception:
    raise SystemExit("Please install scikit-learn: pip install scikit-learn")

# ---------- utils ----------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def json_safe_dump(obj, path):
    import numpy as np
    def _def(o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.ndarray,)):  return o.tolist()
        if isinstance(o, set):             return list(o)
        return str(o)
    with open(path, "w") as f:
        f.write(json.dumps(obj, indent=2, default=_def))

def holm_bonferroni(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.ones_like(pvals, dtype=float)
    for rank, k in enumerate(order, start=1):
        adjusted[k] = min(1.0, pvals[k]*(m - rank + 1))
    return adjusted

def cohens_dz(x, y):
    d = x - y
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / (sd + 1e-12)) if len(d) > 1 else np.nan

def cliffs_delta(x, y):
    x = x.reshape(-1,1); y = y.reshape(1,-1)
    comp = (x > y).astype(int) - (x < y).astype(int)
    return float(np.sum(comp) / (x.size*y.size))

def write_tex_table(df, path, caption, label, cols, floatfmt=None):
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    spec = "l" * len(cols)
    lines.append(f"\\begin{tabular}{{{spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\midrule")
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                if floatfmt is not None: vals.append(("{:"+floatfmt+"}").format(v))
                else:                    vals.append("{:.4f}".format(v))
            else:
                vals.append(str(v))
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(lines))

# ---------- loaders ----------
def load_perwindow_csv(perwin_path):
    df = pd.read_csv(perwin_path)
    required = {"dataset","gap","window_id","arch","MAE_EBM","MAE_AR2","MAE_Lin"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise SystemExit(f"Missing required columns: {missing} in {perwin_path}")
    return df

# ---------- tables ----------
def full_results_table(df):
    rows = []
    for (ds, gap), sub in df.groupby(["dataset","gap"]):
        # baselines
        base_n = len(sub)
        rows.append({"dataset":ds,"gap":gap,"variant":"Linear",
                     "MAE":sub["MAE_Lin"].mean(),"SD":sub["MAE_Lin"].std(ddof=1),"N":base_n})
        rows.append({"dataset":ds,"gap":gap,"variant":"AR(2)",
                     "MAE":sub["MAE_AR2"].mean(),"SD":sub["MAE_AR2"].std(ddof=1),"N":base_n})
        # ebm variants
        for arch, s in sub.groupby("arch"):
            rows.append({"dataset":ds,"gap":gap,"variant":arch,
                         "MAE":s["MAE_EBM"].mean(),
                         "SD": s["MAE_EBM"].std(ddof=1),
                         "N": len(s)})
    out = pd.DataFrame(rows).sort_values(["dataset","gap","variant"])
    return out

def significance_ebm_variants(df):
    pairs = []
    for (ds, gap), sub in df.groupby(["dataset","gap"]):
        piv = sub.pivot_table(index="window_id", columns="arch", values="MAE_EBM", aggfunc="first").dropna()
        archs = list(piv.columns)
        for i, a in enumerate(archs):
            for b in archs[i+1:]:
                xa, xb = piv[a].values, piv[b].values
                if len(xa) < 5: continue
                try: _, p = wilcoxon(xa, xb, alternative="two-sided", zero_method="wilcox", mode="auto")
                except Exception: p = np.nan
                pairs.append({"dataset":ds,"gap":gap,"A":a,"B":b,
                              "p_raw":float(p) if np.isfinite(p) else np.nan,
                              "dz":cohens_dz(xa, xb),
                              "cliffs_delta":cliffs_delta(xa, xb),
                              "N":len(xa)})
    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df):
        for (ds, gap), sub in pairs_df.groupby(["dataset","gap"]):
            idx = sub.index
            adj = holm_bonferroni(sub["p_raw"].fillna(1.0).values)
            pairs_df.loc[idx, "p_holm"] = adj
    return pairs_df.sort_values(["dataset","gap","p_holm"])

def best_vs_baselines(df):
    out = []
    for (ds, gap), sub in df.groupby(["dataset","gap"]):
        arch_means = sub.groupby("arch")["MAE_EBM"].mean().sort_values()
        if len(arch_means)==0: continue
        best_arch = arch_means.index[0]
        s = sub[sub["arch"]==best_arch].set_index("window_id")
        ar2 = sub.set_index("window_id")["MAE_AR2"]
        lin = sub.set_index("window_id")["MAE_Lin"]
        for base_name, base in [("AR(2)", ar2), ("Linear", lin)]:
            common = s["MAE_EBM"].dropna().index.intersection(base.dropna().index)
            if len(common) < 5: continue
            xa = s.loc[common,"MAE_EBM"].values
            xb = base.loc[common].values
            try: _, p = wilcoxon(xa, xb, alternative="two-sided", zero_method="wilcox", mode="auto")
            except Exception: p = np.nan
            out.append({"dataset":ds,"gap":gap,"arch":best_arch,"baseline":base_name,
                        "N":len(common),
                        "MAE_EBM_mean":float(np.mean(xa)),
                        "MAE_base_mean":float(np.mean(xb)),
                        "p_raw":float(p) if np.isfinite(p) else np.nan,
                        "dz":cohens_dz(xa, xb),
                        "cliffs_delta":cliffs_delta(xa, xb)})
    res = pd.DataFrame(out)
    if len(res):
        for (ds, gap), sub in res.groupby(["dataset","gap"]):
            idx = sub.index
            adj = holm_bonferroni(sub["p_raw"].fillna(1.0).values)
            res.loc[idx, "p_holm"] = adj
    return res.sort_values(["dataset","gap","baseline","p_holm"])

def energy_reliability(df, q=0.75):
    rows = []
    need = {"UNC_mean","MAE_EBM"}
    have_unc = need.issubset(set(df.columns))
    if not have_unc:
        return pd.DataFrame(columns=["dataset","gap","arch","N","spearman_rho","spearman_p","auroc_hardwin"])
    for (ds, gap, arch), sub in df.groupby(["dataset","gap","arch"]):
        e = sub["MAE_EBM"].values
        u = sub["UNC_mean"].values
        if len(e) < 5 or np.all(np.isnan(u)): continue
        try: rho, pr = spearmanr(u, e, nan_policy="omit")
        except Exception: rho, pr = np.nan, np.nan
        thr = np.nanquantile(e, q)
        y_true = (e >= thr).astype(int)
        au = np.nan
        if len(np.unique(y_true)) == 2:
            try:
                au = roc_auc_score(y_true, u)
            except Exception:
                au = np.nan
        rows.append({"dataset":ds,"gap":gap,"arch":arch,"N":len(e),
                     "spearman_rho":float(rho) if np.isfinite(rho) else np.nan,
                     "spearman_p":float(pr) if np.isfinite(pr) else np.nan,
                     "auroc_hardwin":float(au) if np.isfinite(au) else np.nan})
    return pd.DataFrame(rows).sort_values(["dataset","gap","arch"])

def recommendations(df):
    recs = {}
    tbl = (df.groupby(['dataset','gap','arch'])['MAE_EBM'].mean()
             .reset_index().rename(columns={'MAE_EBM':'MAE_mean'}))
    for ds, sub in tbl.groupby("dataset"):
        winners = sub.sort_values(["gap","MAE_mean"]).groupby("gap").first().reset_index()
        overall = sub.groupby("arch")["MAE_mean"].mean().sort_values().index.tolist()
        recs[ds] = {
            "overall_rank": list(map(str, overall)),
            "per_gap_best": [{"gap":int(r["gap"]), "arch":str(r["arch"]), "MAE_mean":float(r["MAE_mean"])}
                             for _, r in winners.iterrows()]
        }
    if len(tbl):
        archs = sorted(tbl["arch"].unique())
        scores = {a:0.0 for a in archs}
        for ds in recs:
            ranks = {a:i for i,a in enumerate(recs[ds]["overall_rank"])}
            for a in archs:
                scores[a] += ranks.get(a, len(archs))
        recs["combined_overall_rank"] = [str(a) for a in sorted(archs, key=lambda a: scores[a])]
    return recs

def significant_graph(pairs_df):
    out = {}
    for (ds, gap), sub in pairs_df.groupby(["dataset","gap"]):
        better = {}
        for _, r in sub.iterrows():
            A, B = str(r["A"]), str(r["B"])
            p = r["p_holm"]; dz = r["dz"] if "dz" in r else r.get("cohen_dz", np.nan)
            if np.isnan(dz):
                dz = r.get("cohen_dz", np.nan)
            if np.isfinite(p) and p < 0.05 and np.isfinite(dz):
                if dz < 0:  # A better
                    better.setdefault(A, set()).add(B)
                elif dz > 0:  # B better
                    better.setdefault(B, set()).add(A)
        out.setdefault(str(ds), {})[str(int(gap))] = {a: sorted(list(v)) for a, v in better.items()}
    return out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Folder containing PER_WINDOW_ALL.csv")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    perwin = os.path.join(root, "PER_WINDOW_ALL.csv")
    if not os.path.exists(perwin):
        raise SystemExit(f"PER_WINDOW_ALL.csv not found in {root}")
    outdir = ensure_dir(os.path.join(root, "analysis"))

    df = load_perwindow_csv(perwin)

    # Full results (baseline + EBM)
    fr = full_results_table(df)
    fr_path = os.path.join(outdir, "FULL_RESULTS.csv")
    fr.to_csv(fr_path, index=False)

    # EBM variant means & pairwise significance
    pairs = significance_ebm_variants(df)
    pairs_path = os.path.join(outdir, "EBM_variant_significance.csv")
    pairs.to_csv(pairs_path, index=False)
    mae_tbl = (df.groupby(['dataset','gap','arch'])['MAE_EBM']
                 .agg(MAE_mean='mean', MAE_std=lambda x: x.std(ddof=1), N='count')
                 .reset_index())
    mae_tbl_path = os.path.join(outdir, "EBM_variant_mae_table.csv")
    mae_tbl.to_csv(mae_tbl_path, index=False)

    # Best vs baselines
    vsb = best_vs_baselines(df)
    vsb_path = os.path.join(outdir, "SIG_BEST_VS_BASELINES.csv")
    vsb.to_csv(vsb_path, index=False)

    # Uncertainty reliability
    unc = energy_reliability(df, q=0.75)
    unc_path = os.path.join(outdir, "UNC_RELIABILITY.csv")
    unc.to_csv(unc_path, index=False)

    # Recommendations & significant winners graph
    recs = recommendations(df)
    json_safe_dump(recs, os.path.join(outdir, "EBM_variant_recommendations.json"))
    sig_map = significant_graph(pairs)
    json_safe_dump(sig_map, os.path.join(outdir, "SIGNIFICANT_MODELS.json"))

    # LaTeX tables
    # (1) compact full results
    comp = fr.copy()
    comp["MAE"] = comp["MAE"].map(lambda x: f"{x:.4f}")
    comp["SD"]  = comp["SD"].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "—")
    comp_tex = comp[["dataset","gap","variant","MAE","SD","N"]]
    write_tex_table(comp_tex,
                    os.path.join(outdir, "table_full_results.tex"),
                    "Imputation error (MAE) by dataset, gap and variant. Baselines included.",
                    "tab:full_results",
                    ["dataset","gap","variant","MAE","SD","N"],
                    floatfmt=None)

    if len(pairs):
        pairs_tex = pairs[["dataset","gap","A","B","p_holm","dz","cliffs_delta","p_raw"]].copy()
        write_tex_table(pairs_tex,
                        os.path.join(outdir, "table_sig_ebm_variants.tex"),
                        "Paired Wilcoxon (Holm-corrected) among EBM variants.",
                        "tab:sig_ebm_variants",
                        ["dataset","gap","A","B","p_holm","dz","cliffs_delta","p_raw"],
                        floatfmt=".4g")

    if len(vsb):
        vsb_tex = vsb[["dataset","gap","arch","baseline","N","MAE_EBM_mean","MAE_base_mean","p_holm","dz","cliffs_delta","p_raw"]]
        write_tex_table(vsb_tex,
                        os.path.join(outdir, "table_sig_best_vs_baselines.tex"),
                        "Best EBM (per dataset×gap) vs baselines (paired Wilcoxon, Holm-corrected).",
                        "tab:sig_best_vs_baselines",
                        ["dataset","gap","arch","baseline","N","MAE_EBM_mean","MAE_base_mean","p_holm","dz","cliffs_delta","p_raw"],
                        floatfmt=".4g")

    print("\nAnalysis written to:", outdir)

if __name__ == "__main__":
    main()
