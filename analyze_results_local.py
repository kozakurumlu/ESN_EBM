# analyze_results_local.py
import argparse, os, json, math
import numpy as np
import pandas as pd

# Robust imports
try:
    from scipy.stats import wilcoxon, spearmanr
except Exception:
    raise SystemExit("Please install SciPy: pip install scipy")
try:
    from sklearn.metrics import roc_auc_score
except Exception:
    raise SystemExit("Please install scikit-learn: pip install scikit-learn")

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
    # Nonparametric effect size
    x = x.reshape(-1,1); y = y.reshape(1,-1)
    comp = (x > y).astype(int) - (x < y).astype(int)
    return float(np.sum(comp) / (x.size*y.size))

def load_perwindow(perwin_path):
    df = pd.read_csv(perwin_path)
    # expected: dataset,gap,window_id,start,MAE_EBM,RMSE_EBM,R2_EBM,MAE_AR2,MAE_Lin,UNC_mean
    needed = {"dataset","gap","window_id","arch","MAE_EBM","MAE_AR2","MAE_Lin","RMSE_EBM","R2_EBM","UNC_mean"}
    missing = [c for c in ["arch"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns {missing} in {perwin_path}. Re-run the latest script which writes 'arch' per window.")
    return df

def table_full_results(df):
    # Aggregate means across windows: EBM variants + baselines for each dataset×gap
    rows = []
    for (ds, gap), sub in df.groupby(["dataset","gap"]):
        # baselines
        base_mae_lin = sub["MAE_Lin"].mean()
        base_mae_ar2 = sub["MAE_AR2"].mean()
        base_n = len(sub)
        rows.append({"dataset":ds,"gap":gap,"variant":"Linear","MAE_mean":base_mae_lin,"MAE_std":sub["MAE_Lin"].std(ddof=1),"N":base_n})
        rows.append({"dataset":ds,"gap":gap,"variant":"AR(2)","MAE_mean":base_mae_ar2,"MAE_std":sub["MAE_AR2"].std(ddof=1),"N":base_n})
        # EBM variants
        for arch, s in sub.groupby("arch"):
            rows.append({"dataset":ds,"gap":gap,"variant":arch,
                         "MAE_mean":s["MAE_EBM"].mean(),
                         "MAE_std":s["MAE_EBM"].std(ddof=1),
                         "RMSE_mean":s["RMSE_EBM"].mean(),
                         "R2_mean":s["R2_EBM"].mean(),
                         "UNC_mean":s["UNC_mean"].mean(),
                         "N":len(s)})
    out = pd.DataFrame(rows).sort_values(["dataset","gap","variant"])
    return out

def significance_ebm_variants(df):
    # Pairwise Wilcoxon for MAE_EBM among EBM variants, per dataset×gap
    pairs = []
    for (ds, gap), sub in df.groupby(["dataset","gap"]):
        piv = sub.pivot_table(index="window_id", columns="arch", values="MAE_EBM", aggfunc="first").dropna()
        archs = list(piv.columns)
        if len(archs) < 2: 
            continue
        for i, a in enumerate(archs):
            for b in archs[i+1:]:
                xa, xb = piv[a].values, piv[b].values
                try:
                    stat, p = wilcoxon(xa, xb, alternative="two-sided", zero_method="wilcox", mode="auto")
                except Exception:
                    p = np.nan
                pairs.append({"dataset":ds,"gap":gap,"A":a,"B":b,
                              "p_raw":float(p) if np.isfinite(p) else np.nan,
                              "dz":cohens_dz(xa, xb),
                              "cliffs_delta":cliffs_delta(xa, xb)})
    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df):
        for (ds, gap), sub in pairs_df.groupby(["dataset","gap"]):
            idx = sub.index
            adj = holm_bonferroni(sub.loc[idx,"p_raw"].fillna(1.0).values)
            pairs_df.loc[idx, "p_holm"] = adj
    return pairs_df.sort_values(["dataset","gap","p_holm"])

def significance_vs_baselines(df):
    # For each dataset×gap×arch: compare EBM MAE vs AR2 and vs Linear (paired across windows)
    out = []
    for (ds, gap), sub in df.groupby(["dataset","gap"]):
        bas_ar2 = sub.set_index("window_id")["MAE_AR2"]
        bas_lin = sub.set_index("window_id")["MAE_Lin"]
        for arch, s in sub.groupby("arch"):
            s = s.set_index("window_id")
            s = s.loc[bas_ar2.index.intersection(s.index)]
            if len(s)==0: 
                continue
            for base_name, base in [("AR(2)", bas_ar2), ("Linear", bas_lin)]:
                common = s["MAE_EBM"].dropna().index.intersection(base.dropna().index)
                if len(common) < 5:
                    continue
                xa = s.loc[common,"MAE_EBM"].values
                xb = base.loc[common].values
                try:
                    _, p = wilcoxon(xa, xb, alternative="two-sided", zero_method="wilcox", mode="auto")
                except Exception:
                    p = np.nan
                out.append({"dataset":ds,"gap":gap,"arch":arch,"baseline":base_name,
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
            adj = holm_bonferroni(sub.loc[idx,"p_raw"].fillna(1.0).values)
            res.loc[idx,"p_holm"] = adj
    return res.sort_values(["dataset","gap","baseline","p_holm"])

def energy_reliability(df, q=0.75):
    # Correlation (Spearman) and AUROC (UNC_mean → top quartile error) per dataset×gap×arch
    rows = []
    for (ds, gap, arch), sub in df.groupby(["dataset","gap","arch"]):
        e = sub["MAE_EBM"].values
        u = sub["UNC_mean"].values
        if len(e) < 5:
            continue
        # Spearman
        try:
            rho, pr = spearmanr(u, e)
        except Exception:
            rho, pr = np.nan, np.nan
        # AUROC for detecting "hard" windows (top quartile by error)
        thr = np.quantile(e, q)
        y_true = (e >= thr).astype(int)
        auroc = np.nan
        if len(np.unique(y_true)) == 2:
            try:
                auroc = roc_auc_score(y_true, u)
            except Exception:
                auroc = np.nan
        rows.append({"dataset":ds,"gap":gap,"arch":arch,"N":len(e),
                     "spearman_rho":rho, "spearman_p":pr, "auroc_hardwin":auroc})
    return pd.DataFrame(rows).sort_values(["dataset","gap","arch"])

def write_tex_table(df, path, caption, label, cols, floatfmt=None):
    # Simple booktabs LaTeX
    with open(path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{%s}\n" % (caption, label))
        f.write("\\begin{tabular}{%s}\n\\toprule\n" % ("l"*len(cols)))
        f.write(" & ".join(cols) + " \\\\\n\\midrule\n")
        for _, r in df.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, (float, np.floating)):
                    if floatfmt is not None:
                        vals.append(("{:"+floatfmt+"}").format(v))
                    else:
                        vals.append("{:.4f}".format(v))
                else:
                    vals.append(str(v))
            f.write(" & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Folder containing PER_WINDOW_ALL.csv")
    args = ap.parse_args()

    perwin_path = os.path.join(args.root, "PER_WINDOW_ALL.csv")
    if not os.path.exists(perwin_path):
        raise SystemExit(f"PER_WINDOW_ALL.csv not found in {args.root}")

    os.makedirs(os.path.join(args.root, "analysis"), exist_ok=True)

    df = load_perwindow(perwin_path)

    # Full results (means)
    full_tbl = table_full_results(df)
    full_tbl.to_csv(os.path.join(args.root, "analysis", "FULL_RESULTS.csv"), index=False)

    # Significance: EBM variants
    ebm_pairs = significance_ebm_variants(df)
    ebm_pairs.to_csv(os.path.join(args.root, "analysis", "SIG_EBM_VARIANTS.csv"), index=False)

    # Significance: best EBM per ds×gap vs baselines
    # choose best by mean MAE in this run
    best_rows = []
    for (ds, gap), sub in df.groupby(["dataset","gap"]):
        arch_means = sub.groupby("arch")["MAE_EBM"].mean().sort_values()
        best_arch = arch_means.index[0]
        best_rows.append({"dataset":ds,"gap":gap,"best_arch":best_arch,"MAE_best":arch_means.iloc[0]})
    best_df = pd.DataFrame(best_rows)
    vs_base = significance_vs_baselines(df.merge(best_df, on=["dataset","gap"])\
                                          .query("arch==best_arch"))
    vs_base.to_csv(os.path.join(args.root, "analysis", "SIG_BEST_VS_BASELINES.csv"), index=False)

    # Uncertainty reliability
    unc = energy_reliability(df)
    unc.to_csv(os.path.join(args.root, "analysis", "UNC_RELIABILITY.csv"), index=False)

    # LaTeX tables (ready to paste)
    # (1) compact summary: dataset×gap×variant MAE_mean ± SD
    comp = full_tbl.copy()
    comp["MAE"] = comp["MAE_mean"].map(lambda x: f"{x:.4f}")
    comp["SD"]  = comp["MAE_std"].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "—")
    comp_tex = comp[["dataset","gap","variant","MAE","SD","N"]]
    write_tex_table(comp_tex, os.path.join(args.root, "analysis", "table_full_results.tex"),
                    "Imputation error (MAE) by dataset, gap and variant. Baselines included.",
                    "tab:full_results",
                    ["dataset","gap","variant","MAE","SD","N"], floatfmt=None)

    # (2) EBM variant significance
    if len(ebm_pairs):
        ebm_tex = ebm_pairs[["dataset","gap","A","B","p_holm","dz","cliffs_delta","p_raw"]].copy()
        write_tex_table(ebm_tex, os.path.join(args.root, "analysis", "table_sig_ebm_variants.tex"),
                        "Paired Wilcoxon (Holm-corrected) among EBM variants.",
                        "tab:sig_ebm_variants",
                        ["dataset","gap","A","B","p_holm","dz","cliffs_delta","p_raw"],
                        floatfmt=".4g")

    # (3) Best EBM vs baselines
    if len(vs_base):
        vb_tex = vs_base[["dataset","gap","arch","baseline","N","MAE_EBM_mean","MAE_base_mean","p_holm","dz","cliffs_delta","p_raw"]]
        write_tex_table(vb_tex, os.path.join(args.root, "analysis", "table_sig_best_vs_baselines.tex"),
                        "Best EBM (per dataset×gap) vs baselines (paired Wilcoxon, Holm-corrected).",
                        "tab:sig_best_vs_baselines",
                        ["dataset","gap","arch","baseline","N","MAE_EBM_mean","MAE_base_mean","p_holm","dz","cliffs_delta","p_raw"],
                        floatfmt=".4g")

    print("Analysis done. See the 'analysis' folder for CSVs and LaTeX tables.")

if __name__ == "__main__":
    main()
