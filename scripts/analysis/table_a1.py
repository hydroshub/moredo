"""Table A1: pooled held-out NSE per configuration (rows) and variable (columns), reported as
mean +/- std across the five spatial cross-validation replicates. Source data behind Figure 3
(fig3_replicate_nse_frozen.npz). Prints a LaTeX booktabs table and a plain preview."""
import os, sys
import numpy as np

from project_paths import REPO
NPZ = os.path.join(REPO, "results_lateral_experiments", "figures", "fig3_replicate_nse_frozen.npz")
VARS = [("et", "ET"), ("q", "Q"), ("swe", "SWE"), ("twsa", "TWSA")]
DDOF = 1  # sample std across the 5 replicates


def cell(arr):
    """Format one mean +/- std table cell from an array of replicate values."""
    return f"{np.mean(arr):.2f}$\\pm${np.std(arr, ddof=DDOF):.2f}"


def main():
    """Print the Table A1 held-out pooled-NSE rows (mean +/- std across replicates) from the cached npz."""
    d = np.load(NPZ, allow_pickle=True)
    labels = [str(x) for x in d["labels"]]
    print("% --- Table A1 (LaTeX) ---")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Configuration & ET & Q & SWE & TWSA \\\\")
    print("\\midrule")
    for i, lab in enumerate(labels):
        cells = " & ".join(cell(d[k][i]) for k, _ in VARS)
        print(f"{lab} & {cells} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    print("\n% --- plain preview ---")
    hdr = f"{'Configuration':20s} " + " ".join(f"{v:>11s}" for _, v in VARS)
    print(hdr)
    for i, lab in enumerate(labels):
        cells = " ".join(f"{np.mean(d[k][i]):.2f}±{np.std(d[k][i], ddof=DDOF):.2f}" .rjust(11)
                          for k, _ in VARS)
        print(f"{lab:20s} {cells}")


if __name__ == "__main__":
    main()
