"""Regenerate seasonality_context_kg_merged_lstm_af with the left y-axis label only on the
leftmost column (panels a and d), keeping fonts, spacing and the right-axis LAI labels unchanged."""
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_analysis import plot_seasonality_context_by_kg
from basin_attributes import build_basin_tables

from project_paths import REPO
RD = os.path.join(REPO, "results_lateral_experiments")
FIG = os.path.join(RD, "figures")


def main(out_name="seasonality_context_kg_merged_lstm_af.png"):
    """Render Figure 7: season-aligned monthly climatologies (P, PET, ET, F_add, LAI) per KG class."""
    lstm = pickle.load(open(os.path.join(RD, "lstm_af", "averaged_full_output.pickle"), "rb"))
    df_coord, _ = build_basin_tables(list(lstm["valid_ids"]))
    plot_seasonality_context_by_kg(
        {"lstm_af": lstm}, df_coord, "lstm_af",
        os.path.join(FIG, out_name),
        hemisphere="merged", skip_codes=("Ds",), left_label_first_col_only=True,
        lai_label_right_only=True, wspace_override=5.0)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="seasonality_context_kg_merged_lstm_af.png")
    a = p.parse_args()
    main(a.out)
