"""
Shared analysis helpers: loading of the averaged ensemble outputs (load_results), the
precipitation concentration index (_compute_pci, Oliver 1980, the annual index of the monthly
precipitation totals averaged over 2001-2020), the seasonal-climatology panels of Fig. 7 (plot_seasonality_context_by_kg)
and the SHAP dependence panels of Fig. 6 (plot_shap_dependence_grid).
"""
import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd

from plot_style import (uplt, sns, CATEGORY_NAMES, CATEGORY_COLORS, EXP_LABELS, EXP_COLORS,
                         LABELSIZE, TICKLABELSIZE, TITLESIZE, LEGENDSIZE, format_map, right_colorbar)
import cartopy.crs as ccrs

from basin_attributes import (build_basin_tables, NORTH_AMERICA_COUNTRIES, EUROPE_COUNTRIES,
                              KG_MAJOR_NAMES, KG_MAJOR_COLORS, KG_MAJOR, load_kg_major_grid)
from metrics import basin_nse_table, basin_nse_twsa

EXPERIMENTS = ["mlp_a", "mlp_af", "lstm_af"]


def load_results(results_dir, experiments=EXPERIMENTS):
    """Load each experiment's 25-instance ensemble-averaged output pickle into a dict keyed by experiment name."""
    results = {}
    for exp in experiments:
        path = os.path.join(results_dir, exp, "averaged_full_output.pickle")
        with open(path, "rb") as f:
            results[exp] = pickle.load(f)
    return results


# =============================================================================
# 1. Spatial pattern
# =============================================================================
# Tight bounding boxes for the Europe+CONUS zoomed maps (basins only, not full continents).
US_LONLIM, US_LATLIM = (-126, -68), (27, 51)
EU_LONLIM, EU_LATLIM = (-12, 30), (35, 62)

# Equal-area regional projections (ccrs.epsg(2163)
# for CONUS; PlateCarree distorts shape/area badly once zoomed to a small lon/lat box).
REGION_PROJ = {"US": ccrs.epsg(2163), "Europe": ccrs.epsg(3035)}

# Fixed per-panel width so the fadd and NSE figures render at identical map sizes
# regardless of colorbar label width (tight layout otherwise steals space from the axes).
MAP_REFWIDTH = "85mm"


# =============================================================================
# 2. Seasonal / temporal pattern (only meaningful for dynamic experiments —
#    mlp_a's lat_alpha is time-invariant per basin, so f_lat has no seasonality there)
# =============================================================================
def _reconstruct_precip(result):
    """P = rain + snow exactly reproduces the raw driver (rainsnow_partition conserves mass)."""
    return np.squeeze(result["fluxes"]["rain"]) + np.squeeze(result["fluxes"]["snow"])


def plot_seasonality_context_by_kg(results, df_coord, exp, out_path, hemisphere="NH", end_year=2020,
                                   single_ylabel=False, skip_codes=(), nh_sh_inset=False, left_ymax=None,
                                   lai_label_right_only=False, left_label_first_col_only=False,
                                   wspace_override=None, figwidth="260mm", hspace_override=None,
                                   label_size=None, tick_size=None, xlabel_size=None, title_size=None):
    """Monthly P/PET/ET/f_add/LAI climatology, one panel per Koppen-Geiger major class present
    in `hemisphere` -- styled after Fig. 4 of https://www.nature.com/articles/s41586-025-09359-3
    (P as bars, PET/ET/f_add as left-axis lines, LAI as a right-axis line). Restricted to a single
    hemisphere to avoid pooling basins across hemispheres within the same calendar month
    (see plot_seasonal_pattern_by_kg_aridity); call once per hemisphere for full basin coverage."""
    from matplotlib.patches import Patch

    result = results[exp]
    P = _reconstruct_precip(result)
    PET = np.squeeze(result["diagnos"]["pet"])
    ET = np.squeeze(result["fluxes"]["et"])
    FADD = np.squeeze(result["fluxes"]["f_lat"])
    LAI = np.squeeze(result["diagnos"]["LAI_normed"])

    dates = pd.date_range(end=f"{end_year}-12-31", periods=P.shape[1], freq="D")
    months = np.asarray(dates.month)
    north_all = (df_coord["basin_lat"] >= 0).values
    south_all = ~north_all
    if hemisphere == "NH":
        hemi_all = north_all
    elif hemisphere == "SH":
        hemi_all = south_all
    elif hemisphere == "merged":
        # pool both hemispheres; SH climatologies are shifted +6 months (below) so seasons align,
        # letting the SH tropical catchments (the dominant tropical high-F_add region in Fig. 5)
        # enter the tropical panel instead of being excluded by the NH-only restriction.
        hemi_all = np.ones(len(north_all), dtype=bool)
    else:
        raise ValueError(f"hemisphere must be 'NH', 'SH', or 'merged', got '{hemisphere}'")

    def monthly(arr, mask):
        """Return the calendar-month climatology of arr averaged over the masked basins; when merged, Southern-Hemisphere basins are rolled by +6 months to season-align before averaging."""
        if hemisphere != "merged":
            daily = np.nanmean(arr[mask, :], axis=0)
            return pd.Series(daily, index=months).groupby(level=0).mean()
        # merged: per-basin calendar-month climatology, roll SH basins by +6 months to align
        # seasons (austral summer -> boreal-summer position), then average across all basins.
        idx = np.where(mask)[0]
        per_basin = np.full((len(idx), 12), np.nan)
        for j, b in enumerate(idx):
            clim = pd.Series(arr[b, :], index=months).groupby(level=0).mean().reindex(range(1, 13)).values
            per_basin[j] = np.roll(clim, 6) if south_all[b] else clim
        return pd.Series(np.nanmean(per_basin, axis=0), index=range(1, 13))

    present = [gid for gid in KG_MAJOR_NAMES
              if ((df_coord["kg_major_id"] == gid).values & hemi_all).any()
              and KG_MAJOR_NAMES[gid] not in skip_codes]
    ncols = 3
    nrows = int(np.ceil(len(present) / ncols))

    # --- common y-axis limits across all panels: left axis (P/PET/ET/F_add) and right axis (LAI) ---
    left_max, lai_min, lai_max = 0.0, np.inf, -np.inf
    for gid in present:
        mask = (df_coord["kg_major_id"] == gid).values & hemi_all
        for arr in (P, PET, ET, FADD):
            left_max = max(left_max, np.nanmax(monthly(arr, mask).values))
        lai_m = monthly(LAI, mask).values
        lai_min = min(lai_min, np.nanmin(lai_m))
        lai_max = max(lai_max, np.nanmax(lai_m))
    left_ylim = (0, left_ymax if left_ymax is not None else left_max * 1.05)
    lai_pad = 0.05 * (lai_max - lai_min)
    lai_ylim = (lai_min - lai_pad, lai_max + lai_pad)

    wspace, hspace = (1.5, 7) if single_ylabel else (6.5, 6)
    if wspace_override is not None:
        wspace = wspace_override
    if hspace_override is not None:
        hspace = hspace_override
    title_sz = TITLESIZE + 4 if single_ylabel else TITLESIZE
    ylab_sz = LABELSIZE + 1 if single_ylabel else LABELSIZE - 3
    tick_sz = TICKLABELSIZE + 3 if single_ylabel else TICKLABELSIZE - 1
    if title_size is not None:
        title_sz = title_size
    if label_size is not None:
        ylab_sz = label_size
    if tick_size is not None:
        tick_sz = tick_size
    xlab_sz = LABELSIZE - 2 if xlabel_size is None else xlabel_size
    fig, axs = uplt.subplots(nrows=nrows, ncols=ncols, figwidth=figwidth, top="8mm", share=False,
                             abc=True, abcloc="l", hspace=hspace, wspace=wspace)

    p_color, pet_color, et_color, fadd_color, lai_color = "#b8d4e8", "#f5a623", "#2c7bb6", "#d7191c", "#1a9641"
    h_pet = h_et = h_fadd = h_lai = None
    xlab = "Month (season-aligned)" if hemisphere == "merged" else "Month"

    for i, gid in enumerate(present):
        code = KG_MAJOR_NAMES[gid]
        mask = (df_coord["kg_major_id"] == gid).values & hemi_all
        n = int(mask.sum())

        col = i % ncols
        # left y-label on every panel, unless single_ylabel or left_label_first_col_only restrict
        # it to the leftmost column (the latter keeps fonts/spacing and the right LAI labels intact)
        show_left = (col == 0) if (single_ylabel or left_label_first_col_only) else True
        left_lbl = "P, PET, ET, $F_{add}$ (mm day$^{-1}$)" if show_left else ""
        lai_lbl = "LAI (normalized)" if (col == ncols - 1 or (not single_ylabel and not lai_label_right_only)) else ""

        ax = axs[i]
        ax.bar(range(1, 13), monthly(P, mask), color=p_color, alpha=0.7, width=0.7, zorder=1)
        h_pet, = ax.plot(range(1, 13), monthly(PET, mask), color=pet_color, marker='o',
                         markerfacecolor='none', linewidth=1.8, markersize=4)
        h_et, = ax.plot(range(1, 13), monthly(ET, mask), color=et_color, marker='s',
                        linewidth=1.8, markersize=4)
        h_fadd, = ax.plot(range(1, 13), monthly(FADD, mask), color=fadd_color, marker='x',
                          linewidth=1.8, markersize=5)
        ax.format(
            title=f"{code} (n={n})", titlesize=title_sz,
            xlabel=xlab, ylabel=left_lbl, ylim=left_ylim,
            xlabelsize=xlab_sz, ylabelsize=ylab_sz,
            xticklabelsize=tick_sz, yticklabelsize=tick_sz,
            grid=False,
        )
        ax2 = ax.alty(color=lai_color)
        h_lai, = ax2.plot(range(1, 13), monthly(LAI, mask), color=lai_color, marker='^',
                          linewidth=1.8, markersize=4)
        ax2.format(ylabel=lai_lbl, ylim=lai_ylim,
                   ylabelsize=ylab_sz, yticklabelsize=tick_sz)
        for sp in ax.spines.values():
            sp.set_linewidth(1.4)
        for sp in ax2.spines.values():
            sp.set_linewidth(1.4)

        if nh_sh_inset:
            nh_n = int((mask & north_all).sum())
            sh_n = int((mask & south_all).sum())
            axin = ax.inset_axes([0.11, 0.64, 0.27, 0.28], zoom=False)
            axin.barh([1, 0], [nh_n, sh_n], height=0.72, color=["#4C72B0", "#DD8452"],
                      edgecolor="k", lw=0.4, zorder=5)
            axin.set_xlim(0, max(nh_n, sh_n) * 1.60 + 1)
            axin.set_ylim(-0.6, 1.6)
            for yi, val in zip([1, 0], [nh_n, sh_n]):
                axin.text(val, yi, " " + str(val), ha="left", va="center", fontsize=tick_sz - 3, zorder=6)
            axin.set_yticks([1, 0])
            axin.set_yticklabels(["NH", "SH"], fontsize=tick_sz - 2)
            axin.set_xticks([])
            axin.tick_params(length=0, pad=1)
            for sp in axin.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.8)
            axin.patch.set_facecolor("white")
            axin.patch.set_alpha(0.85)
        if single_ylabel and col != 0:
            ax.tick_params(axis="y", labelleft=False)
        if single_ylabel and col != ncols - 1:
            ax2.tick_params(axis="y", labelright=False)

    for j in range(len(present), nrows * ncols):
        axs[j].axis("off")
        axs[j].format(abc=False, title="")

    legend_handles = [Patch(facecolor=p_color, alpha=0.7, label="P"), h_pet, h_et, h_fadd, h_lai]
    for h, lbl in zip(legend_handles, ["P", "PET", "ET", "$F_{add}$", "LAI"]):
        h.set_label(lbl)
    leg_size = LEGENDSIZE + 6 if single_ylabel else LEGENDSIZE
    fig.legend(handles=legend_handles, loc="b", ncols=5, frame=False,
               prop={"size": leg_size}, markerscale=1.6, handlelength=2.4, columnspacing=2.2)
    fig.save(out_path, dpi=200)
    print(f"Saved -> {out_path}")


# =============================================================================
# 3. ET / TWSA performance comparison (across the 25 seed x fold runs per experiment)
# =============================================================================
# =============================================================================
# 4. SHAP: attrs -> mean f_lat, for mlp_af and lstm_af only
#    (mlp_a's f_lat is a deterministic function of the same static attrs via a tiny MLP
#     -> SHAP would just re-derive that function, not scientifically informative)
# =============================================================================
def plot_shap_dependence_grid(exp, X_test, shap_array, r2, legend, legend_short, color_values,
                              color_label, out_path, n_features=5, cbar_vmax=None,
                              shap_iqr=None, rank_stats=None, r2_std=None, title=None,
                              bar_group_colors=None, group_legend=None, title_size=None):
    """Alternative layout of the SHAP dependence figure (Fig. 6 v2): the ranking bar chart (panel
    a) is a single tall column on the left spanning the full figure height, and the top-`n_features`
    dependence scatters (b, c, d, ...) fill a 2-row grid to its right, sharing one colorbar on the
    far right. Unlike plot_shap_dependence, panels b-f share a single y-axis range so the SHAP
    magnitudes are directly comparable. Colormap is batlowK; the per-point IQR whiskers and the
    scatter panels are drawn larger for readability."""
    if rank_stats is not None:
        rank_med, rank_q1, rank_q3 = (np.asarray(a, float) for a in rank_stats)
        rel_imp = rank_med
        full_order = np.argsort(rank_med)[::-1]
    else:
        mean_abs_shap = np.abs(shap_array).mean(axis=0)
        rel_imp = mean_abs_shap / mean_abs_shap.sum() * 100
        full_order = np.argsort(rel_imp)[::-1]
    feat_order = X_test.columns[full_order]
    dep_order = full_order if n_features is None else full_order[:n_features]
    dep_feat_order = X_test.columns[dep_order]

    cmap = uplt.Colormap("batlowK")
    vmax = float(cbar_vmax) if cbar_vmax is not None else round(float(np.nanpercentile(color_values, 95)), 1)

    # scatter grid: two rows to the right of the tall panel a; enough columns for n_features.
    # Reserve at least one empty cell for the colorbar: when the two rows would fill exactly (or
    # over-fill) with scatters, add one more column. n_features=5 (Fig. 6 v2) is unaffected.
    n_scatter = len(dep_feat_order)
    ncols_sc = int(np.ceil(n_scatter / 2))
    if 2 * ncols_sc <= n_scatter:
        ncols_sc += 1
    # widen the figure when many predictors are shown so each square stays legible (v2's 5-feature
    # / 3-column layout keeps its original 330mm).
    fw_mm = 330 if ncols_sc <= 3 else 100 * (ncols_sc + 1)
    fig = uplt.figure(figwidth=f"{fw_mm}mm", figheight="185mm", spanx=False, spany=False,
                      sharex=False, sharey=False, tight=True)
    # wider gutter between panel a and the first scatter column so panel a's bars never overlap
    # the "SHAP value" y-labels of panels b / e.
    gs = uplt.GridSpec(nrows=2, ncols=ncols_sc + 1, hspace=1.0,
                       wspace=[6.5] + [4.5] * (ncols_sc - 1))

    # --- panel a: ranking bar chart, tall left column spanning both rows ---
    ax = fig.add_subplot(gs[:, 0], abc=True, abcloc="l")
    bar_labels = [legend_short.get(f, f) for f in feat_order]
    bar_vals = rel_imp[full_order]
    bar_fc = [bar_group_colors[f] for f in feat_order] if bar_group_colors is not None else "slategray"
    xerr = None
    if rank_stats is not None:
        xerr = np.clip(np.vstack([bar_vals - rank_q1[full_order], rank_q3[full_order] - bar_vals]), 0, None)
    if isinstance(bar_fc, list):
        BAR_H = 0.8
        ypos = np.arange(len(bar_labels))
        for i, (v, c) in enumerate(zip(bar_vals, bar_fc)):
            for patch in ax.barh(ypos[i], v, fc=c).patches:
                patch.set_height(BAR_H)
                patch.set_y(ypos[i] - BAR_H / 2)
        if xerr is not None:
            ax.errorbar(bar_vals, ypos, xerr=xerr, fmt="none", ecolor="k", elinewidth=1.8, capsize=3)
        ax.format(yticks=ypos, yticklabels=bar_labels)
    elif xerr is not None:
        ax.barh(bar_labels, bar_vals, fc=bar_fc,
                xerr=xerr, error_kw=dict(ecolor="k", elinewidth=0.8, capsize=2))
    else:
        ax.barh(bar_labels, bar_vals, fc=bar_fc)
    ax.invert_yaxis()
    base_title = title if title is not None else EXP_LABELS[exp]
    r2_txt = f"R$^2$={r2:.2f}" if r2_std is None else f"R$^2$={r2:.2f}$\\pm${r2_std:.2f}"
    ax.format(title=f"{base_title}  ({r2_txt})",
              titlesize=(title_size if title_size is not None else LABELSIZE),
              xlabel="Relative contribution (%)", ylabel="",
              xlabelsize=LABELSIZE - 1, yticklabelsize=LABELSIZE, xticklabelsize=TICKLABELSIZE - 1,
              grid=False)
    ax_a = ax  # keep handle to the ranking panel for post-layout repositioning
    if group_legend is not None:
        from matplotlib.patches import Patch
        _abbr = {"Climate & aridity": "Climate", "Terrain": "Terrain", "Soil texture": "Soil",
                 "Vegetation & LAI": "Vegetation", "Groundwater": "Groundwater"}
        handles = [Patch(facecolor=c, edgecolor="k", lw=0.4, label=_abbr.get(n, n))
                   for n, c in group_legend]
        ax.legend(handles=handles, loc="lower right", ncols=1, frame=False,
                  prop={"size": LEGENDSIZE + 1}, handlelength=1.1, handleheight=1.1,
                  labelspacing=0.35, borderpad=0.3, handletextpad=0.5)

    # --- shared y-range across the dependence scatters (median SHAP +/- IQR whiskers) ---
    lo_all, hi_all = [], []
    for feat_name in dep_feat_order:
        j = X_test.columns.get_loc(feat_name)
        s = shap_array[:, j]
        if shap_iqr is not None:
            q1a, q3a = shap_iqr
            lo_all.append(np.nanmin(q1a[:, j])); hi_all.append(np.nanmax(q3a[:, j]))
        else:
            lo_all.append(np.nanmin(s)); hi_all.append(np.nanmax(s))
    ymin, ymax = float(np.nanmin(lo_all)), float(np.nanmax(hi_all))
    pad = 0.05 * (ymax - ymin)
    shared_ylim = (ymin - pad, ymax + pad)

    # --- b, c, ... dependence scatters, row-major in the columns right of panel a ---
    positions = [(r, c) for r in range(2) for c in range(1, ncols_sc + 1)][:n_scatter]
    sc = None
    scatter_axes = {}   # (row, col) -> axes, for post-layout alignment of panel a
    for panel_idx, feat_name in enumerate(dep_feat_order):
        row, col = positions[panel_idx]
        ax = fig.add_subplot(gs[row, col], abc=True, abcloc="l")
        scatter_axes[(row, col)] = ax
        j = X_test.columns.get_loc(feat_name)
        shap_feat = shap_array[:, j]
        feat_vals = X_test[feat_name].values
        ax.axhline(0, ls="--", color="orange", lw=1, zorder=0)
        if shap_iqr is not None:
            q1a, q3a = shap_iqr
            ylo = shap_feat - q1a[:, j]
            yhi = q3a[:, j] - shap_feat
            ax.errorbar(feat_vals, shap_feat, yerr=np.clip(np.vstack([ylo, yhi]), 0, None),
                        fmt="none", ecolor="0.6", elinewidth=0.6, alpha=0.7, zorder=1)
        sc = ax.scatter(feat_vals, shap_feat, c=color_values, cmap=cmap, s=20, vmin=0, vmax=vmax,
                        edgecolor="k", lw=0.15, zorder=2)
        ax.set_box_aspect(1)
        ax.format(
            xlabel=legend.get(feat_name, feat_name), xlabelsize=LABELSIZE - 1,
            ylabel="SHAP value" if col == 1 else "",
            ylim=shared_ylim,
            ylabelsize=LABELSIZE, xticklabelsize=TICKLABELSIZE - 1, yticklabelsize=TICKLABELSIZE - 1,
            grid=False,
        )

    # colorbar occupies the one empty grid cell (bottom-right) as a thin vertical bar, sized well
    # below the cell width (it does not need to fill the panel space). Freeze the tight layout so
    # the manually-placed colorbar axes is not shifted on save.
    if sc is not None:
        empty_row, empty_col = 1, ncols_sc
        host = fig.add_subplot(gs[empty_row, empty_col])
        host.axis("off")
        fig.canvas.draw()
        pos = host.get_position()
        for attr, val in (("_tight", False), ("_auto_layout", False), ("_tight_active", False)):
            try:
                setattr(fig, attr, val)
            except Exception:
                pass
        try:
            fig.set_layout_engine("none")
        except Exception:
            pass
        # ultraplot re-runs auto_layout() on every canvas draw (incl. savefig), which snaps gridspec
        # axes back to their cell -- neutralise it so the manual panel-a position below survives.
        fig.auto_layout = (lambda *a, **k: None)
        # align panel a's frame flush with the drawn scatter squares: top to panel b's frame top,
        # bottom to panel e's frame bottom. box_aspect=1 centres a square inside each cell (which is
        # taller than wide here), so get_position() returns the full cell, not the square -- compute
        # the square's vertical extent from the cell size and the physical figure aspect.
        fw_in, fh_in = fig.get_size_inches()

        def _square_yspan(a):
            """Return the (y0, y1) figure-fraction span of the largest square centered in axes a."""
            p = a.get_position()
            side = min((p.x1 - p.x0) * fw_in, (p.y1 - p.y0) * fh_in)  # physical square side (in)
            h = side / fh_in                                          # square height (fig fraction)
            cy = 0.5 * (p.y0 + p.y1)
            return cy - h / 2, cy + h / 2

        ax_b = scatter_axes.get((0, 1))
        ax_e = scatter_axes.get((1, 1))
        if ax_b is not None and ax_e is not None:
            b_top = _square_yspan(ax_b)[1]
            e_bot = _square_yspan(ax_e)[0]
            pa = ax_a.get_position()
            ax_a.set_in_layout(False)
            ax_a.set_position([pa.x0, e_bot, pa.width, b_top - e_bot])
        cw = 0.80 * pos.width          # horizontal bar length
        ch = 0.07 * pos.height         # thin
        cx = pos.x0 + (pos.width - cw) / 2
        cy = pos.y0 + 0.50 * pos.height
        cax = fig.add_axes([cx, cy, cw, ch])
        cbar = fig.colorbar(sc, cax=cax, extend="max", orientation="horizontal",
                            ticks=[0, round(vmax / 2, 1), round(vmax, 1)])
        cbar.set_label(color_label, size=LABELSIZE)
        cbar.ax.tick_params(labelsize=TICKLABELSIZE - 1)
        cbar.set_ticks([0, round(vmax / 2, 1), round(vmax, 1)])

    fig.savefig(out_path, dpi=200)
    print(f"Saved -> {out_path}")


def _oliver_pci_annual(monthly):
    """Precipitation Concentration Index of Oliver (1980) for one year of 12 monthly totals,
    PCI = 100 * sum(P_m^2) / (sum P_m)^2. It ranges from 8.3 (equal totals in every month) to 100
    (all precipitation in a single month). NaN for incomplete or dry years."""
    p = np.asarray(monthly, float)
    if p.size != 12 or not np.all(np.isfinite(p)) or p.sum() <= 0:
        return np.nan
    return 100.0 * np.sum(p ** 2) / np.sum(p) ** 2


def _compute_pci(result, start_year=2001):
    """Precipitation Concentration Index per basin following Oliver (1980): the annual PCI of every
    calendar year from start_year to the end of the record (2001-2020), computed from the 12 monthly
    precipitation totals, averaged over the years. Purely observational (P driver reconstructed from
    rain+snow), so it is identical regardless of which experiment's result dict is passed in."""
    idx_daily = pd.DatetimeIndex(result["idx_daily"])
    precip = _reconstruct_precip(result)  # [basins, days]
    if start_year is not None:
        keep = idx_daily.year >= start_year
        idx_daily, precip = idx_daily[keep], precip[:, keep]
    pci = np.full(precip.shape[0], np.nan)
    for i in range(precip.shape[0]):
        monthly = pd.Series(precip[i], index=idx_daily).resample("MS").sum(min_count=1)
        annual = [_oliver_pci_annual(m.values) for _, m in monthly.groupby(monthly.index.year)]
        pci[i] = np.nanmean(annual) if np.any(np.isfinite(annual)) else np.nan
    return pci

