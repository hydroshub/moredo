import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import hashlib
import pickle
import config

class BaseSpatialCV:
    def __init__(self, basin_ids, num_folds=5, spinup_until=None, seed=42):
        self.basin_ids = np.array(sorted(list(set(basin_ids))))
        self.num_folds = num_folds
        self.seed = seed
        self.spinup_until = spinup_until
        self.folds = self._generate_folds()
        
    def _generate_folds(self):
        raise NotImplementedError("Should have a subclass")
    
    def get_splits(self, fold_idx, val_ratio=0.1):
        test_ids = self.folds[fold_idx].tolist()
        non_test_ids = [b for b in self.basin_ids if b not in test_ids]

        rng = np.random.default_rng(self.seed + fold_idx)
        shuffled_non_test = non_test_ids.copy()
        rng.shuffle(shuffled_non_test)

        num_val = int(len(shuffled_non_test) * val_ratio)
        num_val = max(num_val, 1)
        return {
            "train": shuffled_non_test[num_val:],
            "val": shuffled_non_test[:num_val],
            "test": test_ids
        }
    
    def print_summary(self, fold_idx, valid_ids, time_index):
        splits = self.get_splits(fold_idx)
        n_train = len([b for b in splits["train"] if b in valid_ids])
        n_val   = len([b for b in splits["val"]   if b in valid_ids])
        n_test  = len([b for b in splits["test"]  if b in valid_ids])
        years = np.array([ts.year for ts in time_index])
        spinup_years = sorted(set(years[years < self.spinup_until]))
        active_years = sorted(set(years[years >= self.spinup_until]))
        print("─" * 50)
        print(f"Cross-validation summary (Fold {fold_idx})")
        print(f">> Spinup years : {spinup_years[0]}–{spinup_years[-1]}  ({len(spinup_years)} years)")
        print(f">> Active years : {active_years[0]}–{active_years[-1]}  ({len(active_years)} years)")
        print(f">> Basins  train: {n_train} | val: {n_val} | test: {n_test}")


    def get_spatial_indices(self, fold_idx, valid_ids):
        splits = self.get_splits(fold_idx)
        return (
            [valid_ids.index(b) for b in splits["train"] if b in valid_ids],
            [valid_ids.index(b) for b in splits["val"] if b in valid_ids],
            [valid_ids.index(b) for b in splits["test"] if b in valid_ids]
        )
    
    def get_full_mask_dict(self, fold_idx, time_index):
        years = np.array([ts.year for ts in time_index])
        active_mask = years >= self.spinup_until
        return {
            "spinup_mask": ~active_mask, 
            "train_mask": active_mask, 
            "val_mask": active_mask, 
            "test_mask": active_mask
        }

class RandomSpatialCV(BaseSpatialCV):
    def _generate_folds(self):
        rng = np.random.default_rng(self.seed)
        shuffled = self.basin_ids.copy()
        rng.shuffle(shuffled)
        return np.array_split(shuffled, self.num_folds)


class PredefinedSpatialCV(BaseSpatialCV):
    """Spatial CV with externally defined test-set basin IDs per fold.

    fold_dict: {fold_idx: [basin_id, ...]}  — test basins for each fold.
    """
    def __init__(self, basin_ids, fold_dict, spinup_until=None, seed=42):
        self._fold_dict = fold_dict
        super().__init__(basin_ids, num_folds=len(fold_dict), spinup_until=spinup_until, seed=seed)

    def _generate_folds(self):
        return [np.array(self._fold_dict[i]) for i in sorted(self._fold_dict.keys())]


class UnifiedDataset(Dataset):
    def __init__(self, x, y, z, mask_daily, mask_monthly):
        self.x = x
        self.y = y.clone()
        self.z = z.clone()
        self.y[:, ~mask_daily, :] = float("nan")
        self.z[:, ~mask_monthly, :] = float("nan")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.z[i]

class AttributeProcessor:
    """Apply fixed normalization to static attributes.

    x_normed = (x - offset) / scaler  using config.ATTR_NORM.
    Stateless — no fitting required, safe for global extrapolation.
    """
    def __init__(self, nn_attr_cols, passthrough_cols=None):
        self.nn_attr_cols     = list(nn_attr_cols)
        self.passthrough_cols = list(passthrough_cols) if passthrough_cols else []

    def compute_landcover(self, df):
        """Aggregate raw GLC percent columns into cover_* fractions (percent scale)."""
        df_proc = df.copy()
        try:
            df_proc['cover_forest'] = df_proc[[f'glc_pc_s{str(i).zfill(2)}' for i in [1,2,3,4,5,6,7,8,9,10]]].sum(axis=1)
            df_proc['cover_shrub']  = df_proc[[f'glc_pc_s{str(i).zfill(2)}' for i in [11,12,15]]].sum(axis=1)
            df_proc['cover_grass']  = df_proc[[f'glc_pc_s{str(i).zfill(2)}' for i in [13,14]]].sum(axis=1)
            df_proc['cover_crop']   = df_proc[[f'glc_pc_s{str(i).zfill(2)}' for i in [16,17,18]]].sum(axis=1)
            df_proc['cover_others'] = df_proc[[f'glc_pc_s{str(i).zfill(2)}' for i in [19,20,21,22]]].sum(axis=1)
        except KeyError:
            pass  # global grid path — cover_* columns already present
        return df_proc

    def transform(self, df):
        df_proc = self.compute_landcover(df)

        out = {}
        for c in self.passthrough_cols:
            out[f"{c}_raw"] = df_proc[c].values
        for c in self.nn_attr_cols:
            col = df_proc[c].copy()
            if "ele" in c.lower():
                col = np.sqrt(col.clip(lower=0, upper=5000))
            elif "area" in c.lower():
                col = np.log1p(col.clip(lower=0))
            offset, scaler = config.ATTR_NORM[c]
            out[f"{c}_normed"] = (col.values - offset) / scaler
        return pd.DataFrame(out, index=df_proc.index)


def load_attributes(attr_paths, basin_ids):

    if isinstance(attr_paths, str):
        attr_paths = [attr_paths]

    attr_dfs = []
    for path in attr_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Attribute file not found: {path}")
        df = pd.read_csv(path, index_col='gauge_id')
        attr_dfs.append(df)

    attr_df = pd.concat(attr_dfs, axis=1, join='inner')

    missing = [bid for bid in basin_ids if bid not in attr_df.index]
    if missing:
        raise ValueError(f"Missing basin(s) in attributes: {missing}")

    attr_df = attr_df.loc[basin_ids].copy()

    return attr_df


def load_dataset(
    data_dir,
    basin_list_path,
    attr_paths,
    input_cols,
    target_daily_cols,
    target_monthly_cols,
    start_year=2001,
    end_year=2020
):
    basin_ids = pd.read_csv(basin_list_path, header=None)[0].to_list()
    attr_df_raw = load_attributes(attr_paths, basin_ids)

    x_drivers_list, y_all, z_all, valid_ids, skipped_ids = [], [], [], [], []
    idx_daily, idx_monthly = None, None

    for basin_id in tqdm(basin_ids, desc="Loading basins", file=sys.stderr):
        path = os.path.join(data_dir, f"{basin_id}_merged.csv")
        if not os.path.exists(path) or basin_id not in attr_df_raw.index:
            skipped_ids.append(basin_id)
            continue

        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.loc[str(start_year):str(end_year)]

            if df.empty:
                skipped_ids.append(basin_id)
                continue

            if idx_daily is None:
                idx_daily = df.index
            elif not df.index.equals(idx_daily):
                print(f"Skipping {basin_id}: daily index mismatch")
                skipped_ids.append(basin_id)
                continue

            x_drivers = df[input_cols].values
            y = df[target_daily_cols].values
            y[y < 0] = np.nan
            i_swe = target_daily_cols.index(config.TARGET_COLS_MAP["swe"])
            i_q   = target_daily_cols.index(config.TARGET_COLS_MAP["q"])
            y[:, i_swe][y[:, i_swe] <= 0.1] = np.nan
            y[:, i_q][y[:, i_q] >= 100] = np.nan

            df_monthly = df[target_monthly_cols].resample("ME").first()
            if idx_monthly is None:
                idx_monthly = df_monthly.index
            else:
                df_monthly = df_monthly.reindex(idx_monthly)

            z = df_monthly.values

            if x_drivers.shape[0] == y.shape[0]:
                x_drivers_list.append(torch.tensor(x_drivers, dtype=torch.float32))
                y_all.append(torch.tensor(y, dtype=torch.float32))
                z_all.append(torch.tensor(z, dtype=torch.float32))
                valid_ids.append(basin_id)
            else:
                print(f"Skipping {basin_id}: shape mismatch")
                skipped_ids.append(basin_id)

        except Exception as e:
            print(f"Error loading {basin_id}: {e}")
            skipped_ids.append(basin_id)
            continue

    if not x_drivers_list:
        raise ValueError("No valid basins found.")

    x_drivers_all = torch.stack(x_drivers_list)  # [B, T, len(input_cols)]
    y_all = torch.stack(y_all)
    z_all = torch.stack(z_all)

    # Raw attrs for valid basins only (all columns, before any processing)
    attr_raw = attr_df_raw.loc[valid_ids].copy()

    # Fill missing LAI stats from time-series if not present in attribute files
    lai_col = config.DRIVER_COLS_MAP.get("LAI")
    lai_stats = {
        "LAI_AVG": config.ATTR_COLS_MAP.get("LAI_AVG"),
        "LAI_MAX": config.ATTR_COLS_MAP.get("LAI_MAX"),
        "LAI_MIN": config.ATTR_COLS_MAP.get("LAI_MIN"),
    }
    missing_stats = [col for col in lai_stats.values() if col and col not in attr_raw.columns]
    if missing_stats and lai_col and lai_col in input_cols:
        lai_idx = input_cols.index(lai_col)
        lai_ts = x_drivers_all[:, :, lai_idx].numpy()  # [B, T]
        n_days = lai_ts.shape[1]
        n_years = n_days / 365.25
        if n_years < 1.0:
            raise ValueError(
                f"Cannot compute LAI statistics: time series is only {n_days} days "
                f"({n_years:.2f} years). At least 1 full year is required."
            )
        warnings.warn(
            f"[LAI stats] {missing_stats} not found in attribute files — "
            f"computing from LAI time series ({n_days} days, {n_years:.1f} years). "
            f"Results may differ from basin-level climatology."
        )
        if lai_stats["LAI_AVG"] in missing_stats:
            attr_raw[lai_stats["LAI_AVG"]] = np.nanmean(lai_ts, axis=1)
        if lai_stats["LAI_MAX"] in missing_stats:
            attr_raw[lai_stats["LAI_MAX"]] = np.nanpercentile(lai_ts, 99, axis=1)
        if lai_stats["LAI_MIN"] in missing_stats:
            attr_raw[lai_stats["LAI_MIN"]] = np.nanpercentile(lai_ts, 1, axis=1)

    return x_drivers_all, attr_raw, y_all, z_all, idx_daily, idx_monthly, valid_ids


def build_x_all(x_drivers, attr_raw, nn_attr_cols, passthrough_cols=None):
    """Apply preprocessing and concatenate drivers + attrs into x_all.

    Output column order: [f"{c}_normed" for c in nn_attr_cols] + [f"{c}_raw" for c in passthrough_cols]
    This matches HybridModel.all_cols (after input_cols).
    """
    passthrough_cols = list(passthrough_cols) if passthrough_cols else []
    attr_df = AttributeProcessor(nn_attr_cols=nn_attr_cols, passthrough_cols=passthrough_cols).transform(attr_raw)
    col_order = [f"{c}_normed" for c in nn_attr_cols] + [f"{c}_raw" for c in passthrough_cols]
    attr_np = attr_df[col_order].values.astype(np.float32)           # [B, D_attr]
    T = x_drivers.shape[1]
    attr_tensor = torch.tensor(attr_np, dtype=torch.float32).unsqueeze(1).expand(-1, T, -1).contiguous()
    x_all = torch.cat([x_drivers, attr_tensor], dim=-1)             # [B, T, D_input + D_attr]
    return x_all


def get_config_hash(config_dict):
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()

def load_dataset_with_cache(
    data_dir,
    basin_list_path,
    attr_paths,
    input_cols,
    target_daily_cols,
    target_monthly_cols,
    nn_attr_cols,
    start_year,
    end_year,
    cache_dir="cache",
    passthrough_cols=None,
):
    # Hash covers only raw data sources — NOT attr_cols or preprocessing logic.
    # Preprocessing is always re-applied on load, so changing it never needs a cache bust.
    config_hash = get_config_hash({
        "data_dir": data_dir,
        "basin_list_path": basin_list_path,
        "attr_paths": attr_paths,
        "input_cols": input_cols,
        "target_daily_cols": target_daily_cols,
        "target_monthly_cols": target_monthly_cols,
        "start_year": start_year,
        "end_year": end_year,
    })

    os.makedirs(cache_dir, exist_ok=True)
    array_cache_path = os.path.join(cache_dir, f"cached_dataset_{config_hash}.npz")
    meta_cache_path  = os.path.join(cache_dir, f"cached_meta_{config_hash}.pkl")

    if os.path.exists(array_cache_path) and os.path.exists(meta_cache_path):
        npzfile = np.load(array_cache_path, allow_pickle=True)
        x_drivers  = torch.tensor(npzfile["x_drivers"],  dtype=torch.float32)
        y_all      = torch.tensor(npzfile["y_all"],       dtype=torch.float32)
        z_all      = torch.tensor(npzfile["z_all"],       dtype=torch.float32)
        idx_daily   = pd.to_datetime(npzfile["idx_daily"])
        idx_monthly = pd.to_datetime(npzfile["idx_monthly"])

        with open(meta_cache_path, "rb") as f:
            meta = pickle.load(f)
        valid_ids   = meta["valid_ids"]
        attr_raw    = pd.DataFrame(meta["attr_raw_values"],
                                   index=valid_ids,
                                   columns=meta["attr_raw_columns"])
        print(f"[Cache] Loaded raw dataset: {array_cache_path}")
    else:
        x_drivers, attr_raw, y_all, z_all, idx_daily, idx_monthly, valid_ids = load_dataset(
            data_dir=data_dir,
            basin_list_path=basin_list_path,
            attr_paths=attr_paths,
            input_cols=input_cols,
            target_daily_cols=target_daily_cols,
            target_monthly_cols=target_monthly_cols,
            start_year=start_year,
            end_year=end_year,
        )
        np.savez_compressed(array_cache_path,
                            x_drivers=x_drivers.numpy(),
                            y_all=y_all.numpy(),
                            z_all=z_all.numpy(),
                            idx_daily=idx_daily.astype(str),
                            idx_monthly=idx_monthly.astype(str))
        attr_raw_numeric = attr_raw.select_dtypes(include=[np.number])
        with open(meta_cache_path, "wb") as f:
            pickle.dump({
                "valid_ids": valid_ids,
                "attr_raw_values": attr_raw_numeric.values,
                "attr_raw_columns": list(attr_raw_numeric.columns),
            }, f)
        print(f"[Cache] Saved raw dataset: {array_cache_path}")

    x_all = build_x_all(x_drivers, attr_raw, nn_attr_cols, passthrough_cols=passthrough_cols)

    return x_all, y_all, z_all, idx_daily, idx_monthly, valid_ids
