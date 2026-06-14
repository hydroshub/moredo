import argparse
import os
import pandas as pd
import numpy as np
import xarray as xr
import torch

# ----------------------------------------------------------
# Read all driver data for full year
# ----------------------------------------------------------
def safe_open(path, var_name=None):
    if var_name is not None:
        da = xr.open_dataset(path)[var_name]
    else:
        da = xr.open_dataarray(path)

    # Remove 'number' dim if present
    if 'number' in da.dims:
        da = da.isel(number=0)

    # Remove 'number' coordinate if still present
    if 'number' in da.coords:
        da = da.drop_vars('number')

    da['time'] = da['time'].dt.floor('D')

    # Normalize spatial dimension names to lat/lon
    rename_map = {}
    if 'latitude' in da.dims:
        rename_map['latitude'] = 'lat'
    if 'longitude' in da.dims:
        rename_map['longitude'] = 'lon'
    if rename_map:
        da = da.rename(rename_map)

    return da

def read_drivers_for_year(year):
    import config
    driver_list = []
    for month in range(1, 13):
        monthly_das = []
        for driver_key, entry in config.INFER_DRIVER_PATHS.items():
            if isinstance(entry, tuple):
                path_template, var_name = entry
            else:
                path_template, var_name = entry, None
            da = safe_open(path_template.format(year=year, month=month), var_name=var_name)
            da.name = config.DRIVER_COLS_MAP[driver_key]  # rename to INPUT_COLS column name
            monthly_das.append(da)
        driver_list.append(xr.merge(monthly_das))

    return xr.concat(driver_list, dim='time')

# ----------------------------------------------------------
# Spinup: cycle one year n times to warm up model states
# ----------------------------------------------------------
def run_spinup_cycles(spinup_year, n_cycles, model_path, attr_tensor, land_mask, out_dir, device, fold, seed):
    import config
    from libs.model import HybridModel
    print(f"\n===== Spinup: year {spinup_year} × {n_cycles} cycles =====")

    model = HybridModel(config.PARAM_CONFIG, nn_attr_cols=config.ATTR_NN_COLS, passthrough_cols=config.ATTR_PASSTHROUGH_COLS, input_cols=config.INPUT_COLS).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    driver_ds = read_drivers_for_year(spinup_year)
    driver_ds_flat = driver_ds.stack(space=("lat", "lon")).to_array().transpose("space", "time", "variable").values
    driver_tensor = torch.from_numpy(driver_ds_flat.astype(np.float32))[land_mask]

    num_time = driver_tensor.shape[1]
    attr_tensor_expand = attr_tensor.expand(-1, num_time, -1)
    data_extended = torch.cat([driver_tensor, attr_tensor_expand], dim=-1).to(device)

    dates = pd.date_range(f"{spinup_year}-01-01", f"{spinup_year}-12-31", freq='D')
    date_seq_tensor = torch.tensor([d.toordinal() for d in dates], dtype=torch.long, device=device)

    states, diagnos = None, None
    init_state_keys   = set(model.init_states(1).keys())
    init_diagnos_keys = set(model.init_diagnos(1).keys())

    for cycle in range(1, n_cycles + 1):
        print(f"  Cycle {cycle}/{n_cycles} ...", end=" ", flush=True)
        with torch.no_grad():
            full_out = model(data_extended, date_seq=date_seq_tensor, mode='full', show_progress=False,
                             states=states, diagnos=diagnos, log_to_cpu=False, compute_twsa=False)
        states  = {k: v[:, -1, :].clone() for k, v in full_out['states'].items()  if k in init_state_keys}
        diagnos = {k: v[:, -1, :].clone() for k, v in full_out['diagnos'].items() if k in init_diagnos_keys}
        del full_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("done")

    # Save final state so the main loop picks it up as a warm start for INFER_START_YEAR
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        'states':  {k: v.detach().cpu() for k, v in states.items()},
        'diagnos': {k: v.detach().cpu() for k, v in diagnos.items()},
    }
    torch.save(ckpt, os.path.join(out_dir, f"state_{spinup_year}_fold{fold}_seed{seed}.pt"))
    print(f"Spinup complete. State saved as state_{spinup_year}_fold{fold}_seed{seed}.pt")


# ----------------------------------------------------------
# Inference for one year
# ----------------------------------------------------------
def run_inference_for_year(year, model_path, attr_tensor, land_mask, n_space_total, lat_vals, lon_vals, out_dir, device, fold, seed):
    import config
    from libs.model import HybridModel
    print(f"\n===== Running year: {year} =====")

    # Load previous state if available
    prev_state_file = os.path.join(out_dir, f"state_{year-1}_fold{fold}_seed{seed}.pt")
    if os.path.exists(prev_state_file):
        print(f"Warm start from: {prev_state_file}")
        ckpt = torch.load(prev_state_file, map_location='cpu', weights_only=True)
        prev_states  = {k: v.to(device) for k, v in ckpt['states'].items()}
        prev_diagnos = {k: v.to(device) for k, v in ckpt['diagnos'].items()}
    else:
        print("No previous state found. Spin-up mode.")
        prev_states, prev_diagnos = None, None

    model = HybridModel(config.PARAM_CONFIG, nn_attr_cols=config.ATTR_NN_COLS, passthrough_cols=config.ATTR_PASSTHROUGH_COLS, input_cols=config.INPUT_COLS).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    driver_ds = read_drivers_for_year(year)
    driver_ds_flat = driver_ds.stack(space=("lat", "lon")).to_array().transpose("space", "time", "variable").values
    driver_tensor = torch.from_numpy(driver_ds_flat.astype(np.float32))[land_mask]  # skip ocean

    num_time = driver_tensor.shape[1]
    attr_tensor_expand = attr_tensor.expand(-1, num_time, -1)
    data_extended = torch.cat([driver_tensor, attr_tensor_expand], dim=-1).to(device)

    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq='D')
    date_seq_tensor = torch.tensor([d.toordinal() for d in dates], dtype=torch.long, device=device)

    with torch.no_grad():
        full_out = model(data_extended, date_seq=date_seq_tensor, mode='full', show_progress=True, states=prev_states, diagnos=prev_diagnos, log_to_cpu=True, compute_twsa=False)

    os.makedirs(out_dir, exist_ok=True)
    init_state_keys  = set(model.init_states(1).keys())
    init_diagnos_keys = set(model.init_diagnos(1).keys())
    ckpt = {
        'states':  {k: v[:, -1, :].detach().cpu().clone() for k, v in full_out['states'].items() if k in init_state_keys},
        'diagnos': {k: v[:, -1, :].detach().cpu().clone() for k, v in full_out['diagnos'].items() if k in init_diagnos_keys},
    }
    torch.save(ckpt, os.path.join(out_dir, f"state_{year}_fold{fold}_seed{seed}.pt"))
    print(f"Saved state for {year}")

    time_dim = dates
    n_lat, n_lon = len(lat_vals), len(lon_vals)

    # Build lat/lon grid dataset — reconstruct full spatial grid (NaN for ocean pixels)
    ds_dict = {}
    for group_name in ['fluxes', 'states', 'diagnos', 'params']:
        for var_name, var_data in full_out[group_name].items():
            arr_land = var_data.cpu().numpy().squeeze(-1)  # (n_land, time)
            arr_full = np.full((n_space_total, arr_land.shape[1]), np.nan, dtype=np.float32)
            arr_full[land_mask] = arr_land
            arr_2d = arr_full.reshape(n_lat, n_lon, -1)  # (lat, lon, time)
            ds_dict[f"{group_name}_{var_name}"] = (('lat', 'lon', 'time'), arr_2d)

    ds = xr.Dataset(
        ds_dict,
        coords={
            'lat':  lat_vals,
            'lon':  lon_vals,
            'time': time_dim,
        }
    )

    # Daily output — None: all vars; []: skip; [...]: specified vars only
    if config.INFER_SAVE_DAILY != []:
        ds_daily = ds if config.INFER_SAVE_DAILY is None else ds[[v for v in config.INFER_SAVE_DAILY if v in ds]]
        ds_daily.to_netcdf(
            os.path.join(out_dir, f"hydro_output_{year}_fold{fold}_seed{seed}_daily.nc"),
            encoding={var: {"zlib": True, "complevel": 4} for var in ds_daily.data_vars}
        )
        print(f"Saved daily output for {year}")

    # Monthly output — None: all vars; []: skip; [...]: specified vars only
    if config.INFER_SAVE_MONTHLY != []:
        ds_for_monthly = ds if config.INFER_SAVE_MONTHLY is None else ds[[v for v in config.INFER_SAVE_MONTHLY if v in ds]]
        ds_monthly = ds_for_monthly.assign_coords(time=pd.to_datetime(ds_for_monthly['time'].values)).groupby("time.month").mean(dim="time")
        ds_monthly.to_netcdf(
            os.path.join(out_dir, f"hydro_output_{year}_fold{fold}_seed{seed}_monthly.nc"),
            encoding={var: {"zlib": True, "complevel": 4} for var in ds_monthly.data_vars}
        )
        print(f"Saved monthly output for {year}")
    
    del full_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ----------------------------------------------------------
# Entry point
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=str, required=True, help="Path to config.py")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="hydro_outputs")
    parser.add_argument("--infer_start_year", type=int, default=None, help="Override config.INFER_START_YEAR")
    parser.add_argument("--infer_end_year",   type=int, default=None, help="Override config.INFER_END_YEAR")
    args = parser.parse_args()

    from libs.utils import load_config
    config = load_config(args.config)
    if args.infer_start_year is not None:
        config.INFER_START_YEAR = args.infer_start_year
    if args.infer_end_year is not None:
        config.INFER_END_YEAR = args.infer_end_year
    from libs.data import AttributeProcessor

    processor = AttributeProcessor(config.ATTR_NN_COLS, config.ATTR_PASSTHROUGH_COLS)

    # Load global grid attributes
    attr_ds = xr.open_dataset(config.INFER_ATTR_PATH)

    lat_vals = attr_ds['lat'].values
    lon_vals = attr_ds['lon'].values

    attr_ds_flat = attr_ds.stack(space=("lat", "lon"))
    attr_df_flat = attr_ds_flat.to_array().transpose("space", "variable").to_pandas()
    attr_df_flat = processor.transform(attr_df_flat)
    col_order = [f"{c}_normed" for c in config.ATTR_NN_COLS] + [f"{c}_raw" for c in config.ATTR_PASSTHROUGH_COLS]
    attr_np = attr_df_flat[col_order].values[:, np.newaxis, :]  # (n_space_total, 1, n_attrs)

    # Land mask: skip pixels where any attribute is NaN (ocean / missing)
    land_mask = ~np.isnan(attr_np[:, 0, :]).any(axis=-1)  # (n_space_total,)
    n_space_total = attr_np.shape[0]
    print(f"Land pixels: {land_mask.sum()} / {n_space_total}")

    attr_tensor = torch.from_numpy(attr_np[land_mask].astype(np.float32))

    model_path = f"{args.model_dir}/best_model_seed{args.seed}_fold{args.fold}.pt"
    device = torch.device(args.device)

    # Optional spinup: cycle one year n times, save state for warm start
    if getattr(config, 'INFER_SPINUP_YEAR', None) is not None:
        run_spinup_cycles(
            spinup_year=config.INFER_SPINUP_YEAR,
            n_cycles=config.INFER_SPINUP_CYCLES,
            model_path=model_path,
            attr_tensor=attr_tensor,
            land_mask=land_mask,
            out_dir=args.output_dir,
            device=device,
            fold=args.fold,
            seed=args.seed,
        )

    # Loop over years sequentially
    for year in range(config.INFER_START_YEAR, config.INFER_END_YEAR + 1):
        run_inference_for_year(
            year=year,
            model_path=model_path,
            attr_tensor=attr_tensor,
            land_mask=land_mask,
            n_space_total=n_space_total,
            lat_vals=lat_vals,
            lon_vals=lon_vals,
            out_dir=args.output_dir,
            device=device,
            fold=args.fold,
            seed=args.seed,
        )
