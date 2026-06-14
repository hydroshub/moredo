import math
import torch
import torch.nn as nn
from datetime import date
import config  # External configuration



# MLP: static attributes → parameter (time-invariant)
class StaticParamGenerator(nn.Module):
    def __init__(self, input_dim, output_keys, hidden_dim=16, dropout=0.3):
        super().__init__()
        self.output_keys = output_keys
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(output_keys))
        )

    def forward(self, static_input):
        raw_out = self.network(static_input)
        return dict(zip(self.output_keys, torch.split(raw_out, 1, dim=-1)))


class HydroProcess(nn.Module):
    def __init__(self):
        super().__init__()

    def rainsnow_partition(self, drivers, params, states, fluxes, diagnos):
        # Partition precipitation into snow and rain based on temperature
        t, p = drivers['T'], drivers['P']
        tsnow = params['snow_tsnow']
        train = params['snow_train']

        snow_frac = torch.where(t <= tsnow, 1.0,
                                torch.where(t >= train, 0.0,
                                            (train - t) / (train - tsnow)))

        fluxes['snow'] = p * snow_frac
        fluxes['rain'] = p - fluxes['snow']
        return drivers, params, states, fluxes, diagnos

    def snow_bucket(self, drivers, params, states, fluxes, diagnos):
        # Update snowpack by first adding snowfall, then melting
        t = drivers['T']
        fmt = params['snow_fmt']
        tmelt = 0.0

        # Add snowfall to snow storage first
        s_tmp = states['S_SNOW'] + fluxes['snow']

        # Compute potential melt
        potential_melt = torch.where(t > tmelt, (t - tmelt) * fmt, torch.zeros_like(t))

        # Limit melt to available snow
        melt = torch.min(s_tmp, potential_melt)

        # Update state and flux
        fluxes['melt'] = melt
        states['S_SNOW'] = s_tmp - melt
        return drivers, params, states, fluxes, diagnos

    def partition_available_water(self, drivers, params, states, fluxes, diagnos):
        # Partition water into:
        # - to_avai: fraction available to ecosystem functioning
        # - to_unavai: remainder bypassing to runoff
        total = fluxes['rain'] + fluxes['melt']
        fluxes['to_avai'] = total * params['split_avai']
        fluxes['to_unavai'] = total * params['split_unavai']
        return drivers, params, states, fluxes, diagnos

    def _priestley_taylor_pet(self, T_C, Rn, patm):
        """Priestley-Taylor PET (mm/day). T_C in °C, Rn in W/m², patm in Pa."""
        Rn       = torch.clamp(Rn, min=0.0)
        lambda_v = (2501.0 - 2.361 * T_C) * 1e3
        e_sat    = 0.6108 * torch.exp(17.27 * T_C / (T_C + 237.3))
        delta    = torch.clamp(4098.0 * e_sat / ((T_C + 237.3) ** 2), min=0.0)
        gamma    = 1013.0 * patm / (622.0 * lambda_v)
        pet      = 1.26 * delta * Rn / (delta + gamma + 1e-6) / lambda_v * 86400.0
        return torch.clamp(pet, min=0.0)

    def avai_bucket(self, drivers, params, states, fluxes, diagnos):
        # Update ecosystem-available soil water bucket
        s_tmp = states['S_AVAI'] + fluxes['to_avai']

        # Relative soil moisture (0–1) using fixed capacity threshold
        relmoist = torch.clamp(s_tmp / (params['avai_cap'] + 1e-6), 0.0, 1.0)
        ratio     = torch.clamp(relmoist / (params['avai_wetpoint'] + 1e-6), min=1e-3, max=1.0)
        f_wetness = torch.where(ratio >= 1.0, torch.ones_like(ratio), ratio ** params['avai_stress_exp'])

        # Vegetation control via LAI
        f_phenology = 1 - params['avai_beta'] * (1 - diagnos['LAI_normed']) ** 2

        pet = self._priestley_taylor_pet(drivers['T'], drivers['RAD'], params['patm'])
        diagnos['pet'] = pet

        # Actual ET limited by moisture and vegetation
        et = pet * f_wetness * f_phenology
        fluxes['et'] = torch.minimum(et, s_tmp)

        # Update state
        states['S_AVAI'] = torch.clamp((s_tmp - fluxes['et']),
                                       min=torch.tensor(0.0, dtype=params['avai_cap'].dtype, device=params['avai_cap'].device),
                                       max=params['avai_cap']
        )

        diagnos['rel_moist']   = torch.clamp(states['S_AVAI'] / (params['avai_cap'] + 1e-6), 0.0, 1.0)
        diagnos['f_wetness']   = f_wetness
        diagnos['f_phenology'] = f_phenology
        return drivers, params, states, fluxes, diagnos

    def fast_bucket(self, drivers, params, states, fluxes, diagnos):
        # Simulate fast flow bucket
        s_tmp = states['S_FAST'] + fluxes['to_unavai']

        # Quick flow response (e.g., surface runoff or rapid lateral flow)
        q_fast = params['fast_kf'] * s_tmp
        fluxes['q_fast'] = torch.clamp(q_fast, max=s_tmp)

        # Percolation to slower bucket
        fluxes['perc'] = torch.minimum(params['fast_perc'], s_tmp - fluxes['q_fast'])

        # Update fast bucket storage
        states['S_FAST'] = torch.clamp(s_tmp - fluxes['q_fast'] - fluxes['perc'], min=0.0)
        return drivers, params, states, fluxes, diagnos

    def slow_bucket(self, drivers, params, states, fluxes, diagnos):
        # Simulate slow flow bucket (e.g., groundwater or baseflow reservoir)
        s_tmp = states['S_SLOW'] + fluxes['perc']

        # Outflow from slow reservoir
        q_slow = params['slow_ks'] * s_tmp
        fluxes['q_slow'] = torch.clamp(q_slow, max=s_tmp)

        # Update slow bucket storage
        states['S_SLOW'] = s_tmp - fluxes['q_slow']
        return drivers, params, states, fluxes, diagnos

    def river_bucket(self, drivers, params, states, fluxes, diagnos):
        # Total runoff: sum of fast and slow flow
        fluxes['q_total'] = fluxes['q_fast'] + fluxes['q_slow']  # [B, 1]
        return drivers, params, states, fluxes, diagnos

class HybridModel(nn.Module):
    def __init__(self, param_config, nn_attr_cols, passthrough_cols, input_cols):
        super().__init__()

        self.param_config     = param_config
        self.nn_attr_cols     = list(nn_attr_cols)
        self.passthrough_cols = list(passthrough_cols) if passthrough_cols else []
        normed_attrs = [f"{c}_normed" for c in self.nn_attr_cols]
        raw_attrs    = [f"{c}_raw"    for c in self.passthrough_cols]
        self.all_cols = input_cols + normed_attrs + raw_attrs

        self.param_generators = nn.ModuleDict()   # static

        for pname, (_, mode, attrs) in param_config.items():
            if mode == 'static':
                self.param_generators[pname] = StaticParamGenerator(len(attrs), [pname])

        self.process = HydroProcess()

    
    def init_states(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        states = {
            'S_SNOW': torch.zeros(batch_size, 1, device=device),
            'S_AVAI': torch.zeros(batch_size, 1, device=device),
            'S_FAST': torch.zeros(batch_size, 1, device=device),
            'S_SLOW': torch.zeros(batch_size, 1, device=device),
        }
        return states

    def init_diagnos(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        diagnos = {
            'rel_moist': torch.zeros(batch_size, 1, device=device),
        }
        return diagnos


    def generate_params(self, x_attrs, drivers, states, diagnos):

        param_raw = {}
        for pname, (_, mode, attrs) in self.param_config.items():
            if mode == 'static':
                attr_input = torch.cat([x_attrs[f"{c}_normed"] for c in attrs], dim=-1)
                param_raw[pname] = self.param_generators[pname](attr_input)[pname]

        params = {}
        for k, ((low, high), _, __) in self.param_config.items():
            v_scaled = torch.sigmoid(param_raw[k]) * (high - low) + low
            params[k] = torch.clamp(v_scaled, min=low, max=high)

        if config.AVAI_CAP_MODE == "informed":
            lai_avg      = x_attrs[config.ATTR_COLS_MAP["LAI_AVG"] + "_raw"]   # raw passthrough
            slope_normed = x_attrs[config.ATTR_COLS_MAP["SLOPE"] + "_normed"] # normed NN input
            lai_scaled   = lai_avg / (lai_avg + 3)
            slope_scaled = 1 - (torch.sigmoid(slope_normed) - 0.5)
            params['avai_cap'] = params['avai_cap_base'] * lai_scaled * (1 + slope_scaled) + 50
        # "direct" mode: PARAM_CONFIG must define an "avai_cap" entry itself
        # (instead of "avai_cap_base") — see AVAI_CAP_MODE in config_base.py


        rel_moist  = torch.clamp(diagnos['rel_moist'], min=0.01, max=0.99)
        split_avai = (1.0 - rel_moist) ** params['split_k']
        params['split_avai'] = 0.01 + 0.99 * split_avai
        params['split_unavai'] = 1.0 - params['split_avai']

        # Atmospheric pressure from elevation (ele_mt_sav_raw is true elevation in metres)
        ele_m = torch.clamp(x_attrs["ele_mt_sav_raw"], min=0.0)
        params['patm'] = 101325.0 * ((1.0 - 0.0065 * ele_m / 288.15) ** 5.255)  # Pa

        return params
    

    def forward(self, x_seq, date_seq, mode='supervised', show_progress=False, states=None, diagnos=None, log_to_cpu=False, compute_twsa=True):
        
        B, T, _ = x_seq.shape
        device = x_seq.device

        # Unpack flat tensor to named dict — no hardcoded indices beyond this point
        x = {col: x_seq[:, :, i:i+1] for i, col in enumerate(self.all_cols)}

        # ---- ----- ----
        # Initialize logs
        # ---- ----- ----
        all_fluxes, all_states, all_diagnos, all_params = {}, {}, {}, {}
        all_tws, all_date = [], []

        if date_seq.ndim == 1:
            date_seq = date_seq.unsqueeze(0).repeat(B, 1)
    
        # ---- ----- ----
        # Initialize states
        # ---- ----- ----
        if states is None:
            states = self.init_states(B, device=device)
        if diagnos is None:
            diagnos = self.init_diagnos(B, device=device)

        # ---- ----- ----
        # Prepare normed LAI
        # ---- ----- ----
        fm = config.DRIVER_COLS_MAP

        # Time-varying LAI for phenology normalization
        lai_raw = x[fm["LAI"]]  # [B, T, 1]

        # Static LAI statistics for phenology normalization
        am = config.ATTR_COLS_MAP
        lai_max = x[am["LAI_MAX"] + "_raw"][:, 0:1, :]  # [B, 1, 1] — keepdim for broadcasting over T
        lai_min = x[am["LAI_MIN"] + "_raw"][:, 0:1, :]  # [B, 1, 1]
        lai_normed_seq = torch.clamp((lai_raw - lai_min) / (lai_max - lai_min + 1e-6), 0.0, 1.0)

        # Build attr lookup: normed (NN) + raw (passthrough, static → take t=0)
        x_attrs = {f"{c}_normed": x[f"{c}_normed"].mean(dim=1) for c in self.nn_attr_cols}
        for c in self.passthrough_cols:
            x_attrs[f"{c}_raw"] = x[f"{c}_raw"][:, 0, :]

        # ---- ----- ----
        # Time loop
        # ---- ----- ----
        time_iter = range(T)

        if show_progress:
            from tqdm import tqdm
            import sys
            time_iter = tqdm(time_iter, desc="Forward", ncols=100, miniters=100, file=sys.stderr)

        # Loop over timesteps
        for t in time_iter:
            # ---- drivers inputs ----
            drivers = {
                'P':   x[fm["P"]][:, t, :],
                'T':   x[fm["T"]][:, t, :],
                'RAD': x[fm["RAD_SW"]][:, t, :] + x[fm["RAD_LW"]][:, t, :],
                'LAI': x[fm["LAI"]][:, t, :],
            }
            diagnos['LAI_normed'] = lai_normed_seq[:, t, :]

            # ---- Generate parameters ----
            params = self.generate_params(x_attrs, drivers, states, diagnos)

            # ---- Run hydro process ----
            fluxes = {}
            for fn in [
                self.process.rainsnow_partition,
                self.process.snow_bucket,
                self.process.partition_available_water,
                self.process.avai_bucket,
                self.process.fast_bucket,
                self.process.slow_bucket,
                self.process.river_bucket,
            ]:
                drivers, params, states, fluxes, diagnos = fn(drivers, params, states, fluxes, diagnos)

            # ---- Update logs ----
            s_tws = states['S_SNOW'] + states['S_AVAI'] + states['S_FAST'] + states['S_SLOW']
            diagnos['S_TWS'] = s_tws
            if compute_twsa:
                all_tws.append(s_tws.unsqueeze(1))
                all_date.append(date_seq[:, t].unsqueeze(1))

            for k, v in fluxes.items():
                v_log = v.cpu().detach() if log_to_cpu else v
                all_fluxes.setdefault(k, []).append(v_log.unsqueeze(1))

            for k, v in states.items():
                v_log = v.cpu().detach() if log_to_cpu else v
                all_states.setdefault(k, []).append(v_log.unsqueeze(1))

            for k, v in params.items():
                v_log = v.cpu().detach() if log_to_cpu else v
                all_params.setdefault(k, []).append(v_log.unsqueeze(1))

            for k, v in diagnos.items():
                v_log = v.cpu().detach() if log_to_cpu else v
                all_diagnos.setdefault(k, []).append(v_log.unsqueeze(1))


        # ---- ----- ----
        # Final post-processing
        # ---- ----- ----
        all_fluxes  = {k: torch.cat(v, dim=1) for k, v in all_fluxes.items()}
        all_states  = {k: torch.cat(v, dim=1) for k, v in all_states.items()}
        all_diagnos = {k: torch.cat(v, dim=1) for k, v in all_diagnos.items()}
        all_params  = {k: torch.cat(v, dim=1) for k, v in all_params.items()}

        # ---- ----- ----
        # Aggregate TWS to monthly and compute TWSA anomaly
        # ---- ----- ----
        if compute_twsa:
            all_tws_tensor = torch.cat(all_tws, dim=1)  # [B, T, 1]
            all_date_tensor = torch.cat(all_date, dim=1)  # [B, T, 1]
            date_list = [date.fromordinal(int(d)) for d in all_date_tensor[0].cpu().numpy()]

            month_starts = [i for i in range(len(date_list)) if i == 0 or (date_list[i].month != date_list[i - 1].month)]
            month_starts.append(len(date_list))
            monthly_tws = [torch.nanmean(all_tws_tensor[:, s:e, :], dim=1) for s, e in zip(month_starts[:-1], month_starts[1:])]
            monthly_tensor = torch.stack(monthly_tws, dim=1)  # [B, M, 1]

            # ---- GRACE baseline reference: Jan 2004 – Dec 2009 ----
            month_dates = [date_list[i] for i in month_starts[:-1]]
            baseline_mask = [(d >= date(2004, 1, 1)) and (d <= date(2009, 12, 31)) for d in month_dates]
            baseline_mask = torch.tensor(baseline_mask, device=monthly_tensor.device, dtype=torch.bool)
            baseline_idx = baseline_mask.nonzero(as_tuple=False).squeeze(-1)

            baseline = monthly_tensor[:, baseline_idx, :].nanmean(dim=1, keepdim=True)
            anomaly_tensor = monthly_tensor - baseline

            all_diagnos['TWS_monthly'] = monthly_tensor
            all_diagnos['TWSA_monthly'] = anomaly_tensor

        # ---- ----- ----
        # Output selection
        # ---- ----- ----
        if mode == 'supervised':
            return {
                'q': all_fluxes['q_total'],
                'et': all_fluxes['et'],
                'swe': all_states['S_SNOW'],
                'twsa': all_diagnos['TWSA_monthly'],
            }
        elif mode == 'full':
            return {
                'fluxes':  all_fluxes,
                'states':  all_states,
                'diagnos': all_diagnos,
                'params':  all_params,
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")


    
