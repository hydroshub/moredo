import argparse
import pickle
import torch

def get_full_out(x_all, idx_daily, fold, seed, device, model_dir):
    import config
    from libs.model import HybridModel

    date_seq = torch.tensor([d.toordinal() for d in idx_daily], dtype=torch.long, device=device)

    model = HybridModel(config.PARAM_CONFIG, nn_attr_cols=config.ATTR_NN_COLS, passthrough_cols=config.ATTR_PASSTHROUGH_COLS, input_cols=config.INPUT_COLS).to(device)
    model.load_state_dict(torch.load(f"{model_dir}/best_model_seed{seed}_fold{fold}.pt", weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        full_out = model(x_all.to(device), date_seq=date_seq, mode='full', show_progress=True)

    return full_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=str, required=True, help="Path to config.py")
    parser.add_argument("--basin_list", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from libs.utils import load_config
    config = load_config(args.config)
    from libs.data import load_dataset_with_cache

    x_all, _, _, idx_daily, _, _ = load_dataset_with_cache(
        data_dir=config.DATA_DIR,
        basin_list_path=args.basin_list,
        attr_paths=config.ATTR_PATHS,
        input_cols=config.INPUT_COLS,
        target_daily_cols=config.TARGET_DAILY_COLS,
        target_monthly_cols=config.TARGET_MONTHLY_COLS,
        nn_attr_cols=config.ATTR_NN_COLS,
        start_year=config.INFER_START_YEAR,
        end_year=config.INFER_END_YEAR,
        passthrough_cols=config.ATTR_PASSTHROUGH_COLS,
    )

    full_out = get_full_out(x_all=x_all, idx_daily=idx_daily, fold=args.fold, seed=args.seed, device=args.device, model_dir=args.model_dir)

    torch.save(full_out, f'{args.model_dir}/results_fold{args.fold}_seed{args.seed}.pickle')