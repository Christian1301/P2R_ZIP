import os, yaml, random, numpy as np, torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from losses.zip_nll import zip_nll
from losses.p2r_losses import p2r_density_mse
from train_utils import resume_if_exists, save_checkpoint, setup_experiment

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def evaluate_joint(model, loader, device, cfg):
    model.eval(); total = 0.0
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            blocks = batch["zip_blocks"].to(device)
            points = [p.to(device) for p in batch["points"]]
            out = model(img)
            l_zip = zip_nll(out["pi"], out["lam"], blocks)
            l_p2r = p2r_density_mse(out["density"], points,
                                    sigma=cfg["LOSS"]["P2R_SIGMA"],
                                    count_l1_w=cfg["LOSS"]["COUNT_L1_W"])
            total += (l_zip + cfg["LOSS"]["JOINT_ALPHA"] * l_p2r).item()
    model.train(); return total / len(loader)

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    set_seed(cfg["SEED"])
    device = torch.device(cfg["DEVICE"] if torch.cuda.is_available() else "cpu")

    Dataset = get_dataset(cfg["DATASET"])
    train_ds = Dataset(cfg["DATA"]["ROOT"], split=cfg["DATA"]["TRAIN_SPLIT"], block_size=cfg["DATA"]["ZIP_BLOCK_SIZE"])
    val_ds = Dataset(cfg["DATA"]["ROOT"], split=cfg["DATA"]["VAL_SPLIT"], block_size=cfg["DATA"]["ZIP_BLOCK_SIZE"])
    dl_train = DataLoader(train_ds, batch_size=cfg["OPTIM"]["BATCH_SIZE"], shuffle=True,
                          num_workers=cfg["OPTIM"]["NUM_WORKERS"], drop_last=True)
    dl_val = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = P2R_ZIP_Model(
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        gate=cfg["MODEL"]["GATE"],
        upsample_to_input=cfg["MODEL"]["UPSAMPLE_TO_INPUT"]
    ).to(device)

    ck_pre = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"] + "_p2r", "best_model.pth")
    if not os.path.isfile(ck_pre):
        ck_pre = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"] + "_zip", "best_model.pth")
    assert os.path.isfile(ck_pre), "Serve almeno un checkpoint valido."
    model.load_state_dict(torch.load(ck_pre, map_location="cpu"), strict=False)

    # === Optimizer con LR differenziato ===
    params = [
        {"params": model.backbone.parameters(), "lr": cfg["OPTIM"].get("LR_BACKBONE", cfg["OPTIM"]["LR"])},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n],
         "lr": cfg["OPTIM"]["LR"]}
    ]
    opt = torch.optim.AdamW(params, weight_decay=cfg["OPTIM"]["WEIGHT_DECAY"])

    # === Scheduler: warm-up + cosine annealing ===
    def warmup_lambda(epoch):
        warmup_epochs = cfg["OPTIM"].get("WARMUP_EPOCHS", 0)
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        return 1.0
    warmup = LambdaLR(opt, lr_lambda=warmup_lambda)
    cosine = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2)

    exp_dir = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"] + "_joint")
    writer = setup_experiment(exp_dir)
    start_ep, best_val = (1, float("inf"))
    if cfg["OPTIM"]["RESUME_LAST"]:
        start_ep, best_val = resume_if_exists(model, opt, exp_dir, device)

    epochs = cfg["OPTIM"]["EPOCHS"]
    alpha = float(cfg["LOSS"]["JOINT_ALPHA"])

    for ep in range(start_ep, epochs + 1):
        total = 0.0
        pbar = tqdm(dl_train, desc=f"[JOINT] Epoch {ep}/{epochs}")
        for batch in pbar:
            img = batch["image"].to(device)
            blocks = batch["zip_blocks"].to(device)
            points = [p.to(device) for p in batch["points"]]
            out = model(img)
            l_zip = zip_nll(out["pi"], out["lam"], blocks)
            l_p2r = p2r_density_mse(out["density"], points,
                                    sigma=cfg["LOSS"]["P2R_SIGMA"],
                                    count_l1_w=cfg["LOSS"]["COUNT_L1_W"])
            loss = l_zip + alpha * l_p2r
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train = total / len(dl_train)
        writer.add_scalar("train/loss", avg_train, ep)
        writer.add_scalar("lr/backbone", opt.param_groups[0]["lr"], ep)
        writer.add_scalar("lr/decoder", opt.param_groups[1]["lr"], ep)

        if ep % cfg["OPTIM"]["VAL_INTERVAL"] == 0 or ep == epochs:
            val_loss = evaluate_joint(model, dl_val, device, cfg)
            writer.add_scalar("val/loss", val_loss, ep)
            is_best = val_loss < best_val
            if is_best: best_val = val_loss
            save_checkpoint(model, opt, ep, val_loss, best_val, exp_dir, is_best=is_best)
            print(f"Val: {val_loss:.4f} | Best: {best_val:.4f}")

        # === Step scheduler ===
        if ep <= cfg["OPTIM"].get("WARMUP_EPOCHS", 0):
            warmup.step()
        else:
            cosine.step()

    writer.close()

if __name__ == "__main__":
    main()
