# P2R_ZIP/train_utils.py
import os, torch
from torch.utils.tensorboard import SummaryWriter

def setup_experiment(exp_dir):
    """
    Crea directory esperimento e writer TensorBoard.
    """
    os.makedirs(exp_dir, exist_ok=True)
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def resume_if_exists(model, optimizer, exp_dir, device):
    last_ck = os.path.join(exp_dir, "last.pth")
    if os.path.isfile(last_ck):
        ckpt = torch.load(last_ck, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_val", float("inf"))
        print(f"[Resume] Ripreso da {last_ck} (epoch={start_epoch})")
        return start_epoch, best_loss
    return 1, float("inf")

def save_checkpoint(model, optimizer, epoch, val_loss, best_loss, exp_dir, is_best=False):
    os.makedirs(exp_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "best_val": best_loss,
    }
    torch.save(ckpt, os.path.join(exp_dir, "last.pth"))
    if is_best:
        torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
