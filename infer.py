# P2R_ZIP/infer.py
import os, yaml, torch
from PIL import Image
import numpy as np

from models.p2r_zip_model import P2R_ZIP_Model
from data.adapters import _to_tensor

def to_numpy(img_t):
    x = img_t.detach().cpu().numpy()
    return x

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="percorso immagine")
    ap.add_argument("--ckpt", default="weights/joint.pth")
    ap.add_argument("--out_dir", default="pred_vis")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    device = torch.device(cfg["DEVICE"] if torch.cuda.is_available() else "cpu")

    # load img
    img = Image.open(args.img).convert("RGB")
    W, H = img.size
    t = _to_tensor(img)
    # normalizza come training
    mean = torch.tensor(cfg["DATA"]["NORM_MEAN"]).view(3,1,1)
    std  = torch.tensor(cfg["DATA"]["NORM_STD"]).view(3,1,1)
    t = (t - mean) / std
    t = t.unsqueeze(0).to(device)

    dataset_name = cfg["DATASET"]
    bin_cfg = cfg["BINS_CONFIG"][dataset_name]
    zip_head_cfg = cfg.get("ZIP_HEAD", {})
    zip_head_kwargs = {
        "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 0.5),
        "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
        "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
        "lambda_noise_std": zip_head_cfg.get("LAMBDA_NOISE_STD", 0.0),
    }

    model = P2R_ZIP_Model(
        bins=bin_cfg["bins"],
        bin_centers=bin_cfg["bin_centers"],
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        gate=cfg["MODEL"]["GATE"],
        upsample_to_input=cfg["MODEL"]["UPSAMPLE_TO_INPUT"],
        zip_head_kwargs=zip_head_kwargs,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    model.eval()

    with torch.no_grad():
        out = model(t)
        pi = out["logit_pi_maps"].softmax(dim=1)[:, 1:]
        dens = out["p2r_density"]
        count = dens.sum().item()

    os.makedirs(args.out_dir, exist_ok=True)
    # salva heatmap densità come npy per precisione
    np.save(os.path.join(args.out_dir, "density.npy"), to_numpy(dens[0,0]))
    # salva pi (griglia) come npy
    np.save(os.path.join(args.out_dir, "pi.npy"), to_numpy(pi[0,0]))

    print(f"Predicted count: {count:.2f}")
    print(f"Salvati: {args.out_dir}/density.npy e pi.npy")

if __name__ == "__main__":
    main()
