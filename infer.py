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

    # model
    model = P2R_ZIP_Model(
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        gate=cfg["MODEL"]["GATE"],
        upsample_to_input=cfg["MODEL"]["UPSAMPLE_TO_INPUT"]
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    model.eval()

    with torch.no_grad():
        out = model(t)
        pi = out["pi"]
        dens = out["density"]
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
