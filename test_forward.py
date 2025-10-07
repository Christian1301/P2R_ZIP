import torch
from models.p2r_zip_model import P2R_ZIP_Model
from losses.zip_nll import zip_nll
from losses.p2r_losses import p2r_density_mse

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Device: {device}")

    # Crea il modello
    model = P2R_ZIP_Model(
        backbone_name="vgg16_bn",
        pi_thresh=0.5,
        gate="multiply",
        upsample_to_input=True
    ).to(device)
    model.eval()

    # Batch fittizio di 2 immagini 3x256x256
    img = torch.randn(2, 3, 256, 256).to(device)

    with torch.no_grad():
        out = model(img)
        print("Output keys:", out.keys())
        for k, v in out.items():
            print(f"{k}: {tuple(v.shape)}")

    # Dummy ground truth
    # crea blocchi fittizi con la stessa forma di pi e lam
    fake_blocks = torch.randint(0, 3, out["pi"].shape).float().to(device)
    fake_points = [torch.tensor([[50, 80], [200, 150]]).float().to(device) for _ in range(2)]

    l_zip = zip_nll(out["pi"], out["lam"], fake_blocks)
    l_p2r = p2r_density_mse(out["density"], fake_points, sigma=4.0, count_l1_w=0.01)

    print(f"Loss ZIP = {l_zip.item():.4f}")
    print(f"Loss P2R = {l_p2r.item():.4f}")
    print("🎯 Test forward completato con successo.")
