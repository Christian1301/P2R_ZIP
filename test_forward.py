import torch
from models.p2r_zip_model import P2R_ZIP_Model
from losses.composite_loss import ZIPCompositeLoss
from losses.p2r_region_loss import P2RLoss  # Assicurati che il nome della classe sia corretto

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Device: {device}")

    # 1. Definire una configurazione di BIN fittizia per il test
    dummy_bins = [
        [0, 0], [1, 1], [2, 2], [3, 9999]
    ]
    dummy_bin_centers = [
        0.0, 1.0, 2.0, 4.5
    ]
    ZIP_BLOCK_SIZE = 16

    # 2. Creare il modello con la nuova firma
    model = P2R_ZIP_Model(
        bins=dummy_bins,
        bin_centers=dummy_bin_centers,
        backbone_name="vgg16_bn",
        pi_thresh=0.5,
        gate="multiply",
        upsample_to_input=True
    ).to(device)
    
    # Per ottenere tutti gli output, mettiamo il modello in modalità training
    model.train()

    # 3. Batch fittizio di 2 immagini 3x256x256
    img = torch.randn(2, 3, 256, 256).to(device)

    with torch.no_grad():
        out = model(img)
        print("--- Output del Modello (modalità training) ---")
        print("Output keys:", out.keys())
        for k, v in out.items():
            print(f"- {k}: {tuple(v.shape)}")
        print("-" * 20)

    # 4. Dummy ground truth per il calcolo delle loss
    # Mappa di densità a piena risoluzione (come dal dataloader)
    fake_density_map = torch.randn(2, 1, 256, 256).abs().to(device)
    # Lista di punti per la P2R loss
    fake_points = [torch.tensor([[50, 80], [200, 150]]).float().to(device) for _ in range(2)]

    # 5. Inizializzare e calcolare le loss correttamente
    criterion_zip = ZIPCompositeLoss(bins=dummy_bins, weight_ce=1.0, zip_block_size=ZIP_BLOCK_SIZE)
    criterion_p2r = P2RLoss() # Assumendo che non richieda parametri all'init

    loss_zip, loss_dict_zip = criterion_zip(out, fake_density_map)
    loss_p2r = criterion_p2r(out["p2r_density"], fake_points)

    print(f"Loss ZIP (composita) = {loss_zip.item():.4f}")
    for name, val in loss_dict_zip.items():
        print(f"  - {name}: {val:.4f}")
        
    print(f"Loss P2R (region) = {loss_p2r.item():.4f}")
    print("\n🎯 Test forward completato con successo.")