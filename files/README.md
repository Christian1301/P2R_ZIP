# P2R-ZIP Crowd Counting Demo

A Claude-inspired chat interface for crowd counting inference using the P2R-ZIP architecture.

## Screenshot

Upload a crowd image → get estimated count + π-map, density maps, and detailed stats.

## Setup

### 1. File Structure

Place the `demo/` folder **inside** your P2R-ZIP project root:

```
P2R_ZIP/
├── models/
│   ├── p2r_zip_model.py
│   ├── backbone.py
│   ├── p2r_head.py
│   └── zip_head.py
├── datasets/
├── losses/
├── train_utils.py
├── config.yaml
├── config_shhb.yaml
├── config_qnrf.yaml
├── config_jhu.yaml
├── exp/
│   ├── shha_v15/
│   │   └── stage3_fusion_best.pth
│   ├── shhb/
│   │   └── stage3_fusion_best.pth (or stage2_bypass_best.pth)
│   ├── qnrf/
│   │   └── stage3_fusion_best.pth
│   └── jhu/
│       └── stage3_fusion_best.pth
└── demo/               ← THIS FOLDER
    ├── server.py
    ├── inference.py
    ├── index.html
    ├── requirements.txt
    └── README.md
```

### 2. Install Dependencies

```bash
cd demo
pip install -r requirements.txt
```

### 3. Checkpoint Paths

The demo looks for checkpoints at these paths (relative to project root):

| Model              | Config            | Checkpoint Path                            |
|--------------------|-------------------|--------------------------------------------|
| ShanghaiTech A     | config.yaml       | exp/shha_v15/stage3_fusion_best.pth        |
| ShanghaiTech B     | config_shhb.yaml  | exp/shhb/stage3_fusion_best.pth            |
| UCF-QNRF           | config_qnrf.yaml  | exp/qnrf/stage3_fusion_best.pth            |
| JHU-Crowd++        | config_jhu.yaml   | exp/jhu/stage3_fusion_best.pth             |

If a checkpoint is missing, the model will appear greyed out in the interface.

To change paths, edit `MODELS_REGISTRY` in `inference.py`.

### 4. Run

```bash
cd demo
python server.py
```

Then open **http://localhost:5000** in your browser.

## Usage

1. Select a model from the top bar (choose the one trained on the most similar dataset)
2. Click the upload button (↑) or drag & drop an image
3. The model will analyze the image and show:
   - **Estimated count** (soft fusion method)
   - **Stats**: raw count, hard mask count, π coverage, scale compensation
   - **Visualizations**: π occupancy map, raw density, soft fusion density
4. Click any visualization to see it full-screen
5. Upload more images to continue the analysis

## Architecture

```
Browser (index.html)  ←→  Flask (server.py)  →  Inference Engine (inference.py)
                                                        ↓
                                                 P2R_ZIP Model (your trained checkpoints)
```

- **Frontend**: Single HTML file, no build tools needed, Claude-inspired design
- **Backend**: Flask server with REST API
- **Engine**: Loads models lazily, caches them for fast subsequent inference
