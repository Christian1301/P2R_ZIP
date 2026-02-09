
# P2R-ZIP: Point-to-Region Counting with Zero-Inflatable Poisson Modeling

This repository contains the implementation of the **P2R-ZIP** model, an evolution of the *Point-to-Region* (P2R) architecture designed for crowd counting in complex scenarios. The project introduces a statistical classification head based on the **Zero-Inflated Poisson (ZIP)** distribution to improve semantic distinction between crowd areas and background.

## Model Architecture

The model integrates two main modules for joint scene analysis:

1.  **ZIP Head (Semantic Reasoning):** A statistical module that analyzes features extracted from the backbone (VGG16) to model pixel distribution. It produces two fundamental maps:
        * **$\pi$-map:** Represents the occupancy probability (crowd presence) for each region.
        * **$\lambda$-map:** Estimates the local crowd intensity based on bin regression.
2.  **P2R Head (Localization):** The final regression module that receives backbone features augmented with semantic information from the ZIP Head (concatenated $\pi$ and $\lambda$ channels).

![Architecture](readme_image/architecture.png)

### Fusion Strategies
The project supports different semantic integration modes to guide density regression:
* **Soft Fusion (Bypass Gating):** The $\pi$ and $\lambda$ maps are used as additional features, allowing the model to retain all backbone information and guide it with statistical "hints".
* **Hard Gating:** The $\pi$-map acts as a binary mask to actively suppress noise in regions identified as background.

## Training Protocol (Curriculum Learning)

Training follows a multi-stage strategy to ensure stable convergence and progressive learning of features:

* **Stage 1 (ZIP Training):** Isolated training of the **ZIP Head** using the Zero-Inflated Poisson NLL Loss to model scene sparsity. Run with:
    ```bash
    python train_stage1_zip.py --config config_shhb.yaml
    ```
* **Stage 2 (P2R Training):** Training of the **P2R** module with $\pi$ and $\lambda$ as additional inputsì. Run with:
    ```bash
    python train_stage2_p2r.py --config config_shhb.yaml
    ```
* **Stage 3 (Joint Fine-Tuning):** Joint training of the entire architecture using a **Composite Loss** that balances counting accuracy and correct semantic classification. Run with:
    ```bash
    python train_stage3_joint.py --config config_shhb.yaml
    ```

You can use the provided bash scripts to launch all stages sequentially:
```bash
./train_all.sh
```
or for specific datasets:
```bash
./train_all_jhu.sh
./train_all_qnrf.sh
```
The config file in the `configs/` folder determines the dataset and training parameters.

## Requirements and Installation

The code requires a Python 3.8+ environment with the following main libraries:
* PyTorch
* Seaborn / Matplotlib (for Confusion Matrix and map visualization)
* Scikit-learn

```bash
git clone https://github.com/Christian1301/p2r.git
cd p2r
pip install -r requirements.txt
```

## Launch the demo
To launch the demo, use the following commands:
```bash
cd files
pip install -r requirements.txt
python server.py
```