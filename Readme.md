# P2R-ZIP: Crowd Counting with Zero-Inflated Poisson + Point-to-Region Refinement

---

### 📘 Descrizione del progetto

**P2R-ZIP** è un framework di *crowd counting* che unisce due approcci complementari:

1. **ZIP (Zero-Inflated Poisson)** — Modello statistico per stimare la presenza e l’intensità di persone a livello di blocchi.  
2. **P2R (Point-to-Region)** — Rete di raffinamento che genera una mappa di densità precisa a livello pixel.

L’obiettivo è creare un’architettura che prima individua le regioni dove ci sono persone (*coarse estimation*), e poi le raffina in modo dettagliato (*fine estimation*).  
Questo è ottenuto tramite un flusso sequenziale:
Input → Backbone CNN → ZIP Head → Mask π → P2R Head → Density Map

---

## 🧩 Architettura

- **Backbone condiviso:** es. VGG16-BN o ResNet50 pre-addestrato su ImageNet.
- **ZIP Head:** calcola per ogni blocco:
  - \( \pi \): probabilità che il blocco contenga persone,
  - \( \lambda \): valore atteso di conteggio (Poisson rate).
- **P2R Head:** prende le feature mascherate dalla ZIP Head e produce una mappa di densità raffinata.

---

## ⚙️ Struttura del repository

P2R_ZIP/
├─ models/
│ ├─ backbone.py # Backbone CNN
│ ├─ zip_head.py # Testa ZIP (π, λ)
│ ├─ p2r_head.py # Testa P2R
│ └─ p2r_zip_model.py # Architettura combinata ZIP→P2R
│
├─ datasets/
│ ├─ base_dataset.py # Classe base per caricamento immagini e punti
│ ├─ jhu.py, shha.py,
│ ├─ ucf_qnrf.py, nwpu.py # Dataset supportati
│
├─ losses/
│ ├─ zip_nll.py # Loss Zero-Inflated Poisson NLL
│ └─ p2r_losses.py # Loss MSE + opzionale L1 sul conteggio
│
├─ train_utils.py # Resume, checkpoint, TensorBoard writer
│
├─ train_stage1_zip.py # Fase 1: pre-training ZIP
├─ train_stage2_p2r.py # Fase 2: training P2R con ZIP congelato
├─ train_stage3_joint.py # Fase 3: fine-tuning congiunto
│
├─ infer.py # Inferenza end-to-end ZIP→P2R
├─ config.yaml # Configurazione completa esperimento
└─ README.md


---

## 🧠 Perché ci sono **tre file di training**

Il modello è addestrato in **tre stadi progressivi** per garantire stabilità e specializzazione dei moduli:

### 🩵 1️⃣ Stage 1 — Pre-training ZIP
- Allena **solo** il backbone e la ZIP Head.
- Loss: *Zero-Inflated Poisson NLL*.
- Obiettivo: imparare a identificare blocchi contenenti persone.
- Output: `exp/<run_name>_zip/best_model.pth`.

### 💙 2️⃣ Stage 2 — Training P2R
- Carica ZIP pre-addestrata e **la congela**.
- Allena **solo la P2R Head** sulle annotazioni puntuali.
- Loss: *MSE sulla mappa di densità + opzionale L1 sul conteggio*.
- Output: `exp/<run_name>_p2r/best_model.pth`.

### 💜 3️⃣ Stage 3 — Joint Fine-tuning
- Sblocca tutto il modello.
- Loss combinata:
  \[
  L_{total} = L_{ZIP} + \alpha L_{P2R}
  \]
- Ottimizza coerenza e precisione globale.
- Output: `exp/<run_name>_joint/best_model.pth`.

---

## ⚖️ Approfondimento: la **Loss combinata**

Durante il fine-tuning congiunto (Stage 3), la rete viene ottimizzata con una **loss ibrida** che bilancia due obiettivi:

\[
L_{total} = L_{ZIP} + \alpha \, L_{P2R}
\]

dove:

### 🔹 1. \( L_{ZIP} \): Zero-Inflated Poisson Loss
Serve a modellare i **conteggi per blocco**.  
Ogni blocco ha due parametri:
- \( \pi \): probabilità che il blocco sia vuoto (nessuna persona),
- \( \lambda \): intensità media (Poisson rate) se il blocco è occupato.

La loss NLL per il blocco \( i \) è:
\[
L_{ZIP} = - \log \left[ \pi_i \mathbf{1}_{\{c_i=0\}} + (1-\pi_i) e^{-\lambda_i} \frac{\lambda_i^{c_i}}{c_i!} \right]
\]

In sintesi:
- Se il blocco è vuoto, la rete è premiata se \( \pi_i \) è alto.
- Se è occupato, la rete è premiata se \( \lambda_i \) stima correttamente il conteggio.

Questa formulazione permette di gestire dataset **sbilanciati**, in cui la maggior parte dei blocchi è priva di persone.

---

### 🔹 2. \( L_{P2R} \): Point-to-Region Loss
Serve a **raffinare la mappa di densità** a livello di pixel.

Viene calcolata come:
\[
L_{P2R} = \frac{1}{HW} \sum_{x,y} (D_{pred}(x,y) - D_{gt}(x,y))^2 + \beta \, \left| \sum D_{pred} - \sum D_{gt} \right|
\]

dove:
- \( D_{pred} \): mappa di densità predetta,
- \( D_{gt} \): mappa generata dai punti annotati con un kernel gaussiano (σ definito in `config.yaml`),
- \( \beta \): coefficiente del termine L1 sul conteggio totale (parametro `COUNT_L1_W`).

Il primo termine (MSE) forza la rete a replicare la forma della mappa di densità,  
il secondo (L1) mantiene il **conteggio totale coerente** con le annotazioni.

---

### 🔹 3. Ruolo di **α (alpha)**

Il parametro `α` (`JOINT_ALPHA` nel file di configurazione) **bilancia l’importanza** tra la parte ZIP e la parte P2R della loss totale:

- `α < 1` → priorità al conteggio globale (ZIP prevale)  
  → utile per dataset **sparsi** (es. NWPU-Crowd).  
- `α = 1` → bilanciamento standard,  
  → consigliato per dataset **equilibrati** (es. ShanghaiTechA).  
- `α > 1` → priorità alla precisione locale (P2R prevale)  
  → utile per dataset **densi** (es. JHU-CROWD, UCF-QNRF).

In pratica, **α controlla la scala di attenzione**:  
se la rete deve concentrarsi più sul “dove” (ZIP) o sul “quanto” (P2R).

---

### 🔹 4. Loss totale finale

Combinando tutto:

\[
L_{total} = L_{ZIP} + \alpha \left[ L_{P2R}^{MSE} + \beta L_{count} \right]
\]

dove:
- \( L_{ZIP} \) → regola la struttura globale del crowd,  
- \( L_{P2R}^{MSE} \) → regola la precisione locale,  
- \( L_{count} \) → mantiene coerente il conteggio totale,  
- \( \alpha \) → bilancia i due livelli (globale ↔ locale).

---

### 💡 Intuizione finale

> La loss combinata guida la rete in modo gerarchico:  
> prima **capisci dove** ci sono persone (ZIP), poi **affina quanto e dove esattamente** (P2R).  
>  
> Il parametro **α** regola l’equilibrio tra queste due “intelligenze” complementari.

---

## 📦 Dataset supportati

| Dataset | Split | Formato Ground Truth | Descrizione |
|----------|--------|----------------------|--------------|
| **JHU-CROWD++** | train/val | `.txt` (x y) | Scene urbane e affollate |
| **UCF-QNRF** | train/test | `.mat` (annPoints) | Immagini ad alta risoluzione |
| **ShanghaiTech Part A** | train/test | `.mat` | Scene indoor e outdoor |
| **NWPU-Crowd** | train/val | `.txt` (x y) | Dataset su larga scala |

---

## ⚙️ Configurazione (`config.yaml`)

```yaml
RUN_NAME: "jhu_p2rzip_v1"
DATASET: "jhu"
DATA:
  ROOT: "/mnt/localstorage/datasets/jhu_crowd_v2"
  ZIP_BLOCK_SIZE: 32
MODEL:
  BACKBONE: "vgg16_bn"
  GATE: "multiply"
OPTIM:
  EPOCHS: 300
  VAL_INTERVAL: 10
  RESUME_LAST: true
LOSS:
  JOINT_ALPHA: 1.0       # bilancia ZIP vs P2R
  P2R_SIGMA: 4.0         # σ per generare mappa densità
  COUNT_L1_W: 0.01       # peso β del conteggio
EXP:
  OUT_DIR: "exp"