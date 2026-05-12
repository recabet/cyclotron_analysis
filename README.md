# Cyclotron Analysis — LSTM Super-Resolution for FT-ICR MS

An LSTM Seq2Seq + Attention Bridge model for super-resolution of Fourier Transform Ion Cyclotron Resonance Mass Spectrometry (FT-ICR MS) signals.

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | **macOS**, **Linux**, or **Windows** (including WSL2) |
| Python | 3.10 – 3.12 |
| GPU | NVIDIA GPU recommended for training; CPU inference works for small datasets |
| CUDA | 12.1 compatible driver |
| RAM | 16 GB minimum; 32 GB recommended |
| Disk | ~5 GB for datasets and LMDB stores |

> **macOS (Apple Silicon / M1-M4):** Native MPS (Metal Performance Shaders) backend is used automatically when CUDA is unavailable.
> **Windows:** Native CUDA support via NVIDIA drivers. Use PowerShell or Git Bash.
> **Linux / WSL2:** Native CUDA support.

---

## Dependencies

All dependencies are listed in `requirements.txt`:

```
h5py~=3.16
numpy~=1.26.0
torch==2.5.1
scipy~=1.14.0
IsoSpecPy~=2.3.3
matplotlib~=3.9.0
scikit-learn~=1.5.0
lmdb~=1.4.1
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/recabet/cyclotron_analysis
cd cyclotron_analysis
```

### 2. Create a Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Windows (Git Bash / WSL2):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Verify your environment is active — you should see `(.venv)` in your terminal prompt.

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install PyTorch

**Linux / WSL2 (NVIDIA GPU — CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**macOS (Apple Silicon / M1-M4):**
```bash
pip install torch torchvision torchaudio
```

**macOS (Intel):**
```bash
pip install torch torchvision torchaudio
```

**Windows (NVIDIA GPU — CUDA 12.1):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU-only (all platforms):**
```bash
pip install torch torchvision torchaudio
```

Verify GPU is available:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('MPS available:', torch.backends.mps.is_available())"
```

Expected outputs:
- Linux/Windows + NVIDIA: `CUDA available: True`
- macOS Apple Silicon: `MPS available: True`
- CPU-only: both are `False`

### 5. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
cyclotron_analysis/
├── .venv/                  # Virtual environment (created during setup)
├── data/
│   ├── compounds/           # Compound isotope data
│   ├── lmdb/               # LMDB databases for training data
│   └── waves/              # HDF5 signal data (FFT, FID, segments)
├── figures/                # Output plots
├── train_plots/                  # Training plots
├── test_plots/                  # Testing plots
├── src/
│   ├── ad_hoc/             # Ad-hoc analysis scripts
│   ├── config/             # Simulation and training configuration
│   ├── io/                 # HDF5 read/write utilities
│   ├── isotope/            # Isotope distribution processing
│   ├── models/             # LSTM Seq2Seq + Attention Bridge
│   ├── processing/         # Signal filtering and segmentation
│   ├── scripts/            # Data generation and training scripts
│   ├── signal_processing/  # FID generation, FFT, windowing, noise
│   ├── training/           # Dataset classes, trainer, metrics
│   └── visualization/       # Plotting utilities
├── weights/                # Saved model checkpoints
├── report/                 # Thesis/report figures and LaTeX
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Running the Project

### Set Up Output Directories

Create the required directories before running for the first time:

**macOS / Linux / WSL2 / Git Bash:**
```bash
mkdir -p data/waves/fft data/waves/fid data/waves/segments data/waves/narrowband
mkdir -p data/lmdb/narrowband data/lmdb/cluster_prof_v2 data/lmdb/training_segments
mkdir -p figures train_plots weights
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Path data/waves/fft, data/waves/fid, data/waves/segments, data/waves/narrowband -Force
New-Item -ItemType Directory -Path data/lmdb/narrowband, data/lmdb/cluster_prof_v2, data/lmdb/training_segments -Force
New-Item -ItemType Directory -Path figures, plots, weights -Force
```

### Run the Full Pipeline

> **Always run from the project root directory.**

```bash
source .venv/bin/activate      # macOS / Linux / WSL2 / Git Bash
# .venv\Scripts\activate        # Windows PowerShell

python -m src.run
```

> **Do not run** `python src/run.py` — use `python -m src.run` from the project root.

### Run Individual Steps

If you want to run steps separately:

```bash
# Step 1: Generate clustered training data with constant noise
python -m src.scripts.generate_cluster_const_noise

# Step 2: Train the super-resolution model
python -m src.scripts.train_super_resolution

# Step 3: Test the model
python -m src.scripts.test_cluster_super_resolution
```

---

## Hardware-Specific Notes

### NVIDIA GPU (Linux / Windows)

Training uses **automatic mixed precision (AMP)** and **CUDA** by default. The `GradScaler` and `autocast` in `torch.amp` handle this automatically.

If you have a newer GPU (RTX 40-series or RTX 50-series) that requires CUDA 12.x, use:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Apple Silicon (macOS)

On M1-M4 Macs, the code uses the **MPS backend** automatically when CUDA is unavailable. Set the device explicitly if needed:

```python
import torch
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
```

Most training and inference code already handles this via `torch.cuda.is_available()` checks; MPS will be selected on Apple Silicon automatically.

### CPU-Only

Training on CPU is supported but very slow. Reduce `BATCH_SIZE` and `NUM_WORKERS` in `src/config/settings.py` for CPU runs.

---

Rajab Iskandarli
