🔬 Cyclotron Analysis — LSTM Super-Resolution for FT-ICR MS

This project implements an LSTM Seq2Seq + Attention Bridge model for super-resolution of FT-ICR mass spectrometry signals.
🖥 System Requirements

    OS: Linux / WSL

    GPU: NVIDIA GPU (Tested on RTX 5070 Ti Mobile)

    Drivers: NVIDIA drivers installed

    CUDA: 12.8 compatible driver

    Note: This setup assumes CUDA 12.8 because RTX 50-series GPUs require CUDA 12.x builds of PyTorch.

📦 Recommended Python Version

Use Python 3.11:

```bash
sudo apt install python3.11 python3.11-venv python3.11-dev
```

🚀 Installation

1️⃣ Clone Repository
```bash
git clone https://github.com/recabet/cyclotron_analysis
cd cyclotron_analysis
```

2️⃣ Create Virtual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Verify:

```bash
which python
```
Should point to: 
```shell
.../cyclotron_analysis/.venv/bin/python
````
3️⃣ Upgrade pip
```bash
pip install --upgrade pip
```

4️⃣ Install PyTorch (CUDA 12.8 Build)

Because RTX 5070 Ti Mobile requires CUDA 12.x:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```


Expected output: ```True```

5️⃣ Install Remaining Dependencies

Install:

```bash
pip install -r requirements.txt
```

```
cyclotron_analysis/
│
├── .venv/
├── src/
│   ├── run.py
│   ├── config/
│   ├── models/
│   ├── training/
│   ├── scripts/
│   ├── processing/
│   ├── signal/
│   └── visualization/
|
├── requirements.txt
└── README.md
```


▶️ Running The Project

⚠️ Always run from the project root.

Activate environment:

```bash
source .venv/bin/activate
```
Run:

```bash
python -m src.run
```


❌ DO NOT run: python src/run.py
⚙️ What run.py Does

src/run.py:

```python
from src.scripts import generate_training_data
from src.scripts import train_super_resolution

def main():
    # generate_training_data.main()
    train_super_resolution.main()

if __name__ == "main":
    main()
```


Running python -m src.run will:

    (Optional) Generate training data

    Train the LSTM super-resolution model


Rajab Iskandarli