# Neural Network ASCII Art

> Feedforward neural networks that learn to recognize ASCII characters and convert images to ASCII text using PyTorch.

**Author:** Yurii Ostapchuk

---

## About

This project implements and compares 3 neural network architectures trained to recognize 8 ASCII characters (` | / \ _ - ^ o`) from 8x14 pixel blocks, then uses the trained models to convert full images into ASCII art.

```
Image Block (8x14) -> Flatten (112) -> Neural Net -> One-Hot (8) -> ASCII Char
     [pixels]         [vector]        [model]       [classif.]     [char]
```

## Characters

| Index | Char | Description |
|:-----:|:----:|-------------|
| 0 | ` ` | Space (empty block) |
| 1 | `\|` | Vertical line |
| 2 | `/` | Forward slash |
| 3 | `\` | Backslash |
| 4 | `_` | Underscore |
| 5 | `-` | Horizontal line |
| 6 | `^` | Caret / arrow up |
| 7 | `o` | Circle |

## Models

| Model | Architecture | Activation | Parameters |
|-------|-------------|------------|------------|
| **NetSmall** | 112-64-32-8 | Sigmoid | ~8,000 |
| **NetMedium** | 112-128-64-32-8 | Sigmoid | ~25,000 |
| **NetLarge** | 112-256-128-64-32-8 | ReLU + Sigmoid | ~65,000 |

Each model is tested with 3 learning rate strategies:

| # | Strategy | Learning Rate |
|---|----------|---------------|
| 1 | Constant (baseline) | `1.0` (or `0.5` for large) |
| 2 | Constant (higher) | `2.0` |
| 3 | Step schedule | High -> Medium -> Low |

## Dataset

- Hand-crafted 8x14 binary templates for each character
- Multiple variants per character (different positions, thicknesses)
- Augmentation via shifts (+/- 2px horizontal, +/- 1px vertical) and noise
- Labels: one-hot encoded (8 classes)

## Project Structure

```
neural-networks-ascii/
├── ascii-art-ostapchuk.ipynb   # Jupyter notebook (school version, no outputs)
├── ascii-art-ostapchuk.py      # Python script (standalone)
├── test-photo.png              # Test image for ASCII conversion
├── requirements.txt            # Dependencies
├── .gitignore
└── models/                     # Saved model weights (.pth)
```

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run as script
python ascii-art-ostapchuk.py

# Or open in Jupyter
jupyter notebook ascii-art-ostapchuk.ipynb
```

## Requirements

- Python 3.10+
- PyTorch 2.11.0
- NumPy 2.4.2
- Pillow 11.2.1
