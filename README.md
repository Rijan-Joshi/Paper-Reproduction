# 🧠 From-Scratch: Deep Learning Paper Implementations

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Active-success)
![Schedule](https://img.shields.io/badge/New%20Implementation-Every%20Friday-blueviolet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> *"What I cannot create, I do not understand."* — Richard Feynman
><!-- <img width="687" height="440" alt="image" src="https://github.com/user-attachments/assets/694f01ff-a0c8-47af-ab14-a73501c150d0" /> -->


## 📖 About This Repository
This repository documents my journey of **implementing, replicating, and experimenting** with seminal Deep Learning papers and architectures from scratch.

**The Goal:** To demystify the "black box" of neural networks.
**The Rule:** No pre-built model imports (e.g., `torchvision.models`). Every layer, training loop, and architecture is built from the ground up to understand the internal mechanics.

## 🗓️ The Schedule
I commit to adding a new implementation **every Friday**.

---

## 🚀 Progress & Benchmarks

| SN | Paper / Architecture | Domain | Key Concepts | Result | Status | Code |
|:--:|:---|:---|:---|:---|:--:|:--:|
| **01** | **[LeNet-5 (LeCun et al., 1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)** | CV | CNN, Tanh, AvgPool | 98.4% Acc | ✅ | [📂 View](/01_lenet5_1998) |
| **02** | *Next Paper...* | — | — | — | 🚧 | — |
| **03** | — | — | — | — | ⏳ | — |
| **04** | — | — | — | — | ⏳ | — |
| **05** | — | — | — | — | ⏳ | — |
| **06** | — | — | — | — | ⏳ | — |
| **07** | — | — | — | — | ⏳ | — |
| **08** | — | — | — | — | ⏳ | — |
| **09** | — | — | — | — | ⏳ | — |
| **10** | — | — | — | — | ⏳ | — |
| **11** | — | — | — | — | ⏳ | — |
| **12** | — | — | — | — | ⏳ | — |

---

## 📂 Repository Structure

Each paper is contained in its own folder to keep the environment self-contained.

```text
├── 01_lenet5_1998/
│   ├── model.py         # The model architecture (from scratch)
│   ├── train.py         # Training loop
│   ├── utils.py         # Data loading and helper functions
│   ├── notebooks/       # EDA and Visualization experiments
│   └── README.md        # Specific results/notes for this paper
├── 02_next_paper/
├── common/              # Shared utilities (logging, metrics)
└── README.md

```
## 🛠️ Getting Started

Follow these steps to set up the project locally.

### 1. Clone the repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

```
## 2. Run a specific implementation
Navigate to the specific folder to run
```bash
cd 01_lenet5_1998
python train.py
