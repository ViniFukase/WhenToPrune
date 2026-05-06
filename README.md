# Critical Learning Periods: Identifying Moments That Matter in Deep Neural Networks

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

This repository contains the official implementation of the Master's thesis **"Critical Learning Periods: Identifying Moments That Matter in Deep Neural Networks"** by Vinícius Yuiti Fukase (Universidade de São Paulo, 2026). 

## 📖 Abstract

Critical learning periods are early phases in deep learning where regularization and data density decisively shape a model's final generalization capacity. This work introduces a systematic approach to pinpoint these critical periods dynamically during training. By leveraging **Layer Rotation** as a generalization estimator, our method identifies the exact epoch to safely transition from resource-intensive training (full data/augmentations) to efficient modes (data pruning). 

Our approach guarantees a stable, monotonic reduction in training complexity, achieving near-lossless predictive performance (mean accuracy drop limited to 0.09%) while cutting CO₂ emissions and financial costs by up to ~60%.

## 🗂 Repository Structure

- `models/`: Contains the neural network architectures (e.g., ResNet family).
- `utils/`: Helper functions for data augmentation, calculating cosine distance, and applying linear regression over the 5-epoch window.
- `main.py`: The primary entry point implementing the core algorithm with dynamic conditional checking.
- `cifar_main.py`: Script integrating the Instance-dependent Early Stopping (IES) data pruning method. It ensures IES reduction mechanisms are activated only *after* the Critical Period is identified, avoiding sample re-introduction instability.
- `cifar_main_random.py`: Script for dynamic random pruning experiments. Allows customizable pruning ratios (e.g., 50%) and toggling the Critical Period (CP) logic to evaluate the impact of timing on simple stochastic baselines.
- `Annealing.py`: Script to apply the annealing parameter ($\delta$), re-introducing the full dataset during final epochs.
- `carbonEmission_FinancialCost.py`: Module to estimate and log training time, CO₂ emissions, and financial savings.

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/ViniFukase/WhenToPrune.git](https://github.com/ViniFukase/WhenToPrune.git)
   cd WhenToPrune
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 🚀 Running the Experiments

To start a standard training run with automated critical period detection:
```bash
python main.py --dataset cifar10 --model resnet32 --epochs 200 --batch_size 128
```

To run the annealing experiments:
```bash
python Annealing.py --annealing_factor 0.99
```

To measure environmental impact (Green AI metrics):
```bash
python carbonEmission_FinancialCost.py
```

## 📊 Main Results

- **Generalization**: Limits the mean accuracy drop to just **0.09%** compared to baseline full-data training (outperforming standard IES at 0.26%).
- **Efficiency**: Reduces training time by up to **59.67%**.
- **Green AI**: Achieves a **59.47% decrease in CO₂ emissions** and a 60% reduction in financial costs.
