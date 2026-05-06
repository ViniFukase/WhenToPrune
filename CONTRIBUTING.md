## CONTRIBUTING.md

```markdown
# Contributing to Critical Learning Periods Research

Thank you for your interest in contributing to our research! This project prioritizes **scientific reproducibility**. Whether you are fixing a bug, extending the method to Large Language Models (LLMs), or testing new pruning metrics, please follow the guidelines below.

## 🔬 Scientific Rigor & Reproducibility
1. **Deterministic Execution**: Always set random seeds for NumPy, PyTorch, and Python's built-in `random` module. Researchers must be able to achieve the exact same results shown in the main paper.
2. **Layer Rotation Integrity**: Any changes to how weights are saved or accessed in the `weights/` directory must not disrupt the $d_{cos}$ calculation comparing $\theta^t$ and $\theta^0$.
3. **Hyperparameter Logging**: Any new experiment or configuration must log its hyperparameters (batch size, learning rate, augmentation factor $k$) clearly.

## 🛠 Pull Request Process
1. Fork the repository and create your feature branch (`git checkout -b feature/new-pruning-metric`).
2. Ensure any new dependencies are added to `requirements.txt`.
3. If you modify the core logic in `main.py` or `utils/`, please run a baseline test on CIFAR-10 with ResNet32 to ensure the 5-epoch window linear regression still effectively triggers at $\alpha < 45^{\circ}$.
4. Submit your PR with a clear description of the computational overhead your change introduces.

## 📝 Code Style
- Keep the code well-documented, focusing on *why* a mathematical operation is performed.
- Use type hinting where applicable.
