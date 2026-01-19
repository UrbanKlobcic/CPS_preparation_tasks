# SF CPS 2025 – Practical Part

This repository contains the implementation for:

- **Task 1:** GridWorld environment (JAX)
- **Task 2:** Tabular Q-Learning (JAX + Optax)
- **Task 3:** REINFORCE (Flax + Optax)

from the course *Stochastic Foundations of Cyber-Physical Systems*.

To run the code, you must first create and activate the Python environment described below.

---

## 🔧 1. Environment Setup (assuming wsl)

We use the **uv** package manager, which automatically provides the correct Python version.

### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL

source .venv/bin/activate

install all the needed libraries of the project. 

python {code you want to run}
