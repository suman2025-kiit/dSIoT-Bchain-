# dSIoT-Bchain-
Distributed Social Behaviour for SIoT using Blockchain network 

#**To excute the application please consider the following steps-**

1. Create an isolated environment (recommended)

python3 -m venv .venv
source .venv/bin/activate               # Linux / macOS
# .venv\Scripts\activate                 # Windows (PowerShell)

2. Install the following dependencies

pip install --upgrade pip
pip install -r requirements.txt

Install following requirements as well.
networkx>=3.3
numpy>=1.26
scikit-learn>=1.5
matplotlib>=3.9
jupyterlab>=4.2

3.Launch Jupyter and run the notebook

jupyter lab dSIoT_BChain.ipynb

Select “Run ▸ Run All Cells”

The notebook will:

Generate a synthetic SIoT network (20 random devices).

Cluster devices with DBSCAN in 11-D feature space.

Compute intrinsic & reputation trust scores.

Plot cluster evolution and trust-score convergence.

Print summary metrics (avg latency, false-alarm rate, energy/tx).

Figures are saved automatically to figures/ as PNG/JPG files.

4. Typical troubleshooting
   
| Symptom                               | Fix                                                                                 |
|---------------------------------------|-------------------------------------------------------------------------------------|
| `ModuleNotFoundError`                 | `pip install` the missing package (see list above).                                 |
| `ImportError: cannot import name ...` | Make sure **scikit-learn ≥ 1.5** (older builds changed DBSCAN’s import path).       |
| Blank plots / backend errors          | Add `matplotlib` to the venv; on servers set `MPLBACKEND=Agg` before running.       |
| Very long runtime                     | Lower `NUM_DEVICES` in the first code cell or run on Python ≥ 3.10 for faster numpy |


5. Regenerating paper artefacts

 | Artefact (paper section) | How to reproduce |
|--------------------------|------------------|
| **Fig. 2 – Architecture** | Static SVG (`assets/architecture.svg`) – committed. |
| **Fig. 4/5 – Cluster Evolution** | Run notebook; outputs `figures/cluster_iter_1.png`, `cluster_iter_5.png`. |
| **Table V – Cluster stats** | Automatically printed to console and stored as `outputs/cluster_stats.csv`. |
| **Equation (3) & (10)** | Implemented in `calculate_trust_score()` inside notebook (no extra steps). |

6. Unit tests (if Required)
pytest tests/

tests/test_trust_model.py checks weight coefficients and that hit-time isolation triggers when expected.

7. One-liner demo (if required)

python - <<'PY'
from dSIoT_BChain import UnsupervisedSIoTNetwork
net = UnsupervisedSIoTNetwork()
for i in range(10): net.add_device(f"dev{i}", "sensor")
net.run_simulation(steps=5)
print("Average trust:", net.get_average_trust())
PY

# Sample Simulation output for 5 different Interactions of  'dSIoT-Bchain'  -


![1](https://github.com/user-attachments/assets/4f5e3e10-a0d3-4dd9-8c36-6ca40c6dc18f)

![2](https://github.com/user-attachments/assets/b4502df0-b82e-4743-9181-c3f23d9e3bbc)
![3](https://github.com/user-attachments/assets/ca7b1ac8-6560-450a-861f-43b31e809201)
![4](https://github.com/user-attachments/assets/46af1752-b740-4f46-b043-9cfb49382225)
![5](https://github.com/user-attachments/assets/641dc8ba-ca14-45c5-9ba1-361b44ff62ea)

# Sample Simulation output for 1st  Interactions at background of  'dSIoT-Bchain'  -


![1s](https://github.com/user-attachments/assets/db96c171-f2bf-4ec5-826a-31997ebbe5ba)
