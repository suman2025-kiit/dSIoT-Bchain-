# Validation and running guide of "dSIoT-Bchain" -
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

Here you go—clear run + validate steps for **sensor\_validation.py**.

### 1) Setup

```bash
python -V                 # any 3.9+
pip install numpy
```

### 2) Run the built-in demo

```bash
python sensor_validation.py
```

You should see three `ValidationResult` lines (DHT22, MQ-135, PIR) plus their Reliability/ISR numbers.

### 3) Use in your pipeline

```python
from sensor_validation import (
    validate_dht22, validate_mq135, validate_pir,
    make_reliability_from_validation, make_isr_from_validation)

vr_dht = validate_dht22("dht_12", [25.1, 25.4, None, 26.0, 80.5], [56.0, 55.4, 55.2, None, 54.9])
R_dht  = make_reliability_from_validation(vr_dht)
ISR_dht= make_isr_from_validation(vr_dht)
```

### 4) Quick validation checks (synthetic)

* **Gaps handled:** `None` values increase `gap_ratio` and lower scores.
* **RoC guard:** Large jumps are clipped (e.g., DHT22 >1 °C/sample).
* **Outliers removed:** IQR mask lowers `outlier_ratio` toward 0 on clean data.
* **Drift (MQ-135):**

```python
from sensor_validation import validate_mq135
x = [400]*60 + [700]*60         # clear shift
vr = validate_mq135("mq", x, drift_win=50, drift_thresh=0.15)
print(vr.drift_flag)  # True
```

* **PIR debouncing:** Short bursts are merged; improbable high rates reduce score if `max_event_rate_hz` is set.

### 5) Expected ranges

* `validation_score` ∈ \[0,1]; higher is better.
* `gap_ratio` ∈ \[0,1]; lower is better.
* `ok_rate` tracks fraction of kept samples.
* MQ-135 drift: `rel_shift ≥ drift_thresh` ⇒ `drift_flag=True` and soft score penalty.

**File:** [sensor\_validation.py](sandbox:/mnt/data/sensor_validation.py)

