# VuVoPy

🎙️ **VuVoPy** — Voice Utility Library for **Speech Parametrization**
A Python library for extracting acoustic features from speech signals,
specifically developed for **biomedical voice analysis** and Parkinson’s disease research.
---

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
[![Documentation Status](https://readthedocs.org/projects/vuvopy-documentation/badge/?version=latest)](https://vuvopy-documentation.readthedocs.io/en/latest/?badge=latest)

---

## 📦 Installation

Install VuVoPy with pip:
```bash
pip install VuVoPy
```
📚 **Documentation**
The full documentation is hosted on Read the Docs:
👉 https://vuvopy-documentation.readthedocs.io

Includes:

*   📦 Module overviews
*   📊 Feature extraction functions
*   🧠 Usage examples
*   🛠 Developer reference

🔬 **Features**

*   **durmad** — Duration of voiced segments (mean absolute deviation)
*   **durmed** — Duration of voiced segments (median)
*   **duv** — Percentage of unvoiced frames
*   **hnr** — Harmonics-to-noise ratio
*   **jitterPPQ** — Pitch perturbation quotient (jitter)
*   **shimmerAPQ** — Amplitude perturbation quotient (shimmer)
*   **mpt** — Maximum phonation time
*   **ppr** — Pitch period ratio
*   **relF0SD** — Relative standard deviation of F0
*   **relF1SD, relF2SD** — Relative deviation of formants F1 and F2
*   **relSEOSD** — Relative deviation at sentence ends
*   **spir** — Silence-to-phonation ratio

### Getting Started

Import VuVoPy and run a feature extraction function:
```python
python import VuVoPy as vp value = vp.durmad("my_signal.wav")
```
### 🧠 Example: Compute Speech Parameters
```python
import VuVoPy as vp
import pandas as pd

file_path = "signal.wav"

durmad = vp.durmad(file_path, winlen=512, winover=256, wintype='hamm')
durmed = vp.durmed(file_path, winlen=512, winover=256, wintype='hamm')
duv = vp.duv(file_path, winlen=512, winover=256, wintype='hamm')
hnr = vp.hnr(file_path)
jitter = vp.jitterPPQ(file_path)
mpt = vp.mpt(file_path, winlen=512, winover=256, wintype='hamm')
ppr = vp.ppr(file_path, winlen=512, winover=256, wintype='hamm')
relf0sd = vp.relF0SD(file_path)
relf1sd = vp.relF1SD(file_path, winlen=512, winover=256, wintype='hamm')
relf2sd = vp.relF2SD(file_path)
relseosd = vp.relSEOSD(file_path, winlen=512, winover=256, wintype='hamm')
shimmer = vp.shimmerAPQ(file_path)
spir = vp.spir(file_path, winlen=512, winover=256, wintype='hamm')

data = {
    "durmad": [durmad],
    "durmed": [durmed],
    "duv": [duv],
    "hnr": [hnr],
    "jitter": [jitter],
    "mpt": [mpt],
    "ppr": [ppr],
    "relf0sd": [relf0sd],
    "relf1sd": [relf1sd],
    "relf2sd": [relf2sd],
    "relseosd": [relseosd],
    "shimmer": [shimmer],
    "spir": [spir]
}

df = pd.DataFrame(data)
print(df)

```