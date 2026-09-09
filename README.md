# R-ratio

Code repository for estimating the **recurrent-ratio (R-ratio)** using a modified parametric feedback inhibition control (pFIC) model.

The R-ratio is designed to quantify the relative balance between **intra-regional recurrent excitation** and **inter-regional excitatory amplification** in a whole-brain circuit model. By extending the pFIC framework, this repository provides analysis scripts for estimating model parameters, validating model performance, and computing the R-ratio from optimized regional excitation parameters.

> paper: XX

---

## Repository structure

```text
Rratio/
├── code/        # Core Python scripts for model simulation, training, validation, and testing
├── config/      # configuration files for paths, model parameters, and simulation settings
├── datasets/    # Prepared input files used for running the model 
├── scripts/     # Shell scripts for executing pipelines
└── README.md    
```

---

## Analysis workflow
### Model optimization

Example commands:

```bash
bash scripts/HCP.sh
or
bash scripts/ADNI.sh
```

### R-ratio calculation

The R-ratio is computed from the optimized intra-regional recurrent excitation and inter-regional excitatory amplification parameters.

```python
import numpy as np
import pandas as pd

# Load optimized parameter estimates from the test output
test_df = pd.read_csv("output_dir/test/test_all.csv", header=None).iloc[2:, :]

# Extract parameter estimates
# Example for the aparc atlas with 68 cortical regions
n_roi = 68

wEE_intra = test_df.iloc[12: 12 + n_roi, :].to_numpy(dtype=float)
wEE_inter = test_df.iloc[-n_roi:, :].to_numpy(dtype=float)

# Compute R-ratio
R_ratio = wEE_intra / (wEE_intra + wEE_inter)
```

## Citation

If you use this code, please cite:

```text
Citation information will be added upon publication.
```

---

## Acknowledgements

This repository builds on the pFIC modeling framework and replication resources developed by the Thomas Yeo Lab as part of the CBIG repository:

- [CBIG Zhang2024 pFIC replication code](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/fMRI_dynamics/Zhang2024_pFIC)

We gratefully acknowledge the authors and contributors of the original pFIC work and the CBIG repository for providing openly available resources that served as an important foundation for this implementation.

---

## Contact

For questions or correspondence, please contact:

```text
Yeongjun Park / pyjkhn@korea.ac.kr
Bo-yong Park / boyongpark@korea.ac.kr
```
