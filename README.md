# Quantitative Portfolio Optimization
### Investment Advisory Model for Private Investors

> **Disclaimer:** This is a university project developed for academic purposes only.
> It is intended as an illustrative example of quantitative portfolio construction
> techniques and should not be interpreted as financial advice or as a production-ready
> investment system. The methodology, constraints, and results are simplified relative
> to what a real-world implementation would require.

---

## Overview

This repository contains the full implementation of a quantitative investment advisory
model designed for a hypothetical Private Investor division of a bank. The project
covers the entire pipeline from client profiling to portfolio construction and
out-of-sample backtesting.

The model is built around two methodological pillars:

- **Strategic Asset Allocation (SAA):** Markowitz mean-variance optimisation with
  Michaud resampling, applied to all client profiles.
- **Partial Tactical Asset Allocation (TAA):** Black-Litterman framework for
  informed clients, incorporating semi-annual analyst views into the posterior
  expected return vector.

For a full description of the methodology, mathematical formulation, and empirical
results, please refer to the [report](report.pdf).

---

## Repository Structure

├── datataset.xls/                  # Input dataset (monthly total-return indices)
├── final_version.py                # Full Python implementation
├── Report.pdf                   # Full investment advisory report
└── README.m

---

## Requirements

The code is written in Python 3 and requires the following libraries:

numpy
scipy
pandas
matplotlib


---

## How to Run

1. Clone the repository
2. Ensure the dataset is placed in the `data/` folder
3. Run the main script:

```bash
python code.py
```

The script will sequentially execute:
- SAA optimisation and resampling for all five risk profiles
- In-sample and out-of-sample backtesting for non-informed clients
- Black-Litterman optimisation and backtesting for informed clients
- Performance metrics and equity curve plots

---

## Dataset

The dataset consists of monthly total-return observations for eleven base asset
class indices (fixed income, global equity, and real assets) and up to nineteen
additional instruments available to informed clients (sectoral equity indices and
a commodity aggregate). Index sources are documented in the report.

---

## Methodology Summary

| Component | Method |
|---|---|
| Client Profiling | Questionnaire-based scoring, binary informed/non-informed split |
| SAA Optimisation | Constrained minimum-variance with Michaud resampling ($S=100$) |
| TAA Optimisation | Black-Litterman posterior with semi-annual GMR views |
| Rebalancing | Semi-annual, $\ell_1$ drift tolerance $\varepsilon = 0.01$ |
| Transaction Costs | Variable $c_v = 10$ bps + fixed $c_f = £10$ per asset |

For the full mathematical formulation, including the Black-Litterman posterior
derivation and constraint specifications, please refer to the report.

## Repository Structure
