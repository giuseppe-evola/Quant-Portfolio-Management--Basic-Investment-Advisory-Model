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
results, please refer to the [report](Report.pdf).

---

## Repository Structure

```text
.
├── dataset.xls           # Input dataset (monthly total-return indices)
├── final_version.py      # Full Python implementation
├── Report.pdf            # Full investment advisory report
└── README.md             # This file
```
---

## Requirements

The code is written in Python 3 and requires the following libraries:

numpy
scipy
pandas
matplotlib


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
| SAA Optimisation | Constrained minimum-variance with Michaud resampling (S=100, R=7 restarts) |
| TAA Optimisation | Black-Litterman posterior with semi-annual GMR views |
| Rebalancing | Semi-annual, L1 drift tolerance = 0.01 |
| Transaction Costs | Variable 10 bps + fixed GBP 10 per asset |

For the full mathematical formulation please refer to the report.
