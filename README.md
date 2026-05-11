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


├── dataset.xls          # Input dataset (monthly total-return indices)
├── final_version.py     # Full Python implementation
├── Report.pdf           # Full investment advisory report
└── README.md
---

## Repository Structure
