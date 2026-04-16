# Domain & Literature Research Notes

## 1. mAb Bioprocess Fundamentals

Monoclonal antibody (mAb) production uses CHO (Chinese Hamster Ovary) cells in fed-batch culture. The key dynamics:

- **Fed-batch process**: Cells grow in a bioreactor; glucose and glutamine are fed during a defined window to sustain growth
- **Temperature/pH shifts**: Used to trigger different metabolic phases (growth vs. production phase)
- **Titer** is the cumulative product concentration — driven by **viable cell density (VCD) x specific productivity x time**
- Longer cultures generally produce more, but late-phase metabolic byproducts (ammonia, lactate) can inhibit production and cell viability

This maps to our data: Z: = DoE settings, W: = temporal profiles (deterministic from Z:), X: = observed responses, Y:Titer = target.

## 2. Key Papers

**Park et al. (2023)** — Data-driven prediction models for CHO culture toward bioprocess digital twins. Compared multistep-ahead forecasting strategies, input features, and AI algorithms. [DOI](https://doi.org/10.1002/bit.28405)

**Barberi et al. (2022)** — ML on metabolomics + process data for titer prediction. Titer predictable from early timepoints using metabolic phenotype evolution. [DOI](https://doi.org/10.1016/j.ymben.2022.03.015)

**Clavaud et al. (2013)** — PCA/PLS for inline CHO monitoring. PCA: 96% variance in 2 PCs. PLS: accurately predicted VCD, titer, glucose, osmolality. **Integrated viable cell count (iVCC) as standard feature** — analogous to our VCD integral. [DOI](https://doi.org/10.1016/j.talanta.2013.03.044)

**Hutter et al. (2021)** — GP models for knowledge transfer across cell lines. DataHow team. Data structure nearly identical to our challenge. Derivative-based features preferred. [DOI](https://doi.org/10.1002/bit.27907)

**Portela et al. (2021)** — Digital twins in biopharma manufacturing, co-authored by Moritz von Stosch (DataHow AG). [DOI](https://doi.org/10.1007/10_2020_138)

## 3. DataHow's Methodology

- DataHow AG develops bioprocess digital twin software
- Rooted in multivariate data analysis (MVDA): PCA for monitoring, PLS regression for prediction
- Connects to Process Analytical Technology (PAT) guidelines

Sources: [DataHow - 8 Steps to Digital Bioprocessing](https://datahow.ch/8-steps-to-prepare-for-digital-bioprocessing-and-unlock-pharma-4-0/)

## 4. Variable-Length Time Series → Tabular Models

For our problem (100 experiments, variable 8-15 timesteps, small dataset):

- Deep learning inappropriate — too few samples
- Standard approach: extract features from time series, train tabular model
- Feature types: summary statistics, integrals (iVCC), rates (specific consumption/production), windowed features
- sklearn cannot handle variable-length sequences natively — feature extraction first
