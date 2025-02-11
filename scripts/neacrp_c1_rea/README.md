# NEACRP C1 Rod Ejection Accident Data Generation

In this directory, we include all the PARCS input files and processing scripts we used to generate the data set for the [NEACRP C1 Rod Ejection Accident](https://pymaise.readthedocs.io/en/latest/benchmarks/rod_ejection.html).

## PARCS Input and Processing Scripts

The scripts and input files include:

- `C1B1b1.inp`: The main PARCS input file with find/replace strings `new_bank8_pos`, `new_gamma_frac`, and `new_hgap`.
- `xsec/XSEC_NEACRP_temp`: The PARCS cross section file with find/replace the string `new_beta`.
- `rod_worth.py`: Python script for running PARCS at different bank 8 positions (400 linearly spaced positions between 0 and 228).
- `data_generation.py`: Samples the distributions discussed in the following sections, replaces the strings in `C1B1b1.inp` and `xsec/XSEC_NEACRP_temp`, writes these to `C1.inp` and `xsec/XSEC_NEACRP`, runs PARCS, and saves each `C1.parcs_plt` to a numpy binary `outputs3D.npy`. Writes the input samples to `rea_inputs.csv`.
- `analysis.py`: Takes four outputs from `outputs3D.npy`: max fuel temperature, max power, average outlet temperature, and the burst width, which is the full width at half maximum of the power curve. Writes the output samples to `rea_outputs.csv`.

## Sampling Distributions

The sampling distributions used include:

| Input Parameter | Distribution Type | Distribution Parameters
| ---| ---| ---|
| Rod Worth $`(\rho)`$ | Uniform $`[\rho^{Max}(1 - 0.15), \rho^{Max}]`$ | $`\rho^{Max} = 0.009438`$|
| Delayed Neutron Fraction $`(\beta)`$ | Normal $`[\mu, \sigma]`$ | $`\mu = 0.076, \sigma=0.05\\mu`$|
| Gap Conductance $`\Big(h_{gap}\Big[\frac{W}{m^2\cdot K}\Big]\Big)`$ | Normal $`[\mu, \sigma]`$ | $`\mu = 10000, \sigma=0.20\mu`$|
| Direct Heating Fraction $`(\gamma_{frac})`$ | Normal $`[\mu, \sigma]`$ | $`\mu = 0.019, \sigma=0.20\mu`$|

When these distributions are sampled in `data_generation.py`, `new_gamma_frac` and `new_hgap` are replaced with their result for each configuration of `C1B1b1.inp`. The sampled rod worth is used to lin-lin interpolate the bank8 position according to the `rod_worth.csv` table. The generation of this table is discussed in the next section. The `new_bank8_pos` in `C1B1b1.inp` is then replaced. The delayed neutron fraction is sampled such that the fraction of delayed neutron fraction is preserved for each group. This follows

```math
\beta_i^{Sampled} = \frac{\beta_i^{Ref}}{{\sum}_{j=1}^{6} \beta_j^{Ref}} \beta^{Sampled},
```

where $`\beta_i^{Ref} \in \{0.0002584,0.00152,0.0013908,0.0030704,0.001102,0.0002584\}`$.

## Construction of Rod Worth Interpolation Table

To construct the rod worth interpolation table, we pertubed the bank position for 400 linearly spaced positions between 0 and 228. These configurations were run in PARCS, and the rod worth is given by

```math
\rho = \frac{k_{eff} - 1}{k_{eff}}.
```

The rod worth curve is shown in the figure below.

![alt text](https://github.com/aims-umich/pyMAISE/blob/develop/scripts/neacrp_c1_rea/rod_worth.png)

## Running These Scripts

To run these scripts, replace `[PLACE PATH TO PARCS]` in both `data_generation.py` and `rod_worth.py` and run the `job.sh` bash script.
