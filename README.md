# neuroscience_sandbox
Learning playground for brain charts and normative modelling simulations

-> A small collection of basic, self-written Python scripts to wrap my head around lifespan brain charts and normative modelling (loosely inspired by papers like Bethlehem et al., 2022). 

## Disclaimer
**Work in progress/learning projects:** These scripts are purely educational "toy models" and basic heuristics built for practice. They are not methodically  researche pipelines; just me to understand the concepts.

## Scripts:
- `simulation_age_genexpression_first_try_two_genes.py`: very basic simulation trying to link lifespan brain volume trajectories with gene expression trends (very basic regression with simulated brain volume metrics)
- `normative_modeling_moving_avg.py`: Simulating a healthy reference cohort vs a clinical cohort to look at individual deviations and heterogeneity (moving avg)
- `normative_modeling_gamlss.py`: Building a GAMLSS-styled model using `pygam` to capture non-linear age trajectories and heteroscedastic variance (and some exploration what happens when data are overfittet and smoothing is reduced)
