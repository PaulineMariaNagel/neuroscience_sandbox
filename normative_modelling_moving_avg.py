# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 20:10:28 2026

@author: paulinemarianagel
"""

# -*- coding: utf-8 -*-
"""
trying to simulate normative modelling to capture clinical heterogeneity; moving
away from case-control studies and leaning towards quantifying individual deviations from a healthy reference baseline
Inspired by Bethlehem et al.'s work on using brain charts / normative models 
to study conditions like autism 
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_cohorts(n_healthy=1500, n_clinical=200):
  np.random.seed(42)

  # We need healthy references as they build our baseline lifespan distribution
  age_healthy = np.random.uniform(3, 80, n_healthy)
  mean_vol_h = (
      1100 + 300 * (1 - np.exp(-0.1 * age_healthy)) - 50 * (age_healthy / 80) ** 2
  )
  vol_healthy = mean_vol_h + np.random.normal(0, 40, n_healthy)

  df_healthy = pd.DataFrame(
      {"Age": age_healthy, "Brain_Volume": vol_healthy, "Group": "Healthy"}
  )

  # then the clinical cohort (eg a neurodevelopmental condition like autism or adhd)
  # -> no uniform group shift but high individual scatter/heterogeneity
  age_clinical = np.random.uniform(5, 50, n_clinical)
  mean_vol_c = (
      1100
      + 300 * (1 - np.exp(-0.1 * age_clinical))
      - 50 * (age_clinical / 80) ** 2
  )

  # Higher variance to mimic individual patient divergence
  vol_clinical = mean_vol_c + np.random.normal(0, 75, n_clinical)

  df_clinical = pd.DataFrame(
      {"Age": age_clinical, "Brain_Volume": vol_clinical, "Group": "Clinical"}
  )

  # Merge them together into one big dataframe for analysis
  return pd.concat([df_healthy, df_clinical], ignore_index=True)


if __name__ == "__main__":
  print("Generating healthy reference and clinical cohorts...")
  df = generate_cohorts()

  # Sort healthy subset chronologically so we can calculate smooth rolling baselines
  df_healthy_sorted = (
      df[df["Group"] == "Healthy"].sort_values("Age").reset_index(drop=True)
  )

  # Calculate the normative mean and sd across healthy lifespan
  window_size = 100
  df_healthy_sorted["normative_mean"] = (
      df_healthy_sorted["Brain_Volume"]
      .rolling(window_size, center=True, min_periods=20)
      .mean()
  )
  df_healthy_sorted["normative_std"] = (
      df_healthy_sorted["Brain_Volume"]
      .rolling(window_size, center=True, min_periods=20)
      .std()
  )

  # drop NaNs caused by rolling window
  df_healthy_sorted = df_healthy_sorted.dropna()

  print("Visualising normative model vs clinical heterogeneity...")
  plt.figure(figsize=(10, 6))

  #pllot healthy normative median trajectory
  plt.plot(
      df_healthy_sorted["Age"],
      df_healthy_sorted["normative_mean"],
      color="navy",
      linewidth=2.5,
      label="Normative Median (Healthy Baseline)",
  )

  # Shade 95% normative range (mean +- 1.96*sd)
  plt.fill_between(
      df_healthy_sorted["Age"],
      df_healthy_sorted["normative_mean"]
      - 1.96 * df_healthy_sorted["normative_std"],
      df_healthy_sorted["normative_mean"]
      + 1.96 * df_healthy_sorted["normative_std"],
      color="navy",
      alpha=0.1,
      label="95% Normative Range",
  )

  # Now plot the clinical cohort 
  df_clinical = df[df["Group"] == "Clinical"]
  plt.scatter(
      df_clinical["Age"],
      df_clinical["Brain_Volume"],
      color="crimson",
      alpha=0.6,
      edgecolors="k",
      s=40,
      label="Clinical Cohort (High Heterogeneity)",
  )

  # Formatting the plot nicely
  plt.title("Normative Modelling & Clinical Heterogeneity")
  plt.xlabel("Age (years)")
  plt.ylabel("Brain Volume (cm³)")
  plt.legend(loc="upper right")
  plt.grid(True, linestyle="--", alpha=0.3)
  plt.tight_layout()

  plt.show()