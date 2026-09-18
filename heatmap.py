# -*- coding: utf-8 -*-
"""
Created on Wed Mar 2 11:09:37 2026

@author: paulinemarianagel

"""

# -*- coding: utf-8 -*-
"""
visualise a connectivity matrix as a clean correlation heatmap.
"""

import matplotlib.pyplot as plt
import numpy as np


def simulate_fmri_timeseries(n_regions=6, n_timepoints=150):
  np.random.seed(42)

  shared_signal = np.sin(np.linspace(0, 15, n_timepoints))

  time_series = []
  for i in range(n_regions):
    noise = np.random.normal(0, 1, n_timepoints)
    weight = np.random.uniform(
        0.1, 0.7
    ) 
    signal = weight * shared_signal + (1 - weight) * noise
    time_series.append(signal)

  return np.array(time_series)  # Shape: (n_regions, n_timepoints)


if __name__ == "__main__":
  print("Simulating fMRI BOLD time series...")
  n_regions = 6
  data = simulate_fmri_timeseries(n_regions=n_regions)

  # dummy names
  region_names = [
      "PCC (DMN)",
      "mPFC (DMN)",
      "Left IPC",
      "Right IPC",
      "Visual Cortex",
      "Motor Cortex",
  ]

  print(
      "Calculating functional connectivity matrix (Pearson correlation)..."
  )
  
  corr_matrix = np.corrcoef(data)

  print("Plotting functional connectivity heatmap...")
  plt.figure(figsize=(8, 6))
  cax = plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
  plt.colorbar(cax, label="Pearson Correlation (r)")


  plt.xticks(
      range(n_regions), region_names, rotation=45, ha="right"
  )
  plt.yticks(range(n_regions), region_names)

  plt.title("Simulated fMRI Functional Connectivity Matrix")
  plt.tight_layout()
  plt.show()