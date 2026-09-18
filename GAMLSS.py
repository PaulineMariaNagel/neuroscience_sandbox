
"""
GAMLSS;
-> works with pygam; smooths curves (splines) to flexibly capture nonlinear trends whilst keeping the underlying statistics transparent 
-> Instead of rolling quantiles, GAMLSS are used to smoothly fit the mean trajectory and the age-dependent variance.

At the end there is an intentionally overfitted model for comparison
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import LinearGAM, s


def generate_gamlss_data(n=1500):
  np.random.seed(42)
  age = np.random.uniform(3, 80, n)

  # non-linear lifespan trajectory
  mean_vol = 1100 + 300 * (1 - np.exp(-0.1 * age)) - 50 * (age / 80) ** 2

  # brains are heteroscedastic
  # Younger/older brains often have higher spread than mid-adulthood brains
  # simulating variance with age:
  age_scaled_std = 30 + 20 * (abs(age - 40) / 40)
  brain_vol = mean_vol + np.random.normal(0, 1, n) * age_scaled_std

  return pd.DataFrame({"Age": age, "Brain_Volume": brain_vol})


if __name__ == "__main__":
  print("Generating dataset...")
  df = generate_gamlss_data()

  # Prep data for pygam
  X = df[["Age"]].values #we must have 2d array 
  y = df["Brain_Volume"].values

  print("Fitting a smooth GAM to capture the non-linear age trajectory...")
  # s(0) fits a smooth spline to the first feature (here age)
  gam = LinearGAM(s(0, n_splines=10)).fit(X, y)

  # then generate smooth age grid for plotting predictions
  # ERROR:  age_grid = np.linspace(3, 80, 300) 
  age_grid = np.linspace(3, 80, 300).reshape(
      -1, 1
  )  

  # Predict mean trajectory and confidence intervals (approx centiles)
  predicted_mean = gam.predict(age_grid)
  prediction_intervals = gam.prediction_intervals(age_grid, width=0.95)

  print("Plotting the GAMLSS-style smooth centile curves...")
  plt.figure(figsize=(10, 6))

  # Background data
  plt.scatter(
      df["Age"],
      df["Brain_Volume"],
      color="lightgray",
      alpha=0.3,
      label="Reference Cohort",
  )

  # fitted mean and 95% intervals from the GAM
  plt.plot(
      age_grid,
      predicted_mean,
      color="navy",
      linewidth=2.5,
      label="GAM Fitted Mean",
  )
  plt.plot(
      age_grid,
      prediction_intervals[:, 0],
      color="darkred",
      linestyle="--",
      label="Lower 95% Interval (~5th Centile)",
  )
  plt.plot(
      age_grid,
      prediction_intervals[:, 1],
      color="darkred",
      linestyle="--",
      label="Upper 95% Interval (~95th Centile)",
  )

  plt.title("Simplified GAMLSS / Additive Model for Brain Volume")
  plt.xlabel("Age (years)")
  plt.ylabel("Brain Volume (cm³)")
  plt.legend(loc="upper right")
  plt.grid(True, linestyle="--", alpha=0.3)
  plt.tight_layout()
  plt.show()
  
  
  
  
 # for comparison and exploration
 # what happens if data are overfitted or smoothing is reduced

def generate_gamlss_data(n=1500):
  np.random.seed(42)
  age = np.random.uniform(3, 80, n)

  # nonlinear lifespan trajectory
  mean_vol = 1100 + 300 * (1 - np.exp(-0.1 * age)) - 50 * (age / 80) ** 2

  # brains are heteroscedastic
  age_scaled_std = 30 + 20 * (abs(age - 40) / 40)
  brain_vol = mean_vol + np.random.normal(0, 1, n) * age_scaled_std

  return pd.DataFrame({"Age": age, "Brain_Volume": brain_vol})


if __name__ == "__main__":
  print("Generating dataset...")
  df = generate_gamlss_data()

  # Prep data for pygam
  X = df[["Age"]].values
  y = df["Brain_Volume"].values

  print("Fitting an EXTREME, overfitted GAM...")
  # n_splines=60 gives it tons of flexibility, lam=1e-6 to remove almost all smoothing penalty
  gam = LinearGAM(s(0, n_splines=60, lam=1e-6)).fit(X, y)

  # generate smooth age grid for plotting predictions
  age_grid = np.linspace(3, 80, 300).reshape(-1, 1)

  # Predict mean trajectory and confidence intervals
  predicted_mean = gam.predict(age_grid)
  prediction_intervals = gam.prediction_intervals(age_grid, width=0.95)

  print("Plotting the chaos...")
  plt.figure(figsize=(10, 6))

  # Background data
  plt.scatter(
      df["Age"],
      df["Brain_Volume"],
      color="lightgray",
      alpha=0.3,
      label="Reference Cohort",
  )

  plt.plot(
      age_grid,
      predicted_mean,
      color="crimson",
      linewidth=2.5,
      label="Overfitted Wild Mean",
  )
  plt.plot(
      age_grid,
      prediction_intervals[:, 0],
      color="darkorange",
      linestyle="--",
      label="Lower Interval",
  )
  plt.plot(
      age_grid,
      prediction_intervals[:, 1],
      color="darkorange",
      linestyle="--",
      label="Upper Interval",
  )

  plt.title("EXTREME Overfitted GAM (Total Chaos)")
  plt.xlabel("Age (years)")
  plt.ylabel("Brain Volume (cm³)")
  plt.legend(loc="upper right")
  plt.grid(True, linestyle="--", alpha=0.3)
  plt.tight_layout()
  plt.show()
