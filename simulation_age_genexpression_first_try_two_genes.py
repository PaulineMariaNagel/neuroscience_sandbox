# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 19:06:52 2026

@author: pauli
"""

# -*- coding: utf-8 -*-
"""
Erweiterung: Gehirnentwicklung & zwei Gen-Gruppen (Gen X und Gen Y)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Daten generieren (Simulation)
np.random.seed(42)
n_samples = 100

# Alter der Probanden
alter = np.random.uniform(3, 80, n_samples)

# Simuliertes Gehirnvolumen (Lifespan-Kurve)
gehirnvolumen = (
    1100 + 300 * (1 - np.exp(-0.1 * alter)) - 50 * (alter / 80) ** 2
) + np.random.normal(0, 40, n_samples)

# Gen X (Gruppe 1): Nimmt im Alter ab (z.B. entwicklungssitiv)
gen_x_expression = 5.0 - 0.03 * alter + np.random.normal(0, 0.5, n_samples)

# Gen Y (Gruppe 2 - Neu!): Nimmt im Alter zu (z.B. altersassoziiert)
gen_y_expression = 2.0 + 0.04 * alter + np.random.normal(0, 0.5, n_samples)

# In einen Pandas DataFrame verpacken
df = pd.DataFrame(
    {
        "Alter": alter,
        "Gehirnvolumen": gehirnvolumen,
        "Gen_X_Expression": gen_x_expression,
        "Gen_Y_Expression": gen_y_expression,
    }
)

# 2. Erste Einblicke in die Konsole drucken
print("--- Erste Zeilen unserer erweiterten Daten ---")
print(df.head())

# 3. Visualisierung: Gehirnvolumen im Vergleich zu beiden Gen-Gruppen
plt.figure(figsize=(14, 5))

# Plot 1: Gehirnvolumen vs. Gen X
plt.subplot(1, 2, 1)
plt.scatter(
    df["Gehirnvolumen"],
    df["Gen_X_Expression"],
    color="purple",
    alpha=0.7,
    edgecolors="k",
)
plt.title("Gehirnvolumen vs. Gen X (Entwicklung)")
plt.xlabel("Gehirnvolumen (cm³)")
plt.ylabel("Gen X Expression")
plt.grid(True, linestyle="--", alpha=0.5)

# Plot 2: Gehirnvolumen vs. Gen Y
plt.subplot(1, 2, 2)
plt.scatter(
    df["Gehirnvolumen"],
    df["Gen_Y_Expression"],
    color="teal",
    alpha=0.7,
    edgecolors="k",
)
plt.title("Gehirnvolumen vs. Gen Y (Alterung)")
plt.xlabel("Gehirnvolumen (cm³)")
plt.ylabel("Gen Y Expression")
plt.grid(True, linestyle="--", alpha=0.5)

# Layout anpassen und anzeigen
plt.tight_layout()
plt.show()

# 4. Korrelationen berechnen und vergleichen
korr_x = df["Gehirnvolumen"].corr(df["Gen_X_Expression"])
korr_y = df["Gehirnvolumen"].corr(df["Gen_Y_Expression"])

print("\n--- Korrelationsanalyse ---")
print(f"Korrelation (Gehirnvolumen & Gen X): {korr_x:.2f}")
print(f"Korrelation (Gehirnvolumen & Gen Y): {korr_y:.2f}")