"""
06 - Comparativa de tiempos y coste computacional

Compara cuanto tarda cada modalidad, usando los tiempos que se midieron
realmente al ejecutar los scripts 01-04 (impresos por cada uno al
terminar). No vuelve a ejecutar nada, esos tiempos ya estan anotados mas
abajo tal como se registraron.

Salidas:
  - results/tables/comparativa_tiempos.csv
  - results/figures/comparativa_tiempos.png
  - results/figures/comparativa_throughput.png

Uso:
    python 06_comparativa_tiempos.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Tiempos medidos realmente al ejecutar cada script.
# n_muestras: cuantas imagenes o audios se han procesado.
# n_evaluaciones_modelo: cuantas veces se ha ejecutado el modelo en total.
# No son lo mismo porque en los ataques cada muestra pasa por el modelo
# varias veces (una por cada paso de PGD, por ejemplo).
REGISTROS = [
    {
        "script": "01_facial_baseline.py",
        "modalidad": "facial",
        "fase": "baseline (extraccion de embeddings)",
        "n_muestras": 2000,          # 1000 pares x 2 imagenes
        "n_evaluaciones_modelo": 2000,
        "tiempo_segundos": 246.4,
        "hardware": "CPU",
    },
    {
        "script": "02_facial_adversarial.py",
        "modalidad": "facial",
        "fase": "ataques FGSM+PGD (2 objetivos x 5 epsilon, PGD=10 pasos)",
        "n_muestras": 80,            # 40 pares evasion + 40 pares suplantacion
        "n_evaluaciones_modelo": 400 + 4000 + 1600,  # FGSM + PGD (10 pasos) + puntuaciones antes/despues
        "tiempo_segundos": 1521.5,
        "hardware": "CPU",
    },
    {
        "script": "03_voice_baseline.py",
        "modalidad": "voz",
        "fase": "baseline (extraccion de embeddings)",
        "n_muestras": 800,           # 400 pares (200 genuinos + 200 impostores) x 2 locuciones
        "n_evaluaciones_modelo": 800,
        "tiempo_segundos": 701.6,
        "hardware": "CPU",
    },
    {
        "script": "04_voice_adversarial.py (N=40, final)",
        "modalidad": "voz",
        "fase": "ataques FGSM+PGD (2 objetivos x 5 epsilon, PGD=5 pasos)",
        "n_muestras": 80,            # 40 pares evasion + 40 pares suplantacion
        "n_evaluaciones_modelo": 400 + 2000 + 800,  # FGSM + PGD (5 pasos) + puntuaciones despues del ataque
        "tiempo_segundos": 10377.5,
        "hardware": "CPU",
    },
]

# La prueba piloto de voz (N=8) se anota aparte porque su tiempo no es
# fiable, aunque sus resultados si se conservan en
# results/tables/voice_adversarial_sweep_n8.csv.
NOTA_PILOTO_N8 = (
    "04_voice_adversarial.py (N=8, piloto): tiempo de pared no fiable "
    "(~30768 s, incluye una suspension del equipo de varias horas); "
    "se excluye de la comparativa cuantitativa de tiempos."
)


def main():
    df = pd.DataFrame(REGISTROS)
    df["tiempo_minutos"] = df["tiempo_segundos"] / 60
    df["segundos_por_muestra"] = df["tiempo_segundos"] / df["n_muestras"]
    df["segundos_por_evaluacion_modelo"] = df["tiempo_segundos"] / df["n_evaluaciones_modelo"]

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    csv_path = os.path.join(TABLES_DIR, "comparativa_tiempos.csv")
    df.to_csv(csv_path, index=False)
    print(f"Tabla guardada en {csv_path}")
    print(df.to_string(index=False))
    print("\nNota:", NOTA_PILOTO_N8)

    # Escala logaritmica porque los tiempos van de minutos a casi 3 horas.
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["tab:blue" if m == "facial" else "tab:red" for m in df["modalidad"]]
    labels = [f"{row.modalidad}\n{row.fase}" for row in df.itertuples()]
    ax.barh(labels, df["tiempo_minutos"], color=colors)
    ax.set_xlabel("Tiempo total (minutos, escala logarítmica)")
    ax.set_xscale("log")
    ax.set_title("Tiempo de ejecución por fase (solo CPU)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "comparativa_tiempos.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigura guardada en {fig_path}")

    # Comparacion mas directa entre modalidades: una sola pasada por
    # muestra, sin que influyan los ataques ni el numero de pasos de PGD.
    baseline = df[df["fase"].str.contains("baseline")]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(baseline["modalidad"], baseline["segundos_por_muestra"],
           color=["tab:blue", "tab:red"])
    for i, v in enumerate(baseline["segundos_por_muestra"]):
        ax.text(i, v, f"{v:.3f} s", ha="center", va="bottom")
    ax.set_ylabel("Segundos por muestra (un único forward del modelo)")
    ax.set_title("Coste de extraer un embedding: facial vs. voz")
    plt.tight_layout()
    fig_path2 = os.path.join(FIGURES_DIR, "comparativa_throughput.png")
    plt.savefig(fig_path2, dpi=150)
    print(f"Figura guardada en {fig_path2}")

    ratio = (baseline[baseline["modalidad"] == "voz"]["segundos_por_muestra"].iloc[0]
             / baseline[baseline["modalidad"] == "facial"]["segundos_por_muestra"].iloc[0])
    print(f"\nExtraer un embedding de voz es aproximadamente {ratio:.1f}x mas lento "
          f"que extraer uno facial, en este equipo.")


if __name__ == "__main__":
    main()
