"""
05 - Comparativa final entre reconocimiento facial y por voz

Junta los resultados de los scripts 01 a 04 para comparar directamente
las dos modalidades: precision sin ataque, y como de bien resiste cada
una los ataques adversariales. No ejecuta ningun modelo, solo lee los CSV
que ya generaron esos scripts.

Salidas:
  - results/tables/comparativa_baseline.csv
  - results/tables/comparativa_adversarial_resumen.csv
  - results/figures/comparativa_asr.png

Uso:
    python 05_comparativa_resultados.py
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


def load_required(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encuentra {path}. Ejecuta primero los scripts 01-04 "
            "para generar todos los resultados intermedios."
        )
    return pd.read_csv(path)


def main():
    facial_baseline = load_required(os.path.join(TABLES_DIR, "facial_baseline_summary.csv"))
    voice_baseline = load_required(os.path.join(TABLES_DIR, "voice_baseline_summary.csv"))
    facial_sweep = load_required(os.path.join(TABLES_DIR, "facial_adversarial_sweep.csv"))
    voice_sweep = load_required(os.path.join(TABLES_DIR, "voice_adversarial_sweep.csv"))

    baseline_cols = ["modalidad", "eer", "threshold_eer", "far_en_eer", "frr_en_eer",
                     "n_genuine", "n_impostor"]
    comparativa_baseline = pd.concat([
        facial_baseline[baseline_cols],
        voice_baseline[baseline_cols],
    ], ignore_index=True)
    comparativa_baseline.to_csv(os.path.join(TABLES_DIR, "comparativa_baseline.csv"), index=False)
    print("=== Comparativa baseline (sistema sin atacar) ===")
    print(comparativa_baseline.to_string(index=False))

    facial_sweep = facial_sweep.copy()
    voice_sweep = voice_sweep.copy()
    facial_sweep["modalidad"] = "facial"
    voice_sweep["modalidad"] = "voz"

    all_sweeps = pd.concat([facial_sweep, voice_sweep], ignore_index=True)

    resumen_rows = []
    for (modalidad, attack, mode), group in all_sweeps.groupby(["modalidad", "attack", "mode"]):
        group_sorted = group.sort_values("epsilon")
        resumen_rows.append({
            "modalidad": modalidad,
            "attack": attack,
            "mode": mode,
            "epsilon_min": group_sorted["epsilon"].iloc[0],
            "asr_en_epsilon_min": group_sorted["asr"].iloc[0],
            "epsilon_max": group_sorted["epsilon"].iloc[-1],
            "asr_en_epsilon_max": group_sorted["asr"].iloc[-1],
            "asr_maxima_observada": group_sorted["asr"].max(),
        })
    resumen = pd.DataFrame(resumen_rows).sort_values(["modalidad", "mode", "attack"])
    resumen.to_csv(os.path.join(TABLES_DIR, "comparativa_adversarial_resumen.csv"), index=False)
    print("\n=== Comparativa de ataques adversariales (resumen) ===")
    print(resumen.to_string(index=False))

    # Facial y voz usan los mismos valores de epsilon, asi que se pueden
    # dibujar directamente en el mismo eje sin ningun ajuste.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, mode, title in [
        (axes[0], "dodge", "Evasion (dodging) -> falso rechazo"),
        (axes[1], "impersonate", "Suplantacion (impersonation) -> falsa aceptacion"),
    ]:
        for modalidad, color in [("facial", "tab:blue"), ("voz", "tab:red")]:
            for attack, linestyle in [("FGSM", "--"), ("PGD", "-")]:
                subset = all_sweeps[
                    (all_sweeps["modalidad"] == modalidad)
                    & (all_sweeps["mode"] == mode)
                    & (all_sweeps["attack"] == attack)
                ].sort_values("epsilon")
                ax.plot(
                    subset["epsilon"], subset["asr"],
                    marker="o", linestyle=linestyle, color=color,
                    label=f"{modalidad} - {attack}",
                )
        ax.set_xlabel("epsilon (presupuesto de perturbación)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Attack Success Rate (ASR)")
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "comparativa_asr.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigura comparativa guardada en {fig_path}")


if __name__ == "__main__":
    main()
