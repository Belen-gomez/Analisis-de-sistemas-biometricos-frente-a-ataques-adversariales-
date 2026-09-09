"""
01 - Rendimiento del reconocimiento facial sin ningun ataque

Mide como funciona el sistema de reconocimiento facial en condiciones
normales (FAR, FRR, EER, curva ROC), antes de aplicar ningun ataque. Estos
resultados sirven de referencia para compararlos despues con los que se
obtienen bajo ataque, en 02_facial_adversarial.py.

Usa el dataset LFW, que se descarga solo la primera vez que se ejecuta.

Uso:
    python 01_facial_baseline.py
"""

import os
import sys
import time

# Rutas calculadas desde este archivo, para que el script funcione igual
# sin importar desde donde se ejecute.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # sin pantalla; las figuras se guardan a fichero
import matplotlib.pyplot as plt

from common.metrics import roc_curve, compute_eer, far_frr_at_threshold
from face.pipeline import FaceEmbedder, load_lfw_pairs, pairs_to_pil, compute_pair_scores


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    lfw = load_lfw_pairs(subset="test", resize=1.0, color=True, funneled=True)
    print("Numero de pares:", len(lfw.pairs))
    print("Distribucion de etiquetas (1=genuino, 0=impostor):", np.bincount(lfw.target))

    pil_pairs = pairs_to_pil(lfw.pairs)

    embedder = FaceEmbedder(device=device)

    t0 = time.time()
    genuine_scores, impostor_scores, dropped = compute_pair_scores(embedder, pil_pairs, lfw.target)
    print(f"Tiempo de extraccion de embeddings: {time.time() - t0:.1f} s")

    print(f"Pares descartados (sin rostro detectado): {dropped} / {len(pil_pairs)}")
    print(f"Puntuaciones genuinas: {len(genuine_scores)} | Puntuaciones impostoras: {len(impostor_scores)}")

    eer, threshold_eer = compute_eer(genuine_scores, impostor_scores)
    print(f"EER: {eer:.4f}")
    print(f"Umbral de decision en el EER: {threshold_eer:.4f}")

    far, frr = far_frr_at_threshold(genuine_scores, impostor_scores, threshold_eer)
    print(f"FAR en el umbral del EER: {far:.4f}")
    print(f"FRR en el umbral del EER: {frr:.4f}")

    far_curve, tar_curve, thresholds = roc_curve(genuine_scores, impostor_scores)

    plt.figure(figsize=(6, 6))
    plt.plot(far_curve, tar_curve, label="Reconocimiento facial (FaceNet + LFW)")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Clasificador aleatorio")
    plt.xlabel("FAR (Tasa de Falsa Aceptacion)")
    plt.ylabel("TAR (Tasa de Verdadera Aceptacion)")
    plt.title(f"Curva ROC - Reconocimiento facial (EER = {eer:.3f})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    figures_dir = os.path.join(RESULTS_DIR, "figures")
    tables_dir = os.path.join(RESULTS_DIR, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    fig_path = os.path.join(figures_dir, "facial_roc_baseline.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Figura guardada en {fig_path}")

    pd.DataFrame({"score": genuine_scores, "label": 1}).to_csv(
        os.path.join(tables_dir, "facial_genuine_scores.csv"), index=False)
    pd.DataFrame({"score": impostor_scores, "label": 0}).to_csv(
        os.path.join(tables_dir, "facial_impostor_scores.csv"), index=False)

    pd.DataFrame([{
        "modalidad": "facial",
        "eer": eer,
        "threshold_eer": threshold_eer,
        "far_en_eer": far,
        "frr_en_eer": frr,
        "n_genuine": len(genuine_scores),
        "n_impostor": len(impostor_scores),
        "pares_descartados": dropped,
    }]).to_csv(os.path.join(tables_dir, "facial_baseline_summary.csv"), index=False)

    print(f"Resultados guardados en {tables_dir}")


if __name__ == "__main__":
    main()
