"""
03 - Rendimiento del reconocimiento por voz sin ningun ataque

Es el equivalente, para voz, del script 01: mide como funciona el sistema
en condiciones normales (FAR, FRR, EER, curva ROC), antes de aplicar
ningun ataque. Sirve de referencia para 04_voice_adversarial.py.

Usa el dataset LibriSpeech "test-clean", que se descarga solo la primera
vez que se ejecuta. Como no trae pares ya preparados como LFW, aqui se
construyen pares genuinos (misma persona) e impostores (personas
distintas) a partir de los hablantes disponibles.

Uso:
    python 03_voice_baseline.py
"""

import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.metrics import roc_curve, compute_eer, far_frr_at_threshold
from voice.pipeline import (
    VoiceEmbedder, load_librispeech, build_speaker_index,
    build_trial_pairs, compute_pair_scores,
)

N_GENUINE_PAIRS = 200
N_IMPOSTOR_PAIRS = 200
SEED = 42


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    dataset = load_librispeech(root=DATA_DIR, subset="test-clean")
    print("Numero de locuciones:", len(dataset))

    speaker_index = build_speaker_index(dataset)
    print("Numero de hablantes distintos:", len(speaker_index))

    pairs = build_trial_pairs(
        speaker_index, n_genuine=N_GENUINE_PAIRS, n_impostor=N_IMPOSTOR_PAIRS, seed=SEED)
    print(f"Pares construidos: {N_GENUINE_PAIRS} genuinos, {N_IMPOSTOR_PAIRS} impostores")

    embedder = VoiceEmbedder(device=device)

    t0 = time.time()
    genuine_scores, impostor_scores = compute_pair_scores(embedder, dataset, pairs)
    print(f"Tiempo de extraccion de embeddings: {time.time() - t0:.1f} s")
    print(f"Puntuaciones genuinas: {len(genuine_scores)} | Puntuaciones impostoras: {len(impostor_scores)}")

    eer, threshold_eer = compute_eer(genuine_scores, impostor_scores)
    print(f"EER: {eer:.4f}")
    print(f"Umbral de decision en el EER: {threshold_eer:.4f}")

    far, frr = far_frr_at_threshold(genuine_scores, impostor_scores, threshold_eer)
    print(f"FAR en el umbral del EER: {far:.4f}")
    print(f"FRR en el umbral del EER: {frr:.4f}")

    far_curve, tar_curve, thresholds = roc_curve(genuine_scores, impostor_scores)

    plt.figure(figsize=(6, 6))
    plt.plot(far_curve, tar_curve, label="Reconocimiento por voz (ECAPA-TDNN + LibriSpeech)")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Clasificador aleatorio")
    plt.xlabel("FAR (Tasa de Falsa Aceptacion)")
    plt.ylabel("TAR (Tasa de Verdadera Aceptacion)")
    plt.title(f"Curva ROC - Reconocimiento por voz (EER = {eer:.3f})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    figures_dir = os.path.join(RESULTS_DIR, "figures")
    tables_dir = os.path.join(RESULTS_DIR, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    fig_path = os.path.join(figures_dir, "voice_roc_baseline.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Figura guardada en {fig_path}")

    pd.DataFrame({"score": genuine_scores, "label": 1}).to_csv(
        os.path.join(tables_dir, "voice_genuine_scores.csv"), index=False)
    pd.DataFrame({"score": impostor_scores, "label": 0}).to_csv(
        os.path.join(tables_dir, "voice_impostor_scores.csv"), index=False)

    pd.DataFrame([{
        "modalidad": "voz",
        "eer": eer,
        "threshold_eer": threshold_eer,
        "far_en_eer": far,
        "frr_en_eer": frr,
        "n_genuine": len(genuine_scores),
        "n_impostor": len(impostor_scores),
    }]).to_csv(os.path.join(tables_dir, "voice_baseline_summary.csv"), index=False)

    print(f"Resultados guardados en {tables_dir}")


if __name__ == "__main__":
    main()
