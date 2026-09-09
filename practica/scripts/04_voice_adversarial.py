"""
04 - Ataques adversariales sobre reconocimiento por voz (FGSM y PGD)

Es el equivalente, para voz, del script 02: ataca el sistema evaluado en
03_voice_baseline.py con FGSM y PGD, en sus dos objetivos (evasion y
suplantacion), y mide la tasa de exito del ataque y la precision del
sistema tras el ataque, para cada valor de epsilon.

La perturbacion se aplica directamente sobre el audio.

Salidas:
  - results/tables/voice_adversarial_sweep.csv
  - results/figures/voice_asr_vs_epsilon.png
  - results/figures/voice_adversarial_example.png (ejemplo de audio)

Uso:
    python 04_voice_adversarial.py
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
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.metrics import attack_success_rate, adversarial_accuracy
from common.attacks import fgsm_attack, pgd_attack
from voice.pipeline import VoiceEmbedder, load_librispeech, build_speaker_index, build_trial_pairs, load_waveform

# Procesar audio es mucho mas lento que procesar imagenes, asi que este
# script tarda bastante mas que 02_facial_adversarial.py (ver
# 06_comparativa_tiempos.py). Antes de esta version se hizo una prueba
# piloto mas pequeña, con N=8, guardada en voice_adversarial_sweep_n8.csv.
N_DODGE_PAIRS = 40
N_IMPERSONATE_PAIRS = 40
EPSILONS = [0.01, 0.02, 0.04, 0.08, 0.12]  # mismos valores que en facial, para poder comparar
PGD_ALPHA_FACTOR = 0.3
PGD_STEPS = 5
CLIP_MIN, CLIP_MAX = -1.0, 1.0
SEED = 123


def load_threshold():
    """Reutiliza el umbral de decision (EER) calculado en el baseline (03)."""
    summary_path = os.path.join(RESULTS_DIR, "tables", "voice_baseline_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            "No se encuentra voice_baseline_summary.csv. Ejecuta primero "
            "03_voice_baseline.py para generar el umbral de decision."
        )
    df = pd.read_csv(summary_path)
    return float(df.loc[0, "threshold_eer"])


def build_attack_subset(embedder, dataset, pairs):
    """Calcula el embedding de la primera locucion de cada par (plantilla) y
    carga la forma de onda de la segunda (la que se va a atacar)."""
    subset = []
    for idx_a, idx_b, _ in pairs:
        waveform_a = load_waveform(dataset, idx_a)
        waveform_b = load_waveform(dataset, idx_b)
        template = embedder.embed(waveform_a).detach()
        x = waveform_b.unsqueeze(0)
        subset.append((template, x))
    return subset


def compute_baseline_scores(model, subset):
    """Similitud antes de atacar. Es siempre la misma para cada par, asi
    que se calcula una sola vez y se reutiliza, en lugar de recalcularla
    para cada ataque y cada epsilon."""
    scores = []
    for template, x in subset:
        with torch.no_grad():
            scores.append(F.cosine_similarity(model(x), template).item())
    return np.array(scores)


def run_sweep(embedder, dodge_subset, impersonate_subset, threshold):
    model = embedder.embed_grad
    rows = []
    example_records = {}
    mid_eps = EPSILONS[len(EPSILONS) // 2]

    baseline_scores = {
        "dodge": compute_baseline_scores(model, dodge_subset),
        "impersonate": compute_baseline_scores(model, impersonate_subset),
    }

    for attack_name, attack_fn, extra_kwargs in [
        ("FGSM", fgsm_attack, {}),
        ("PGD", pgd_attack, {"steps": PGD_STEPS}),
    ]:
        for mode, subset in [("dodge", dodge_subset), ("impersonate", impersonate_subset)]:
            for epsilon in EPSILONS:
                scores_before = baseline_scores[mode]
                scores_after, linf_perturbations = [], []

                kwargs = dict(extra_kwargs)
                if attack_fn is pgd_attack:
                    kwargs["alpha"] = epsilon * PGD_ALPHA_FACTOR

                for (template, x), score_before in zip(subset, scores_before):
                    x_adv = attack_fn(
                        model, x, template, epsilon, mode=mode,
                        clip_min=CLIP_MIN, clip_max=CLIP_MAX, **kwargs,
                    )

                    with torch.no_grad():
                        score_after = F.cosine_similarity(model(x_adv), template).item()

                    scores_after.append(score_after)
                    linf_perturbations.append((x_adv - x).abs().max().item())

                    if (attack_name, mode) not in example_records and abs(epsilon - mid_eps) < 1e-9:
                        example_records[(attack_name, mode)] = {
                            "original": x.detach().cpu().squeeze(0).numpy(),
                            "adversarial": x_adv.detach().cpu().squeeze(0).numpy(),
                            "score_before": score_before,
                            "score_after": score_after,
                            "epsilon": epsilon,
                        }

                scores_after = np.array(scores_after)

                asr = attack_success_rate(scores_before, scores_after, threshold, mode=mode)
                is_genuine = (mode == "dodge")
                adv_acc = adversarial_accuracy(scores_after, threshold, is_genuine=is_genuine)

                rows.append({
                    "attack": attack_name,
                    "mode": mode,
                    "epsilon": epsilon,
                    "asr": asr,
                    "adversarial_accuracy": adv_acc,
                    "mean_score_before": scores_before.mean(),
                    "mean_score_after": scores_after.mean(),
                    "mean_linf_perturbation": float(np.mean(linf_perturbations)),
                    "n_samples": len(subset),
                })
                print(f"[{attack_name} | {mode} | eps={epsilon:.3f}] "
                      f"ASR={asr:.3f}  precision_adv={adv_acc:.3f}")

    return pd.DataFrame(rows), example_records


def save_qualitative_example(example_records, figures_dir):
    keys = [("FGSM", "impersonate"), ("PGD", "impersonate")]
    keys = [k for k in keys if k in example_records]
    if not keys:
        print("Aviso: no hay ejemplos cualitativos disponibles.")
        return

    fig, axes = plt.subplots(len(keys), 1, figsize=(9, 3 * len(keys)))
    if len(keys) == 1:
        axes = [axes]

    for ax, (attack_name, mode) in zip(axes, keys):
        rec = example_records[(attack_name, mode)]
        ax.plot(rec["original"], alpha=0.6, label=f"Original (sim={rec['score_before']:.3f})")
        ax.plot(rec["adversarial"], alpha=0.6,
                 label=f"{attack_name} eps={rec['epsilon']} (sim={rec['score_after']:.3f})")
        ax.set_title(f"Suplantacion - {attack_name}")
        ax.set_xlabel("Muestra (16 kHz)")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    path = os.path.join(figures_dir, "voice_adversarial_example.png")
    plt.savefig(path, dpi=150)
    print(f"Ejemplo cualitativo guardado en {path}")


def plot_asr_curves(df, figures_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, mode, title in [
        (axes[0], "dodge", "Evasion (dodging) -> falso rechazo"),
        (axes[1], "impersonate", "Suplantacion (impersonation) -> falsa aceptacion"),
    ]:
        for attack_name in ["FGSM", "PGD"]:
            subset = df[(df["attack"] == attack_name) & (df["mode"] == mode)]
            ax.plot(subset["epsilon"], subset["asr"], marker="o", label=attack_name)
        ax.set_xlabel("epsilon (perturbacion L-infinito)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Attack Success Rate (ASR)")
    plt.tight_layout()
    path = os.path.join(figures_dir, "voice_asr_vs_epsilon.png")
    plt.savefig(path, dpi=150)
    print(f"Curvas ASR vs epsilon guardadas en {path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    threshold = load_threshold()
    print(f"Umbral de decision (EER del baseline): {threshold:.4f}")

    dataset = load_librispeech(root=DATA_DIR, subset="test-clean")
    speaker_index = build_speaker_index(dataset)

    pairs = build_trial_pairs(
        speaker_index, n_genuine=N_DODGE_PAIRS, n_impostor=N_IMPERSONATE_PAIRS, seed=SEED)
    genuine_pairs = [p for p in pairs if p[2] == 1]
    impostor_pairs = [p for p in pairs if p[2] == 0]

    embedder = VoiceEmbedder(device=device)

    t0 = time.time()
    dodge_subset = build_attack_subset(embedder, dataset, genuine_pairs)
    impersonate_subset = build_attack_subset(embedder, dataset, impostor_pairs)
    print(f"Subconjunto de evasion: {len(dodge_subset)} pares")
    print(f"Subconjunto de suplantacion: {len(impersonate_subset)} pares")

    df, example_records = run_sweep(embedder, dodge_subset, impersonate_subset, threshold)
    print(f"Tiempo total del barrido de ataques: {time.time() - t0:.1f} s")

    figures_dir = os.path.join(RESULTS_DIR, "figures")
    tables_dir = os.path.join(RESULTS_DIR, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    df.to_csv(os.path.join(tables_dir, "voice_adversarial_sweep.csv"), index=False)
    print(f"Tabla de resultados guardada en {os.path.join(tables_dir, 'voice_adversarial_sweep.csv')}")

    plot_asr_curves(df, figures_dir)
    save_qualitative_example(example_records, figures_dir)


if __name__ == "__main__":
    main()
