"""
02 - Ataques adversariales sobre reconocimiento facial (FGSM y PGD)

Ataca el sistema de reconocimiento facial evaluado en 01_facial_baseline.py
con FGSM y PGD, en sus dos objetivos:

  - Evasión: intenta que el sistema rechace a un usuario legitimo.
  - Suplantación: intenta que el sistema acepte a un atacante como si
    fuera otra persona.

Para cada valor de epsilon (cuanto se le permite cambiar a la imagen) se
mide la tasa de exito del ataque y la precision del sistema tras el
ataque, usando el umbral calculado en 01_facial_baseline.py.

Salidas:
  - results/tables/facial_adversarial_sweep.csv
  - results/figures/facial_asr_vs_epsilon.png
  - results/figures/facial_adversarial_example.png (ejemplo visual)

Uso:
    python 02_facial_adversarial.py
"""

import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

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
from face.pipeline import FaceEmbedder, load_lfw_pairs, pairs_to_pil

N_DODGE_PAIRS = 40        # pares genuinos usados para el ataque de evasion
N_IMPERSONATE_PAIRS = 40  # pares impostores usados para el ataque de suplantacion
EPSILONS = [0.01, 0.02, 0.04, 0.08, 0.12]  # valores de presupuesto de perturbacion a probar
PGD_ALPHA_FACTOR = 0.25   # tamaño de cada paso de PGD, relativo a epsilon
PGD_STEPS = 10
CLIP_MIN, CLIP_MAX = -1.0, 1.0


def load_threshold():
    """Reutiliza el umbral de decision (EER) calculado en el baseline (01)."""
    summary_path = os.path.join(RESULTS_DIR, "tables", "facial_baseline_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            "No se encuentra facial_baseline_summary.csv. Ejecuta primero "
            "01_facial_baseline.py para generar el umbral de decision."
        )
    df = pd.read_csv(summary_path)
    return float(df.loc[0, "threshold_eer"])


def build_attack_subset(embedder, pil_pairs, n_pairs):
    """Prepara los primeros n_pairs pares para el ataque: detecta la cara
    en las dos imagenes y calcula el embedding de la primera, que hara de
    plantilla. Los pares donde no se detecta rostro se descartan."""
    subset = []
    for img_a, img_b in pil_pairs[:n_pairs]:
        face_a = embedder.align(img_a)
        face_b = embedder.align(img_b)
        if face_a is None or face_b is None:
            continue
        face_a = face_a.unsqueeze(0)
        face_b = face_b.unsqueeze(0)
        with torch.no_grad():
            template = embedder.model(face_a.to(embedder.device))
        subset.append((template.detach(), face_b))
    return subset


def run_sweep(embedder, dodge_subset, impersonate_subset, threshold):
    model = embedder.model
    rows = []
    example_records = {}  # guarda un ejemplo de imagen atacada por cada ataque, para mostrarlo despues

    mid_eps = EPSILONS[len(EPSILONS) // 2]

    for attack_name, attack_fn, extra_kwargs in [
        ("FGSM", fgsm_attack, {}),
        ("PGD", pgd_attack, {"steps": PGD_STEPS}),
    ]:
        for mode, subset in [("dodge", dodge_subset), ("impersonate", impersonate_subset)]:
            for epsilon in EPSILONS:
                scores_before = []
                scores_after = []
                linf_perturbations = []

                kwargs = dict(extra_kwargs)
                if attack_fn is pgd_attack:
                    kwargs["alpha"] = epsilon * PGD_ALPHA_FACTOR

                for template, face_x in subset:
                    face_x = face_x.to(embedder.device)

                    with torch.no_grad():
                        score_before = F.cosine_similarity(model(face_x), template).item()

                    x_adv = attack_fn(
                        model, face_x, template, epsilon, mode=mode,
                        clip_min=CLIP_MIN, clip_max=CLIP_MAX, **kwargs,
                    )

                    with torch.no_grad():
                        score_after = F.cosine_similarity(model(x_adv), template).item()

                    scores_before.append(score_before)
                    scores_after.append(score_after)
                    linf_perturbations.append((x_adv - face_x).abs().max().item())

                    if (attack_name, mode) not in example_records and abs(epsilon - mid_eps) < 1e-9:
                        example_records[(attack_name, mode)] = {
                            "original": face_x.detach().cpu().clone(),
                            "adversarial": x_adv.detach().cpu().clone(),
                            "score_before": score_before,
                            "score_after": score_after,
                            "epsilon": epsilon,
                        }

                scores_before = np.array(scores_before)
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


def denormalize(face_tensor):
    """Convierte la imagen ya procesada por MTCNN de vuelta a un formato
    que se pueda visualizar normalmente."""
    img = face_tensor.squeeze(0).permute(1, 2, 0).numpy()
    img = (img * 128.0 + 127.5) / 255.0
    return np.clip(img, 0.0, 1.0)


def save_qualitative_example(example_records, figures_dir):
    """Guarda una figura comparando una imagen original con su version
    atacada, para ver a simple vista si el cambio es perceptible."""
    keys = [("FGSM", "impersonate"), ("PGD", "impersonate")]
    keys = [k for k in keys if k in example_records]
    if not keys:
        print("Aviso: no hay ejemplos cualitativos disponibles (subset vacio).")
        return

    fig, axes = plt.subplots(len(keys), 2, figsize=(6, 3 * len(keys)))
    if len(keys) == 1:
        axes = axes.reshape(1, 2)

    for row, (attack_name, mode) in enumerate(keys):
        rec = example_records[(attack_name, mode)]
        axes[row, 0].imshow(denormalize(rec["original"]))
        axes[row, 0].set_title(f"Original\nsim={rec['score_before']:.3f}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(denormalize(rec["adversarial"]))
        axes[row, 1].set_title(f"{attack_name} (eps={rec['epsilon']}) \nsim={rec['score_after']:.3f}")
        axes[row, 1].axis("off")

    plt.tight_layout()
    path = os.path.join(figures_dir, "facial_adversarial_example.png")
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
    path = os.path.join(figures_dir, "facial_asr_vs_epsilon.png")
    plt.savefig(path, dpi=150)
    print(f"Curvas ASR vs epsilon guardadas en {path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    threshold = load_threshold()
    print(f"Umbral de decision (EER del baseline): {threshold:.4f}")

    lfw = load_lfw_pairs(subset="test", resize=1.0, color=True, funneled=True)
    pil_pairs = pairs_to_pil(lfw.pairs)

    # Los pares de LFW estan agrupados: los primeros son genuinos (label=1) y
    # los ultimos son impostores (label=0).
    genuine_pairs = [p for p, label in zip(pil_pairs, lfw.target) if label == 1]
    impostor_pairs = [p for p, label in zip(pil_pairs, lfw.target) if label == 0]

    embedder = FaceEmbedder(device=device)

    t0 = time.time()
    dodge_subset = build_attack_subset(embedder, genuine_pairs, N_DODGE_PAIRS)
    impersonate_subset = build_attack_subset(embedder, impostor_pairs, N_IMPERSONATE_PAIRS)
    print(f"Subconjunto de evasion: {len(dodge_subset)} pares validos "
          f"(de {N_DODGE_PAIRS} solicitados)")
    print(f"Subconjunto de suplantacion: {len(impersonate_subset)} pares validos "
          f"(de {N_IMPERSONATE_PAIRS} solicitados)")

    df, example_records = run_sweep(embedder, dodge_subset, impersonate_subset, threshold)
    print(f"Tiempo total del barrido de ataques: {time.time() - t0:.1f} s")

    figures_dir = os.path.join(RESULTS_DIR, "figures")
    tables_dir = os.path.join(RESULTS_DIR, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    df.to_csv(os.path.join(tables_dir, "facial_adversarial_sweep.csv"), index=False)
    print(f"Tabla de resultados guardada en {os.path.join(tables_dir, 'facial_adversarial_sweep.csv')}")

    plot_asr_curves(df, figures_dir)
    save_qualitative_example(example_records, figures_dir)


if __name__ == "__main__":
    main()
