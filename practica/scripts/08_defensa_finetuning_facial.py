"""
08 - Segunda mitigacion: reentrenar la ultima capa del modelo facial

Segundo intento de mitigacion, esta vez modificando el propio modelo en
vez de solo la entrada. Se congela toda la red y solo se reentrena su
ultima capa (menos del 3.3% de los parametros totales), mostrandole
ejemplos ya atacados durante el entrenamiento, para que aprenda a ser
menos sensible a ese tipo de perturbacion.

El entrenamiento usa un conjunto de imagenes de LFW distinto al que se
usa para evaluar despues, para no hacer trampa.

Tras el ajuste, se vuelve a medir el EER y la robustez del modelo con el
mismo protocolo que en 01 y 02, para comparar el modelo ajustado con el
original.

Salidas:
  - results/tables/defensa_finetuning_facial_baseline.csv
  - results/tables/defensa_finetuning_facial_ataques.csv
  - results/figures/defensa_finetuning_asr.png
  - results/models/facenet_finetuned_last_layer.pt (pesos ajustados)

Uso:
    python 08_defensa_finetuning_facial.py
"""

import os
import random
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.metrics import compute_eer, far_frr_at_threshold, attack_success_rate
from common.attacks import fgsm_attack, pgd_attack
from face.pipeline import FaceEmbedder, load_lfw_pairs, pairs_to_pil, compute_pair_scores

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

TRAIN_N_GENUINE = 60
TRAIN_N_IMPOSTOR = 60
EPOCHS = 3
LR = 1e-4
TRAIN_EPSILON = 0.03
MARGIN_POS = 0.6   # similitud minima deseada entre pares genuinos
MARGIN_NEG = 0.3   # similitud maxima deseada entre pares impostores
LAMBDA_ADV = 1.0   # peso del termino de robustez adversarial frente al termino limpio
SEED = 2024

# Mismos epsilon, N y pasos de PGD que en 01/02 para que el resultado sea comparable.
EPSILONS = [0.01, 0.02, 0.04, 0.08, 0.12]
N_ATTACK_PAIRS = 40
PGD_STEPS = 10
PGD_ALPHA_FACTOR = 0.25


def align_pairs(embedder, pil_pairs):
    """Alinea ambas imagenes de cada par; descarta los pares sin rostro
    detectado en alguna de las dos imagenes."""
    aligned = []
    for img_a, img_b in pil_pairs:
        face_a = embedder.align(img_a)
        face_b = embedder.align(img_b)
        if face_a is None or face_b is None:
            continue
        aligned.append((face_a.unsqueeze(0), face_b.unsqueeze(0)))
    return aligned


def entrenar(embedder):
    model = embedder.model

    for p in model.parameters():
        p.requires_grad_(False)
    trainable_params = list(model.last_linear.parameters()) + list(model.last_bn.parameters())
    for p in trainable_params:
        p.requires_grad_(True)
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parametros entrenables: {n_trainable} / {n_total} "
          f"({100 * n_trainable / n_total:.1f}%)")

    optimizer = torch.optim.Adam(trainable_params, lr=LR)

    lfw_train = load_lfw_pairs(subset="train", resize=1.0, color=True, funneled=True)
    pil_pairs = pairs_to_pil(lfw_train.pairs)
    genuine_pil = [p for p, l in zip(pil_pairs, lfw_train.target) if l == 1][:TRAIN_N_GENUINE]
    impostor_pil = [p for p, l in zip(pil_pairs, lfw_train.target) if l == 0][:TRAIN_N_IMPOSTOR]

    genuine_pairs = align_pairs(embedder, genuine_pil)
    impostor_pairs = align_pairs(embedder, impostor_pil)
    print(f"Pares de entrenamiento validos: {len(genuine_pairs)} genuinos, "
          f"{len(impostor_pairs)} impostores")

    rng = random.Random(SEED)
    history = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        rng.shuffle(genuine_pairs)
        rng.shuffle(impostor_pairs)
        losses = []

        for a, p in genuine_pairs:
            with torch.no_grad():
                e_a_fixed = model(a)
            p_adv = fgsm_attack(model, p, e_a_fixed, TRAIN_EPSILON, mode="dodge",
                                 clip_min=-1.0, clip_max=1.0)

            optimizer.zero_grad()
            e_a = model(a)
            e_p = model(p)
            e_p_adv = model(p_adv)
            sim_clean = F.cosine_similarity(e_a, e_p)
            sim_adv = F.cosine_similarity(e_a, e_p_adv)
            loss = F.relu(MARGIN_POS - sim_clean).mean() + LAMBDA_ADV * F.relu(MARGIN_POS - sim_adv).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        for a, n in impostor_pairs:
            with torch.no_grad():
                e_a_fixed = model(a)
            n_adv = fgsm_attack(model, n, e_a_fixed, TRAIN_EPSILON, mode="impersonate",
                                 clip_min=-1.0, clip_max=1.0)

            optimizer.zero_grad()
            e_a = model(a)
            e_n = model(n)
            e_n_adv = model(n_adv)
            sim_clean = F.cosine_similarity(e_a, e_n)
            sim_adv = F.cosine_similarity(e_a, e_n_adv)
            loss = F.relu(sim_clean - MARGIN_NEG).mean() + LAMBDA_ADV * F.relu(sim_adv - MARGIN_NEG).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss_medio": mean_loss})
        print(f"Epoch {epoch + 1}/{EPOCHS}: loss medio = {mean_loss:.4f}")

    print(f"Tiempo total de entrenamiento: {time.time() - t0:.1f} s")

    torch.save({
        "last_linear": model.last_linear.state_dict(),
        "last_bn": model.last_bn.state_dict(),
    }, os.path.join(MODELS_DIR, "facenet_finetuned_last_layer.pt"))

    return pd.DataFrame(history)


def reevaluar_baseline(embedder):
    lfw = load_lfw_pairs(subset="test", resize=1.0, color=True, funneled=True)
    pil_pairs = pairs_to_pil(lfw.pairs)
    genuine_scores, impostor_scores, dropped = compute_pair_scores(embedder, pil_pairs, lfw.target)
    eer, threshold = compute_eer(genuine_scores, impostor_scores)
    far, frr = far_frr_at_threshold(genuine_scores, impostor_scores, threshold)
    return eer, threshold, far, frr, len(genuine_scores), len(impostor_scores)


def reevaluar_ataques(embedder, threshold):
    lfw = load_lfw_pairs(subset="test", resize=1.0, color=True, funneled=True)
    pil_pairs = pairs_to_pil(lfw.pairs)
    genuine_pairs = [p for p, l in zip(pil_pairs, lfw.target) if l == 1][:N_ATTACK_PAIRS]
    impostor_pairs = [p for p, l in zip(pil_pairs, lfw.target) if l == 0][:N_ATTACK_PAIRS]

    def build_subset(pairs):
        subset = []
        for img_a, img_b in pairs:
            face_a = embedder.align(img_a)
            face_b = embedder.align(img_b)
            if face_a is None or face_b is None:
                continue
            with torch.no_grad():
                template = embedder.model(face_a.unsqueeze(0))
            subset.append((template.detach(), face_b.unsqueeze(0)))
        return subset

    dodge_subset = build_subset(genuine_pairs)
    impersonate_subset = build_subset(impostor_pairs)

    model = embedder.model
    rows = []
    for attack_name, attack_fn, extra in [("FGSM", fgsm_attack, {}), ("PGD", pgd_attack, {"steps": PGD_STEPS})]:
        for mode, subset in [("dodge", dodge_subset), ("impersonate", impersonate_subset)]:
            for epsilon in EPSILONS:
                kwargs = dict(extra)
                if attack_fn is pgd_attack:
                    kwargs["alpha"] = epsilon * PGD_ALPHA_FACTOR

                scores_before, scores_after = [], []
                for template, x in subset:
                    with torch.no_grad():
                        scores_before.append(F.cosine_similarity(model(x), template).item())
                    x_adv = attack_fn(model, x, template, epsilon, mode=mode,
                                       clip_min=-1.0, clip_max=1.0, **kwargs)
                    with torch.no_grad():
                        scores_after.append(F.cosine_similarity(model(x_adv), template).item())

                asr = attack_success_rate(np.array(scores_before), np.array(scores_after), threshold, mode=mode)
                rows.append({"attack": attack_name, "mode": mode, "epsilon": epsilon, "asr_modelo_ajustado": asr})
                print(f"[Ajustado][{attack_name}|{mode}|eps={epsilon:.3f}] ASR={asr:.3f}")

    return pd.DataFrame(rows)


def main():
    embedder = FaceEmbedder(device="cpu")

    print("=== Entrenamiento (ajuste adversarial de la ultima capa) ===")
    history = entrenar(embedder)
    history.to_csv(os.path.join(TABLES_DIR, "defensa_finetuning_historial_loss.csv"), index=False)

    print("\n=== Reevaluacion del baseline con el modelo ajustado ===")
    eer, threshold, far, frr, n_gen, n_imp = reevaluar_baseline(embedder)
    facial_summary_original = pd.read_csv(os.path.join(TABLES_DIR, "facial_baseline_summary.csv"))
    eer_original = float(facial_summary_original.loc[0, "eer"])
    print(f"EER original: {eer_original:.4f}  |  EER modelo ajustado: {eer:.4f}")

    pd.DataFrame([{
        "eer_original": eer_original,
        "eer_modelo_ajustado": eer,
        "threshold_modelo_ajustado": threshold,
        "far_modelo_ajustado": far,
        "frr_modelo_ajustado": frr,
        "n_genuine": n_gen,
        "n_impostor": n_imp,
    }]).to_csv(os.path.join(TABLES_DIR, "defensa_finetuning_facial_baseline.csv"), index=False)

    print("\n=== Reevaluacion de los ataques con el modelo ajustado ===")
    df_ataques = reevaluar_ataques(embedder, threshold)
    sweep_original = pd.read_csv(os.path.join(TABLES_DIR, "facial_adversarial_sweep.csv"))
    df_ataques = df_ataques.merge(
        sweep_original[["attack", "mode", "epsilon", "asr"]].rename(columns={"asr": "asr_modelo_original"}),
        on=["attack", "mode", "epsilon"])
    df_ataques.to_csv(os.path.join(TABLES_DIR, "defensa_finetuning_facial_ataques.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, mode, title in [
        (axes[0], "dodge", "Evasión (dodging)"),
        (axes[1], "impersonate", "Suplantación (impersonation)"),
    ]:
        for attack_name, color in [("FGSM", "tab:green"), ("PGD", "tab:orange")]:
            subset = df_ataques[(df_ataques["mode"] == mode) & (df_ataques["attack"] == attack_name)].sort_values("epsilon")
            ax.plot(subset["epsilon"], subset["asr_modelo_original"], marker="o", linestyle="--",
                    color=color, label=f"{attack_name} - modelo original")
            ax.plot(subset["epsilon"], subset["asr_modelo_ajustado"], marker="s", linestyle="-",
                    color=color, label=f"{attack_name} - modelo ajustado")
        ax.set_xlabel("epsilon")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Attack Success Rate (ASR)")
    plt.suptitle("Efecto del ajuste adversarial de la última capa sobre la ASR (facial)")
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "defensa_finetuning_asr.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigura guardada en {fig_path}")


if __name__ == "__main__":
    main()
