"""
07 - Defensa por preprocesado de entrada (sin reentrenar el modelo)

Primera mitigacion probada: suavizar un poco la imagen o el audio antes
de calcular el embedding, para intentar atenuar la perturbacion del
ataque sin tocar el modelo. En facial se aplica un desenfoque, en voz un
filtro que suaviza los cambios muy rapidos del sonido.

El atacante no conoce esta defensa al generar el ataque, igual que en
02_facial_adversarial.py y 04_voice_adversarial.py.

Para cada modalidad se mide:
  1. Si la defensa empeora el EER cuando no hay ningun ataque.
  2. Si la defensa reduce la tasa de exito de los ataques FGSM y PGD.

Salidas:
  - results/tables/defensa_preprocesado_facial_baseline.csv
  - results/tables/defensa_preprocesado_facial_ataques.csv
  - results/tables/defensa_preprocesado_voz_baseline.csv
  - results/tables/defensa_preprocesado_voz_ataques.csv
  - results/figures/defensa_preprocesado_asr.png

Uso:
    python 07_defensa_preprocesado.py
"""

import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.metrics import compute_eer, attack_success_rate
from common.attacks import fgsm_attack, pgd_attack
from face.pipeline import FaceEmbedder, load_lfw_pairs, pairs_to_pil
from voice.pipeline import VoiceEmbedder, load_librispeech, build_speaker_index, build_trial_pairs, load_waveform

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

EPSILONS = [0.01, 0.02, 0.04, 0.08, 0.12]
N_FACIAL_ATTACK_PAIRS = 40
N_VOICE_ATTACK_PAIRS = 8   # escala piloto, igual que la primera prueba de 04
PGD_STEPS_FACIAL = 10
PGD_STEPS_VOZ = 5
PGD_ALPHA_FACTOR = 0.25 if True else None


def blur_defense_facial(x):
    """Aplica un desenfoque suave a la imagen ya alineada."""
    return TF.gaussian_blur(x, kernel_size=[5, 5], sigma=[1.5, 1.5])


def lowpass_defense_voz(waveform, sample_rate=16000, cutoff_freq=4000):
    """Suaviza el audio, quitando los cambios muy rapidos donde suele
    concentrarse la perturbacion del ataque."""
    return torchaudio.functional.lowpass_biquad(waveform, sample_rate, cutoff_freq)


def facial_baseline_con_defensa(embedder):
    lfw = load_lfw_pairs(subset="test", resize=1.0, color=True, funneled=True)
    pil_pairs = pairs_to_pil(lfw.pairs)

    genuine, impostor = [], []
    dropped = 0
    t0 = time.time()
    for (img_a, img_b), label in zip(pil_pairs, lfw.target):
        face_a = embedder.align(img_a)
        face_b = embedder.align(img_b)
        if face_a is None or face_b is None:
            dropped += 1
            continue
        face_a = blur_defense_facial(face_a.unsqueeze(0))
        face_b = blur_defense_facial(face_b.unsqueeze(0))
        emb_a = embedder.embed(face_a.squeeze(0))
        emb_b = embedder.embed(face_b.squeeze(0))
        score = F.cosine_similarity(emb_a, emb_b).item()
        (genuine if label == 1 else impostor).append(score)
    print(f"[Facial][baseline+defensa] tiempo: {time.time()-t0:.1f} s, "
          f"descartados: {dropped}/{len(pil_pairs)}")

    eer, threshold = compute_eer(np.array(genuine), np.array(impostor))
    return eer, threshold, len(genuine), len(impostor)


def facial_ataques_con_defensa(embedder, threshold_sin_defensa):
    lfw = load_lfw_pairs(subset="test", resize=1.0, color=True, funneled=True)
    pil_pairs = pairs_to_pil(lfw.pairs)
    genuine_pairs = [p for p, l in zip(pil_pairs, lfw.target) if l == 1][:N_FACIAL_ATTACK_PAIRS]
    impostor_pairs = [p for p, l in zip(pil_pairs, lfw.target) if l == 0][:N_FACIAL_ATTACK_PAIRS]

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
    print(f"[Facial][ataques+defensa] subconjuntos: dodge={len(dodge_subset)}, "
          f"impersonate={len(impersonate_subset)}")

    model = embedder.model
    rows = []
    t0 = time.time()
    for attack_name, attack_fn, extra in [("FGSM", fgsm_attack, {}), ("PGD", pgd_attack, {"steps": PGD_STEPS_FACIAL})]:
        for mode, subset in [("dodge", dodge_subset), ("impersonate", impersonate_subset)]:
            for epsilon in EPSILONS:
                kwargs = dict(extra)
                if attack_fn is pgd_attack:
                    kwargs["alpha"] = epsilon * PGD_ALPHA_FACTOR

                scores_before, scores_after_defensa = [], []
                for template, x in subset:
                    with torch.no_grad():
                        scores_before.append(F.cosine_similarity(model(x), template).item())

                    x_adv = attack_fn(model, x, template, epsilon, mode=mode,
                                       clip_min=-1.0, clip_max=1.0, **kwargs)
                    x_adv_defendido = blur_defense_facial(x_adv)
                    with torch.no_grad():
                        score_after = F.cosine_similarity(model(x_adv_defendido), template).item()
                    scores_after_defensa.append(score_after)

                asr = attack_success_rate(
                    np.array(scores_before), np.array(scores_after_defensa),
                    threshold_sin_defensa, mode=mode)
                rows.append({"attack": attack_name, "mode": mode, "epsilon": epsilon, "asr_con_defensa": asr})
                print(f"[Facial+defensa][{attack_name}|{mode}|eps={epsilon:.3f}] ASR={asr:.3f}")
    print(f"[Facial][ataques+defensa] tiempo total: {time.time()-t0:.1f} s")
    return pd.DataFrame(rows)


def voz_baseline_con_defensa(embedder, dataset, pairs):
    genuine, impostor = [], []
    t0 = time.time()
    for idx_a, idx_b, label in pairs:
        wav_a = lowpass_defense_voz(load_waveform(dataset, idx_a))
        wav_b = lowpass_defense_voz(load_waveform(dataset, idx_b))
        emb_a = embedder.embed(wav_a)
        emb_b = embedder.embed(wav_b)
        score = F.cosine_similarity(emb_a, emb_b).item()
        (genuine if label == 1 else impostor).append(score)
    print(f"[Voz][baseline+defensa] tiempo: {time.time()-t0:.1f} s")
    eer, threshold = compute_eer(np.array(genuine), np.array(impostor))
    return eer, threshold, len(genuine), len(impostor)


def voz_ataques_con_defensa(embedder, dataset, genuine_pairs, impostor_pairs, threshold_sin_defensa):
    def build_subset(pairs):
        subset = []
        for idx_a, idx_b, _ in pairs:
            wav_a = load_waveform(dataset, idx_a)
            wav_b = load_waveform(dataset, idx_b)
            template = embedder.embed(wav_a).detach()
            subset.append((template, wav_b.unsqueeze(0)))
        return subset

    dodge_subset = build_subset(genuine_pairs)
    impersonate_subset = build_subset(impostor_pairs)

    model = embedder.embed_grad
    rows = []
    t0 = time.time()
    for attack_name, attack_fn, extra in [("FGSM", fgsm_attack, {}), ("PGD", pgd_attack, {"steps": PGD_STEPS_VOZ})]:
        for mode, subset in [("dodge", dodge_subset), ("impersonate", impersonate_subset)]:
            for epsilon in EPSILONS:
                kwargs = dict(extra)
                if attack_fn is pgd_attack:
                    kwargs["alpha"] = epsilon * PGD_ALPHA_FACTOR

                scores_before, scores_after_defensa = [], []
                for template, x in subset:
                    with torch.no_grad():
                        scores_before.append(F.cosine_similarity(model(x), template).item())

                    x_adv = attack_fn(model, x, template, epsilon, mode=mode,
                                       clip_min=-1.0, clip_max=1.0, **kwargs)
                    x_adv_defendido = lowpass_defense_voz(x_adv)
                    with torch.no_grad():
                        score_after = F.cosine_similarity(model(x_adv_defendido), template).item()
                    scores_after_defensa.append(score_after)

                asr = attack_success_rate(
                    np.array(scores_before), np.array(scores_after_defensa),
                    threshold_sin_defensa, mode=mode)
                rows.append({"attack": attack_name, "mode": mode, "epsilon": epsilon, "asr_con_defensa": asr})
                print(f"[Voz+defensa][{attack_name}|{mode}|eps={epsilon:.3f}] ASR={asr:.3f}")
    print(f"[Voz][ataques+defensa] tiempo total: {time.time()-t0:.1f} s")
    return pd.DataFrame(rows)


def main():
    facial_summary = pd.read_csv(os.path.join(TABLES_DIR, "facial_baseline_summary.csv"))
    threshold_facial = float(facial_summary.loc[0, "threshold_eer"])
    eer_original_facial = float(facial_summary.loc[0, "eer"])

    face_embedder = FaceEmbedder(device="cpu")

    eer_defensa, threshold_defensa, n_gen, n_imp = facial_baseline_con_defensa(face_embedder)
    pd.DataFrame([{
        "modalidad": "facial", "eer_sin_defensa": eer_original_facial,
        "eer_con_defensa": eer_defensa, "n_genuine": n_gen, "n_impostor": n_imp,
    }]).to_csv(os.path.join(TABLES_DIR, "defensa_preprocesado_facial_baseline.csv"), index=False)
    print(f"Facial: EER sin defensa={eer_original_facial:.4f}  EER con defensa={eer_defensa:.4f}")

    df_facial_ataques = facial_ataques_con_defensa(face_embedder, threshold_facial)
    facial_sweep_original = pd.read_csv(os.path.join(TABLES_DIR, "facial_adversarial_sweep.csv"))
    df_facial_ataques = df_facial_ataques.merge(
        facial_sweep_original[["attack", "mode", "epsilon", "asr"]].rename(columns={"asr": "asr_sin_defensa"}),
        on=["attack", "mode", "epsilon"])
    df_facial_ataques.to_csv(os.path.join(TABLES_DIR, "defensa_preprocesado_facial_ataques.csv"), index=False)

    voice_summary = pd.read_csv(os.path.join(TABLES_DIR, "voice_baseline_summary.csv"))
    threshold_voz = float(voice_summary.loc[0, "threshold_eer"])
    eer_original_voz = float(voice_summary.loc[0, "eer"])

    voice_embedder = VoiceEmbedder(device="cpu")
    dataset = load_librispeech(root=DATA_DIR, subset="test-clean")
    speaker_index = build_speaker_index(dataset)

    pairs_baseline = build_trial_pairs(speaker_index, n_genuine=100, n_impostor=100, seed=7)
    eer_defensa_voz, _, n_gen_v, n_imp_v = voz_baseline_con_defensa(voice_embedder, dataset, pairs_baseline)
    pd.DataFrame([{
        "modalidad": "voz", "eer_sin_defensa": eer_original_voz,
        "eer_con_defensa": eer_defensa_voz, "n_genuine": n_gen_v, "n_impostor": n_imp_v,
    }]).to_csv(os.path.join(TABLES_DIR, "defensa_preprocesado_voz_baseline.csv"), index=False)
    print(f"Voz: EER sin defensa={eer_original_voz:.4f}  EER con defensa={eer_defensa_voz:.4f}")

    pairs_ataque = build_trial_pairs(
        speaker_index, n_genuine=N_VOICE_ATTACK_PAIRS, n_impostor=N_VOICE_ATTACK_PAIRS, seed=99)
    genuine_pairs = [p for p in pairs_ataque if p[2] == 1]
    impostor_pairs = [p for p in pairs_ataque if p[2] == 0]

    df_voz_ataques = voz_ataques_con_defensa(voice_embedder, dataset, genuine_pairs, impostor_pairs, threshold_voz)
    voz_sweep_original = pd.read_csv(os.path.join(TABLES_DIR, "voice_adversarial_sweep.csv"))
    df_voz_ataques = df_voz_ataques.merge(
        voz_sweep_original[["attack", "mode", "epsilon", "asr"]].rename(columns={"asr": "asr_sin_defensa"}),
        on=["attack", "mode", "epsilon"])
    df_voz_ataques.to_csv(os.path.join(TABLES_DIR, "defensa_preprocesado_voz_ataques.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, mode, title in [
        (axes[0], "dodge", "Evasión (dodging)"),
        (axes[1], "impersonate", "Suplantación (impersonation)"),
    ]:
        for df, modalidad, color in [(df_facial_ataques, "facial", "tab:blue"),
                                      (df_voz_ataques, "voz", "tab:red")]:
            subset = df[(df["mode"] == mode) & (df["attack"] == "PGD")].sort_values("epsilon")
            ax.plot(subset["epsilon"], subset["asr_sin_defensa"], marker="o", linestyle="--",
                    color=color, label=f"{modalidad} - PGD sin defensa")
            ax.plot(subset["epsilon"], subset["asr_con_defensa"], marker="s", linestyle="-",
                    color=color, label=f"{modalidad} - PGD con defensa")
        ax.set_xlabel("epsilon")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Attack Success Rate (ASR)")
    plt.suptitle("Efecto de la defensa por preprocesado sobre PGD")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "defensa_preprocesado_asr.png"), dpi=150)
    print("Figura guardada en", os.path.join(FIGURES_DIR, "defensa_preprocesado_asr.png"))


if __name__ == "__main__":
    main()
