"""
Pipeline de reconocimiento por voz.

Un modelo ECAPA-TDNN convierte cada locucion en un embedding de hablante,
que se compara mediante similitud coseno.

El dataset de evaluacion es LibriSpeech "test-clean", que se descarga solo
la primera vez que se usa. Como no trae pares ya preparados como LFW, aqui
se construyen a partir de los hablantes disponibles.
"""

import os
import random

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

try:
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "speechbrain no esta instalado. En Colab: !pip install speechbrain"
    ) from exc


class VoiceEmbedder:
    """Envoltorio del modelo preentrenado ECAPA-TDNN (spkrec-ecapa-voxceleb)
    de SpeechBrain para obtener embeddings de hablante."""

    def __init__(self, device=None, savedir="pretrained_models/spkrec-ecapa-voxceleb"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts={"device": self.device},
            # Copia los ficheros en vez de crear symlinks: en Windows, crear
            # symlinks requiere privilegios de administrador o "modo desarrollador".
            local_strategy=LocalStrategy.COPY,
        )

    def embed(self, waveform):
        """Calcula el embedding de un audio."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        with torch.no_grad():
            emb = self.model.encode_batch(waveform.to(self.device))
        return emb.squeeze(1)

    def embed_grad(self, waveform):
        """Igual que embed() pero permitiendo gradientes respecto a la
        entrada, necesario para los ataques adversariales."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        emb = self.model.encode_batch(waveform.to(self.device))
        return emb.squeeze(1)


def load_librispeech(root="./data", subset="test-clean"):
    """Descarga (si es necesario) y carga LibriSpeech test-clean."""
    return torchaudio.datasets.LIBRISPEECH(root=root, url=subset, download=True)


def load_waveform(dataset, idx):
    """Carga el audio de la locucion idx-esima, a 16 kHz."""
    path, sample_rate, *_ = dataset.get_metadata(idx)
    full_path = os.path.join(dataset._archive, path)
    wav, sr = sf.read(full_path, dtype="float32")
    assert sr == sample_rate, f"sample rate inesperado: {sr} != {sample_rate}"
    return torch.from_numpy(wav)


def build_speaker_index(dataset):
    """Agrupa los indices del dataset por identidad de hablante.

    Devuelve un dict {speaker_id: [indices]}.
    """
    index = {}
    for i in range(len(dataset)):
        _, _, _, speaker_id, _, _ = dataset.get_metadata(i)
        index.setdefault(speaker_id, []).append(i)
    return index


def build_trial_pairs(speaker_index, n_genuine=200, n_impostor=200, seed=42):
    """Construye pares genuino/impostor a partir del indice de hablantes,
    de forma analoga al protocolo de pares de LFW.

    Devuelve una lista de tuplas (idx_a, idx_b, label), label=1 genuino.
    """
    rng = random.Random(seed)
    speakers = [s for s, idxs in speaker_index.items() if len(idxs) >= 2]

    pairs = []
    # Pares genuinos: dos locuciones del mismo hablante.
    for _ in range(n_genuine):
        speaker = rng.choice(speakers)
        idx_a, idx_b = rng.sample(speaker_index[speaker], 2)
        pairs.append((idx_a, idx_b, 1))

    # Pares impostores: locuciones de dos hablantes distintos.
    for _ in range(n_impostor):
        speaker_a, speaker_b = rng.sample(speakers, 2)
        idx_a = rng.choice(speaker_index[speaker_a])
        idx_b = rng.choice(speaker_index[speaker_b])
        pairs.append((idx_a, idx_b, 0))

    rng.shuffle(pairs)
    return pairs


def compute_pair_scores(embedder: VoiceEmbedder, dataset, pairs):
    """Calcula la similitud coseno para cada par (idx_a, idx_b, label)."""
    genuine, impostor = [], []
    for idx_a, idx_b, label in pairs:
        waveform_a = load_waveform(dataset, idx_a)
        waveform_b = load_waveform(dataset, idx_b)
        emb_a = embedder.embed(waveform_a)
        emb_b = embedder.embed(waveform_b)
        score = F.cosine_similarity(emb_a, emb_b).item()
        if label == 1:
            genuine.append(score)
        else:
            impostor.append(score)
    return np.array(genuine), np.array(impostor)
