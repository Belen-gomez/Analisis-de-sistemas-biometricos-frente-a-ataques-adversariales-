"""
Pipeline de reconocimiento facial.

Detecta y alinea la cara con MTCNN, calcula su embedding con FaceNet, y
compara embeddings con similitud coseno.

El dataset de evaluacion es LFW, que se descarga solo la primera vez que
se usa, a traves de scikit-learn.
"""

import numpy as np
import torch
from PIL import Image
from sklearn.datasets import fetch_lfw_pairs

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "facenet-pytorch no esta instalado. En Colab: !pip install facenet-pytorch"
    ) from exc


class FaceEmbedder:
    """Envoltorio de MTCNN + InceptionResnetV1 (FaceNet) para obtener
    embeddings faciales de 512 dimensiones a partir de imagenes RGB."""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(image_size=160, margin=14, post_process=True, device=self.device)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def align(self, pil_image):
        """Detecta y alinea el rostro. Devuelve un tensor (3,160,160)
        normalizado a [-1,1], o None si no se detecta ningun rostro."""
        return self.mtcnn(pil_image)

    def embed(self, face_tensor):
        """face_tensor: (3,160,160) o (B,3,160,160) ya alineado y normalizado.
        Devuelve el embedding (B,512). No calcula gradientes."""
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        with torch.no_grad():
            return self.model(face_tensor.to(self.device))

    def embed_no_grad_free(self, face_tensor):
        """Igual que embed(), pero permitiendo calcular gradientes. Los
        ataques adversariales lo necesitan para funcionar."""
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        return self.model(face_tensor.to(self.device))


def load_lfw_pairs(subset="test", resize=1.0, color=True, funneled=True):
    """Carga los pares de imagenes de LFW ya preparados por scikit-learn,
    con su etiqueta (1 = misma persona, 0 = personas distintas)."""
    data = fetch_lfw_pairs(subset=subset, resize=resize, color=color, funneled=funneled)
    return data


def pairs_to_pil(pairs_array):
    """Convierte los pares de imagenes de scikit-learn a un formato de
    imagen normal, con el que ya se puede trabajar."""
    imgs = pairs_array.astype(np.uint8) if pairs_array.max() > 1.5 else (pairs_array * 255).astype(np.uint8)
    out = []
    for pair in imgs:
        img_a = Image.fromarray(pair[0])
        img_b = Image.fromarray(pair[1])
        out.append((img_a, img_b))
    return out


def compute_pair_scores(embedder: FaceEmbedder, pil_pairs, labels):
    """Calcula la similitud coseno para cada par y separa las puntuaciones
    en genuinas e impostoras. Los pares donde MTCNN no detecta rostro en
    alguna de las dos imagenes se descartan (se informa cuantos se pierden)."""
    genuine, impostor = [], []
    dropped = 0
    for (img_a, img_b), label in zip(pil_pairs, labels):
        face_a = embedder.align(img_a)
        face_b = embedder.align(img_b)
        if face_a is None or face_b is None:
            dropped += 1
            continue
        emb_a = embedder.embed(face_a)
        emb_b = embedder.embed(face_b)
        score = torch.nn.functional.cosine_similarity(emb_a, emb_b).item()
        if label == 1:
            genuine.append(score)
        else:
            impostor.append(score)
    return np.array(genuine), np.array(impostor), dropped
