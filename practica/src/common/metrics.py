"""
Metricas para evaluar los sistemas de verificacion facial y de voz.

Todas las funciones trabajan con puntuaciones de similitud: cuanto mas
alta la puntuacion, mas se parecen la muestra y la plantilla. Se acepta al
usuario si esa puntuacion es igual o mayor que el umbral de decision.
"""

import numpy as np
from sklearn.metrics import roc_curve as _sk_roc_curve


def far_frr_at_threshold(genuine_scores, impostor_scores, threshold):
    """FAR y FRR para un unico umbral de decision."""
    genuine_scores = np.asarray(genuine_scores)
    impostor_scores = np.asarray(impostor_scores)
    far = np.mean(impostor_scores >= threshold)
    frr = np.mean(genuine_scores < threshold)
    return far, frr


def roc_curve(genuine_scores, impostor_scores):
    """Curva ROC (FAR vs TAR) barriendo todos los umbrales relevantes.

    Devuelve (far, tar, thresholds), donde tar = 1 - frr.
    """
    genuine_scores = np.asarray(genuine_scores)
    impostor_scores = np.asarray(impostor_scores)
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
    y_score = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, thresholds = _sk_roc_curve(y_true, y_score)
    far = fpr
    tar = tpr
    return far, tar, thresholds


def compute_eer(genuine_scores, impostor_scores):
    """Equal Error Rate: punto donde FAR y FRR se cruzan (interpolado).

    Devuelve (eer, umbral_en_eer).
    """
    far, tar, thresholds = roc_curve(genuine_scores, impostor_scores)
    frr = 1 - tar

    # El cruce esta donde far - frr cambia de signo.
    diff = far - frr
    idx = np.where(np.diff(np.sign(diff)))[0]
    if len(idx) == 0:
        # No hay cruce exacto: nos quedamos con el punto de menor diferencia.
        i = np.argmin(np.abs(diff))
        return float((far[i] + frr[i]) / 2), float(thresholds[i])

    i = idx[0]
    # Interpolacion lineal entre los puntos i e i+1.
    far1, far2 = far[i], far[i + 1]
    frr1, frr2 = frr[i], frr[i + 1]
    t1, t2 = thresholds[i], thresholds[i + 1]
    denom = (far2 - far1) - (frr2 - frr1)
    if denom == 0:
        alpha = 0.5
    else:
        alpha = (frr1 - far1) / denom
    alpha = np.clip(alpha, 0.0, 1.0)
    eer = far1 + alpha * (far2 - far1)
    threshold_at_eer = t1 + alpha * (t2 - t1)
    return float(eer), float(threshold_at_eer)


def attack_success_rate(scores_before, scores_after, threshold, mode):
    """Porcentaje de ataques que consiguen su objetivo.

    En modo "dodge", cuenta las veces que una muestra genuina aceptada
    pasa a ser rechazada tras el ataque. En modo "impersonate", cuenta las
    veces que una muestra impostora rechazada pasa a ser aceptada.
    """
    scores_before = np.asarray(scores_before)
    scores_after = np.asarray(scores_after)

    if mode == "dodge":
        eligible = scores_before >= threshold
        success = eligible & (scores_after < threshold)
    elif mode == "impersonate":
        eligible = scores_before < threshold
        success = eligible & (scores_after >= threshold)
    else:
        raise ValueError("mode debe ser 'dodge' o 'impersonate'")

    if eligible.sum() == 0:
        return 0.0
    return float(success.sum() / eligible.sum())


def adversarial_accuracy(scores_adv, threshold, is_genuine):
    """Porcentaje de muestras atacadas que el sistema sigue clasificando
    bien. is_genuine indica si las muestras son pares genuinos (deberian
    seguir aceptandose) o impostores (deberian seguir rechazandose)."""
    scores_adv = np.asarray(scores_adv)
    if is_genuine:
        correct = scores_adv >= threshold
    else:
        correct = scores_adv < threshold
    return float(np.mean(correct))
