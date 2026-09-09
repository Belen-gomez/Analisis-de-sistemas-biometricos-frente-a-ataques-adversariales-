"""
Ataques FGSM y PGD contra los sistemas de verificacion facial y de voz.

Estan implementados a mano, sin usar ninguna libreria de ataques, para que
el codigo se ajuste exactamente a lo explicado en la memoria.

Hay dos modos de ataque:
  - "dodge" (evasion): parte de una muestra genuina y la modifica para que
    el sistema rechace a un usuario legitimo.
  - "impersonate" (suplantacion): parte de una muestra de un atacante y la
    modifica para que el sistema la acepte como si fuera otra persona.
"""

import torch
import torch.nn.functional as F


def _objective(embedding, template, mode):
    similarity = F.cosine_similarity(embedding, template)
    if mode == "dodge":
        return -similarity.mean()
    elif mode == "impersonate":
        return similarity.mean()
    else:
        raise ValueError("mode debe ser 'dodge' o 'impersonate'")


def fgsm_attack(model, x, template, epsilon, mode, clip_min=None, clip_max=None):
    """Ataque FGSM: modifica x en un unico paso para acercarla o alejarla
    de la plantilla, segun el modo. epsilon limita cuanto se puede cambiar
    cada valor de x como maximo."""
    x = x.clone().detach().requires_grad_(True)
    embedding = model(x)
    loss = _objective(embedding, template, mode)
    loss.backward()

    x_adv = x + epsilon * x.grad.sign()
    if clip_min is not None or clip_max is not None:
        x_adv = x_adv.clamp(min=clip_min, max=clip_max)
    return x_adv.detach()


def pgd_attack(model, x, template, epsilon, alpha, steps, mode,
               clip_min=None, clip_max=None, random_start=True):
    """Ataque PGD: aplica la misma idea que FGSM pero en varios pasos
    pequeños, corrigiendo la direccion en cada uno. epsilon sigue siendo
    el limite maximo total de cambio permitido."""
    x_orig = x.clone().detach()

    if random_start:
        delta = torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(x_orig)

    x_adv = x_orig + delta
    if clip_min is not None or clip_max is not None:
        x_adv = x_adv.clamp(min=clip_min, max=clip_max)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        embedding = model(x_adv)
        loss = _objective(embedding, template, mode)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            # No dejar que el cambio acumulado supere epsilon.
            perturbation = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
            x_adv = x_orig + perturbation
            if clip_min is not None or clip_max is not None:
                x_adv = x_adv.clamp(min=clip_min, max=clip_max)

    return x_adv.detach()
