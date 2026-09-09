"""
00 - Validacion unitaria del entorno de medicion

Comprueba que el codigo de metricas y de ataques funciona bien antes de
usarlo sobre los modelos reales. Para eso usa un caso sencillo e inventado
donde ya se sabe de antemano cual es el resultado correcto, en vez de usar
FaceNet o ECAPA-TDNN. No necesita ningun modelo ni dataset descargado.

Salida:
  - Imprime por consola si cada comprobacion pasa o falla.
  - results/tables/validacion_unitaria.csv con el resumen.
"""

import os
import csv
import torch

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))

from common.metrics import far_frr_at_threshold, compute_eer
from common.attacks import fgsm_attack, pgd_attack

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

resultados = []


def check(nombre, condicion, detalle):
    estado = "OK" if condicion else "FALLO"
    print(f"[{estado}] {nombre}: {detalle}")
    resultados.append({"criterio": nombre, "resultado": estado, "detalle": detalle})
    return condicion


# Criterio 1: metricas (FAR, FRR, EER) sobre puntuaciones inventadas.
# Estan elegidas para que, en el umbral 0.55, tanto el FAR como el FRR
# valgan exactamente 0.2, un resultado facil de comprobar a mano.
genuine_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
impostor_scores = [0.6, 0.5, 0.4, 0.3, 0.2]

far, frr = far_frr_at_threshold(genuine_scores, impostor_scores, threshold=0.55)
check(
    "Modulo de metricas: FAR/FRR en el umbral 0.55",
    abs(far - 0.2) < 1e-9 and abs(frr - 0.2) < 1e-9,
    f"FAR={far:.4f} (esperado 0.2000), FRR={frr:.4f} (esperado 0.2000)",
)

eer, threshold_eer = compute_eer(genuine_scores, impostor_scores)
check(
    "Modulo de metricas: punto de EER",
    abs(eer - 0.2) < 0.02,
    f"EER={eer:.4f} (esperado approx. 0.2000), umbral={threshold_eer:.4f}",
)

# Criterio 2: los ataques mueven la similitud en la direccion correcta.
# Se usa un modelo de prueba muy simple, que devuelve la entrada tal
# cual, para poder predecir con seguridad como deberia comportarse el
# ataque sin depender de una red real.


class ModeloIdentidad(torch.nn.Module):
    def forward(self, x):
        return x


modelo_sintetico = ModeloIdentidad()
torch.manual_seed(0)
x0 = torch.randn(1, 16)
template = torch.randn(1, 16)
cos = torch.nn.functional.cosine_similarity

sim_original = cos(x0, template).item()
epsilons = [0.01, 0.05, 0.1, 0.2]

sims_dodge = []
for eps in epsilons:
    x_adv = fgsm_attack(modelo_sintetico, x0, template, epsilon=eps, mode="dodge")
    sims_dodge.append(cos(x_adv, template).item())

monotono_decreciente = all(sims_dodge[i] > sims_dodge[i + 1] for i in range(len(sims_dodge) - 1))
check(
    "Modulo de ataques: evasion reduce la similitud de forma monotona con epsilon",
    monotono_decreciente,
    f"similitud original={sim_original:.4f}, similitudes tras el ataque={['%.4f' % s for s in sims_dodge]}",
)

sims_impersonate = []
for eps in epsilons:
    x_adv = fgsm_attack(modelo_sintetico, x0, template, epsilon=eps, mode="impersonate")
    sims_impersonate.append(cos(x_adv, template).item())

monotono_creciente = all(sims_impersonate[i] < sims_impersonate[i + 1] for i in range(len(sims_impersonate) - 1))
check(
    "Modulo de ataques: suplantacion aumenta la similitud de forma monotona con epsilon",
    monotono_creciente,
    f"similitud original={sim_original:.4f}, similitudes tras el ataque={['%.4f' % s for s in sims_impersonate]}",
)

# Criterio 3: con el mismo presupuesto de perturbacion, PGD debe mover la
# similitud tanto o mas que FGSM, ya que aplica el mismo ataque en varios
# pasos en lugar de uno solo.
eps_comparacion = 0.1
x_fgsm = fgsm_attack(modelo_sintetico, x0, template, epsilon=eps_comparacion, mode="dodge")
x_pgd = pgd_attack(modelo_sintetico, x0, template, epsilon=eps_comparacion, alpha=eps_comparacion / 4,
                    steps=10, mode="dodge", random_start=False)

sim_fgsm = cos(x_fgsm, template).item()
sim_pgd = cos(x_pgd, template).item()

check(
    "Modulo de ataques: PGD desplaza la similitud tanto o mas que FGSM (mismo epsilon)",
    sim_pgd <= sim_fgsm + 1e-6,
    f"similitud tras FGSM={sim_fgsm:.4f}, similitud tras PGD (10 pasos)={sim_pgd:.4f}",
)

# Guardar el resumen en un CSV.
out_path = os.path.join(TABLES_DIR, "validacion_unitaria.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["criterio", "resultado", "detalle"])
    writer.writeheader()
    writer.writerows(resultados)

n_ok = sum(1 for r in resultados if r["resultado"] == "OK")
print(f"\n{n_ok}/{len(resultados)} criterios superados. Resumen guardado en {out_path}")
