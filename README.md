# Código práctico del TFM

Este directorio contiene el entorno de experimentación desarrollado para la
parte práctica del TFM "Análisis de sistemas biométricos y evaluación de su
robustez frente a ataques adversariales". Implementa y evalúa dos sistemas
de verificación biométrica (reconocimiento facial y por voz) frente a
ataques adversariales (FGSM y PGD), incluyendo dos mecanismos de mitigación.

## Requisitos e instalación

Se necesita Python 3.9 o superior. Todo el código se ha ejecutado en local,
sobre CPU, sin necesidad de GPU.

Antes de ejecutar cualquier script, hay que instalar las dependencias:

```
cd practica
pip install -r requirements.txt
```

Los conjuntos de datos (LFW y LibriSpeech test-clean) y los modelos
preentrenados (FaceNet, ECAPA-TDNN) se descargan automáticamente la primera
vez que se ejecuta un script que los necesita, no hace falta descargarlos
a mano. La primera ejecución tardará más por esta descarga.

## Estructura

```
practica/
  src/
    common/     metricas de verificacion (FAR, FRR, EER, ASR) y ataques
                adversariales (FGSM, PGD), compartidos por las dos
                modalidades
    face/       pipeline de reconocimiento facial (MTCNN + FaceNet)
    voice/      pipeline de reconocimiento por voz (ECAPA-TDNN)
  scripts/      scripts numerados que ejecutan cada prueba (ver mas abajo)
  results/      tablas (CSV), figuras (PNG) y modelo ajustado, ya generados
                a partir de una ejecucion previa completa. Son los mismos
                resultados que se citan en la memoria.
```

## Resultados ya incluidos

La carpeta `results/` ya contiene todas las tablas y figuras generadas por
estos scripts en una ejecución completa previa, las mismas que aparecen en
la memoria. **No es necesario volver a ejecutar los scripts para consultar
los resultados.** Volver a ejecutarlos sirve para comprobar que el código
es reproducible y para verificar que los cálculos son correctos, pero por
el tiempo que requieren algunos de ellos, no es
imprescindible para revisar el trabajo.

## Cómo ejecutar los scripts

Los scripts están numerados en el orden en que se ejecutaron. Cada uno se
lanza desde la carpeta `practica/scripts/`, por ejemplo:

```
cd practica/scripts
python 01_facial_baseline.py
```

| Script | Qué hace | Tiempo aproximado |
|---|---|---|
| `00_validacion_unitaria.py` | Valida que los módulos de métricas y ataques funcionan bien, sobre un modelo sintético | segundos |
| `01_facial_baseline.py` | Rendimiento base del sistema facial (sin ataque) | ~4 min |
| `02_facial_adversarial.py` | Barrido de ataques FGSM/PGD sobre el sistema facial | ~25 min |
| `03_voice_baseline.py` | Rendimiento base del sistema de voz (sin ataque) | ~12 min |
| `04_voice_adversarial.py` | Barrido de ataques FGSM/PGD sobre el sistema de voz, a escala completa | **~3 horas** |
| `05_comparativa_resultados.py` | Compara los resultados de las dos modalidades (solo lee los CSV ya generados por 01-04) | segundos |
| `06_comparativa_tiempos.py` | Compara el coste computacional entre modalidades | segundos |
| `07_defensa_preprocesado.py` | Evalúa la defensa por preprocesado (desenfoque / filtro paso bajo) | ~30-40 min |
| `08_defensa_finetuning_facial.py` | Reentrena la última capa del modelo facial y evalúa el efecto | ~30 min |

