# junior-robotics-ops-cv-nav2-challenge

# Bloque A — Computer Vision (detector 1 clase: `cone`)

Repositorio: `junior-robotics-ops-cv-nav2-challenge`

Este bloque cubre:
- Preparación de un dataset en formato YOLO (con script de preparación).
- Preprocesado / control de calidad del dataset: deduplicación y verificación de splits.
- Entrenamiento reproducible (configs + seeds + runs).
- Validación con métricas y explicación operativa.
- Demostración de data leakage y overfitting (provocados/detectados) y mitigaciones.

## 1. Entorno y dependencias

Entorno usado (referencia):
- Python 3.12.x
- PyTorch + CUDA (opcional, pero recomendado para entrenar rápido)
- YOLOv9 (third_party)

Activación de entorno:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Dataset seleccionado y objetivo

Objetivo: entrenar un detector de 1 sola clase (cono) a partir de un dataset Roboflow.

Dataset procesado final (YOLO, single-class):

- block_a_cv/data/processed/cone_wdg6r_yolo_dedup/

- images/{train,val,test}

- labels/{train,val,test}

- data.yaml

- prepare_report.json

Clase:

names: ['cone']

## 3. Preparación del dataset (formato YOLO + single-class)
### 3.1. Dataset de entrada (Roboflow export)

El dataset raw/processed inicial se maneja en:

block_a_cv/data/processed/cone_wdg6r_dedup/ (con splits train/valid/test)

Se hizo un split reproducible y se comprobó:

Train: 621

Val: 77

Test: 77

Comprobación rápida:

```bash
export DS_DEDUP="block_a_cv/data/processed/cone_wdg6r_dedup"
echo "TRAIN:" $(find "$DS_DEDUP/train/images" -type f | wc -l)
echo "VALID:" $(find "$DS_DEDUP/valid/images" -type f | wc -l)
echo "TEST :" $(find "$DS_DEDUP/test/images"  -type f | wc -l)
```

### 3.2. Detección de leakage por duplicados exactos (sha1)

Se verificó que no hay intersecciones exactas entre splits en el dataset final YOLO:

```bash
python - <<'PY'
from pathlib import Path
import hashlib

ds = Path("block_a_cv/data/processed/cone_wdg6r_yolo_dedup/images")

def sha1(p):
    h=hashlib.sha1()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()

def hashes(split):
    files = sorted((ds/split).glob("*"))
    return {sha1(p) for p in files}

h_train = hashes("train")
h_val   = hashes("val")
h_test  = hashes("test")

print("train∩val:", len(h_train & h_val))
print("train∩test:", len(h_train & h_test))
print("val∩test:", len(h_val & h_test))
PY
```

Resultado esperado (dataset deduplicado):

train∩val = 0

train∩test = 0

val∩test = 0

### 3.3. Conversión a YOLO single-class

Script usado:

block_a_cv/scripts/prepare_dataset.py

Ejemplo de ejecución:

```bash
export DS_DEDUP="block_a_cv/data/processed/cone_wdg6r_dedup"
export DS_YOLO="block_a_cv/data/processed/cone_wdg6r_yolo_dedup"

rm -rf "$DS_YOLO"

python block_a_cv/scripts/prepare_dataset.py \
  --raw "$DS_DEDUP" \
  --out "$DS_YOLO" \
  --remap_all_classes_to_zero \
  --num-classes 1 \
  --class-names "cone"
```
Artefacto de reporte:

block_a_cv/data/processed/cone_wdg6r_yolo_dedup/prepare_report.json

Incluye:

- split mode

- seed

- número total de imágenes

- duplicados exactos encontrados (si aplica)

- muestras problemáticas (si aplica)

## 4. Preprocesado: ¿aumentar o disminuir?
### 4.1. ¿Necesito aumentar (data augmentation)?

Para detección en operaciones reales suele ser útil aumentar si esperas variabilidad en:

- iluminación (hsv/contrast)

- blur / motion blur

- pequeñas rotaciones y traslaciones

- escalado y zoom

- oclusiones parciales

En este proyecto YOLOv9 ya aplica augmentación en entrenamiento mediante hyp.scratch-high.yaml (mosaic, mixup, copy-paste, flips, HSV, etc.).
Esto ayuda a generalización, especialmente si el dataset no cubre toda la variabilidad del entorno operativo.

### 4.2. ¿Necesito disminuir (dataset size / sampling)?

Disminuir puede ser útil para:

- demostrar overfitting intencionalmente (dataset muy pequeño en train)

- acelerar experimentos

- hacer debugging de pipeline

Se creó un dataset "tiny" para provocar overfitting:

block_a_cv/data/processed/cone_wdg6r_yolo_tiny/

ejemplo: train reducido a ~40 imágenes (val/test se mantienen)

## 5. Entrenamiento reproducible (configs + seed + runs)

Entrenamiento base (baseline) con YOLOv9:

- Script: block_a_cv/scripts/train_baseline.sh

- Seed: 42

- IMG: 512

- Batch: 4

- Epochs: 60 (aunque el run puede cortarse si se interrumpe)

Ejemplo:
```bash
export DATASET="block_a_cv/data/processed/cone_wdg6r_yolo_dedup/data.yaml"
bash -c 'DATASET="$DATASET" bash block_a_cv/scripts/train_baseline.sh'
```

Run principal (deduplicado):

block_a_cv/runs/baseline_yolov9s_seed42_img512_b4_e60_DEDUP/

Se guardan:

- opt.yaml (config completa del entrenamiento)

- hyp.yaml

- results.csv

- weights/best.pt, weights/last.pt

- imágenes de batches y labels

## 6. Validación y métricas (interpretación operativa)

Se valida el modelo entrenado usando val.py con --task test.

Ejemplo:
```bash
YOLO_DIR="block_a_cv/third_party/yolov9"
RUN="block_a_cv/runs/baseline_yolov9s_seed42_img512_b4_e60_DEDUP"
DATA="block_a_cv/data/processed/cone_wdg6r_yolo_dedup/data.yaml"

python "$YOLO_DIR/val.py" \
  --data "$DATA" \
  --img 512 \
  --weights "$RUN/weights/best.pt" \
  --task test
```
Ejemplo de salida (test):

- Precision (P): ~0.97

- Recall (R): ~0.915

- mAP@0.5: ~0.969

- mAP@0.5:0.95: ~0.726

### 6.1. Qué significa cada métrica y cuándo usarla

- Precision (P): de todas las detecciones que el modelo da, cuántas son correctas.
Útil cuando el coste de un falso positivo es alto (p. ej. parar el robot sin motivo).

- Recall (R): de todos los objetos reales, cuántos detecta el modelo.
Ùtil cuando el coste de un falso negativo es alto (p. ej. no detectar un obstáculo importante).

- mAP@0.5: promedio de precisión por clase a IoU=0.5.
Indicador rápido de calidad general del detector, comparaciones entre modelos/runs.

- mAP@0.5:0.95: promedio de mAP en múltiples IoUs (0.5..0.95).
Métrica más estricta: penaliza cajas mal ajustadas. Útil si importa la calidad geométrica de la caja (p. ej. estimación de distancia/posición a partir de bbox).

- Curvas PR / F1: permiten elegir umbral de confianza (trade-off P vs R).
Escoger un conf_thres que cumpla requisitos (seguridad vs estabilidad).

## 7. Data leakage: demostración + detección + mitigación
### 7.1. Qué es leakage en este contexto

Leakage típico en visión: que imágenes duplicadas o casi duplicadas aparezcan en train y val/test.
Resultado: métricas artificialmente altas (el modelo ve “lo mismo” en train y eval).

### 7.2. Cómo se detectó

Se usó hashing (sha1) para comprobar intersección de imágenes exactas entre splits (ver sección 3.2).
En el dataset deduplicado el resultado es 0 intersecciones.

### 7.3. Cómo se provocó (dataset leaky)

Se generó un dataset "LEAKY" copiando imágenes+labels de train a val:

block_a_cv/data/processed/cone_wdg6r_yolo_LEAKY/

Esto provoca que val sea “demasiado fácil” y suba artificialmente el rendimiento.

### 7.4. Mitigación

Split correcto y deduplicación (sha1) entre splits.

No mezclar fuentes “por archivo” sin control.

Con datasets de Roboflow: verificar duplicados tras export.

## 8. Overfitting: demostración + mitigación
### 8.1. Qué es overfitting en este contexto

Overfitting: el modelo se ajusta demasiado a train (memoriza) y no generaliza.
Síntomas:

- métricas/loses de train mejoran mucho

- val/test se estancan o empeoran

- gap grande entre train y val/test

### 8.2. Cómo se provocó

Se creó un dataset pequeño para entrenamiento (train muy reducido):

b- lock_a_cv/data/processed/cone_wdg6r_yolo_tiny/ (ej. ~40 imágenes en train)

Se entrenó:

nombre run ejemplo: block_a_cv/runs/overfit_tiny_seed42_img320_b2_e30...

### 8.3. Cómo se evidenció

Se comparan métricas de train/val/test y curvas.
En overfit, típicamente:

- train sube relativamente

- val/test quedan claramente peor que baseline deduplicado

### 8.4. Mitigaciones recomendadas

Aumentar dataset (más variabilidad real).

Augmentación (mosaic/mixup/HSV/blur/scale, ya incluido en hyp).

Regularización (weight decay, label smoothing moderado).

Early stopping (patience) y seleccionar best epoch.

Reducir capacidad del modelo (modelo más pequeño) si el dataset es pequeño.

## 9. Telemetría y reproducibilidad (runs, configs, logs)

Cada entrenamiento guarda en el run:

- opt.yaml: parámetros completos (incluye seed)

- hyp.yaml: hiperparámetros/augmentación

- results.csv: evolución por epoch

- pesos: weights/best.pt, weights/last.pt

Esto permite:

- repetir exactamente el entrenamiento (misma seed/config)

- comparar runs de manera objetiva

## 10. Parches de compatibilidad (PyTorch 2.6+ / YOLOv9)

Con versiones modernas de PyTorch, torch.load() puede fallar por el cambio de weights_only=True por defecto.
Se aplicaron parches mínimos (documentados) para permitir ejecutar train/val sin errores.

Se guarda un diff:

block_a_cv/yolov9_compat_patches.diff


## 11. Cómo reproducir el bloque A de principio a fin

Preparar dataset YOLO single-class:

```bash
python block_a_cv/scripts/prepare_dataset.py \
  --raw block_a_cv/data/processed/cone_wdg6r_dedup \
  --out block_a_cv/data/processed/cone_wdg6r_yolo_dedup \
  --remap_all_classes_to_zero \
  --num-classes 1 \
  --class-names "cone"
```

Entrenar baseline:

```bash
export DATASET="block_a_cv/data/processed/cone_wdg6r_yolo_dedup/data.yaml"
bash -c 'DATASET="$DATASET" bash block_a_cv/scripts/train_baseline.sh'

Validar en test:

python block_a_cv/third_party/yolov9/val.py \
  --data block_a_cv/data/processed/cone_wdg6r_yolo_dedup/data.yaml \
  --img 512 \
  --weights block_a_cv/runs/<RUN_NAME>/weights/best.pt \
  --task test
```

Demostraciones:

Leakage: usar cone_wdg6r_yolo_LEAKY

Overfit: usar cone_wdg6r_yolo_tiny

## 12. Estructura relevante

block_a_cv/scripts/

- prepare_dataset.py

- dedupe_val_against_train.py

- dedupe_test_against_train.py

- train_baseline.sh

block_a_cv/data/processed/

- cone_wdg6r_yolo_dedup/

- cone_wdg6r_yolo_LEAKY/ (demo leakage)

- cone_wdg6r_yolo_tiny/ (demo overfitting)

block_a_cv/runs/

- runs de entrenamiento/validación

- block_a_cv/yolov9_compat_patches.diff