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


# Bloque B - Nodo ROS2 Detecciones


Objetivo: entrenar y desplegar un detector de 1 clase (cono) y demostrar prácticas de MLOps/Robotics Ops:
- overfitting y mitigación
- data leakage (detección/mitigación)
- métricas e interpretación
- telemetría y reproducibilidad (runs/configs/logs)
- despliegue ROS2 (suscripción cámara, detecciones, telemetría, rosbag)

Este repo contiene:
- **Block A (Computer Vision)**: preparación de dataset, deduplicación, escenarios de leakage/overfit, entrenamiento reproducible y métricas.
- **Block B (ROS2 Ops/telemetría/debug)**: nodo ROS2 en C++ modular, inferencia ONNX Runtime, publicación de detecciones + telemetría, guardado JSONL y scripts para grabar/reproducir rosbag.

---

## Estructura del repositorio (alto nivel)

- `block_a_cv/`
  - `scripts/`: preparación de datasets, dedupe, escenarios de leakage/overfit, entrenamiento, resumen de runs
  - `data/`: datasets (NO se suben a git)
  - `runs/`: salidas de entrenamiento (NO se suben a git)
  - `third_party/yolov9/`: YOLOv9 (submódulo/copia vendorizada)
  - `yolov9_compat_patches.diff`: parches mínimos para compatibilidad con PyTorch moderno

- `block_b_ros2_ws/`
  - `src/cv_yolov9_detector_cpp/`: paquete ROS2 C++ (nodo + librerías) y scripts de rosbag
  - `build/ install/ log/`: artefactos de compilación (NO se suben a git)
  - `models/`: modelos ONNX (recomendado NO subir a git; se generan desde Block A)

---

## Requisitos

### Sistema
- Ubuntu 24.04 (noble) (o equivalente)
- ROS2 Jazzy
- CMake / colcon / rosdep

### Dependencias ROS2 (Bloque B)
Instalar dependencias típicas:
```bash
sudo apt-get update
sudo apt-get install -y \
  ros-$ROS_DISTRO-vision-msgs \
  ros-$ROS_DISTRO-cv-bridge \
  ros-$ROS_DISTRO-image-transport \
  nlohmann-json3-dev
```

ONNX Runtime:

- Si libonnxruntime-dev no está disponible en apt para tu distro, se instala desde release binario (ver sección “ONNX Runtime (instalación real)”).

B.1 Qué entrega este bloque

Paquete ROS2 en C++ que:

- Suscribe a un topic de imagen (sensor_msgs/msg/Image)

- Corre inferencia con ONNX Runtime (modelo YOLOv9 exportado a ONNX)

Publica:

- vision_msgs/Detection2DArray en /detections (configurable)

- Telemetría en /telemetry/cv como std_msgs/String (JSON serializado)

- Guarda telemetría en disco como JSONL con timestamps, latencia, FPS y conteo de detecciones

- Incluye scripts para grabar y reproducir un rosbag (mínimo cámara + detecciones)

## B.2 Workspace y build
```bash
cd ~/junior-robotics-ops-cv-nav2-challenge/block_b_ros2_ws
source /opt/ros/$ROS_DISTRO/setup.bash

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install

source install/setup.bash

Nota: evita mezclar el venv de Python del Bloque A con ROS2 (puede romper ros2 tools). Si tienes dudas:

deactivate 2>/dev/null || true
unset PYTHONPATH
unset PYTHONHOME
source /opt/ros/$ROS_DISTRO/setup.bash
```

## B.3 ONNX Runtime (instalación real)

Si no existe libonnxruntime-dev en apt para tu distro, se usa release binario.

Ejemplo (ajustar versión si fuera necesario):

```bash
cd /tmp
wget -O onnxruntime.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-1.24.2.tgz
tar -xzf onnxruntime.tgz

sudo rm -rf /opt/onnxruntime-1.24.2
sudo mv onnxruntime-linux-x64-1.24.2 /opt/onnxruntime-1.24.2

# (opcional) mantener /opt/onnxruntime apuntando a la versión actual
sudo ln -sfn /opt/onnxruntime-1.24.2 /opt/onnxruntime

# Registrar librerías en el linker
echo "/opt/onnxruntime/lib" | sudo tee /etc/ld.so.conf.d/onnxruntime.conf >/dev/null
sudo ldconfig

# pkg-config (para que CMake lo encuentre)
sudo mkdir -p /usr/local/lib/pkgconfig
sudo tee /usr/local/lib/pkgconfig/onnxruntime.pc >/dev/null <<'EOF'
prefix=/opt/onnxruntime
exec_prefix=${prefix}
libdir=${exec_prefix}/lib
includedir=${prefix}/include

Name: onnxruntime
Description: ONNX Runtime
Version: 1.24.2
Libs: -L${libdir} -lonnxruntime
Cflags: -I${includedir}
EOF

pkg-config --modversion onnxruntime
pkg-config --cflags --libs onnxruntime
```

## B.4 Export de modelo a ONNX (desde Block A)

Se exporta desde un best.pt entrenado (ejemplo: baseline DEDUP).

```bash
cd ~/junior-robotics-ops-cv-nav2-challenge
source .venv/bin/activate

pip install pandas onnx onnxsim onnxscript

export YOLO_DIR="block_a_cv/third_party/yolov9"
export W="block_a_cv/runs/baseline_yolov9s_seed42_img512_b4_e60_DEDUP/weights/best.pt"
export IMG=512

python "$YOLO_DIR/export.py" \
  --weights "$W" \
  --imgsz $IMG \
  --batch 1 \
  --device cpu \
  --include onnx \
  --opset 18 \
  --simplify
```

Resultado típico:

block_a_cv/runs/.../weights/best.onnx

Copiarlo al Bloque B:

```bash
mkdir -p block_b_ros2_ws/models
cp -v block_a_cv/runs/baseline_yolov9s_seed42_img512_b4_e60_DEDUP/weights/best.onnx \
  block_b_ros2_ws/models/yolov9_cone.onnx
```

## B.5 Parámetros del nodo

Parámetros principales (ejemplo):

- image_topic (default: /camera/image_raw o el que uses)

- detections_topic (default: /detections)

- telemetry_topic (default: /telemetry/cv)

- telemetry_path (default recomendado: ~/.ros/cv_telemetry.jsonl)

- model_path (ruta al ONNX)

- imgsz (debe coincidir con el export, ej: 512)

- conf_thres, iou_thres, max_det

- class_name (ej: cone)

## B.6 Pruebas reales (webcam) — exactamente qué correr en cada terminal

### Terminal 1: cámara (webcam)

Puedes usar v4l2_camera (recomendado en Jazzy). Si no lo tienes:
```bash
sudo apt-get install -y ros-$ROS_DISTRO-v4l2-camera
```

Arranque ejemplo:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
ros2 run v4l2_camera v4l2_camera_node --ros-args \
  -p image_size:="[640,480]" \
  -p pixel_format:="RGB3" \
  -p output_encoding:="rgb8" \
  -r image_raw:=/image_raw
```

Verifica que publica:

```bash
ros2 topic info /image_raw
Terminal 2: nodo detector (con ONNX)
cd ~/junior-robotics-ops-cv-nav2-challenge/block_b_ros2_ws
deactivate 2>/dev/null || true
unset PYTHONPATH
unset PYTHONHOME

source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash

rm -f /tmp/cv_telemetry.jsonl

ros2 run cv_yolov9_detector_cpp yolov9_detector_node --ros-args \
  -p image_topic:=/image_raw \
  -p telemetry_path:=/tmp/cv_telemetry.jsonl \
  -p model_path:="$HOME/junior-robotics-ops-cv-nav2-challenge/block_b_ros2_ws/models/yolov9_cone.onnx" \
  -p imgsz:=512 \
  -p conf_thres:=0.25 \
  -p iou_thres:=0.7 \
  --log-level info
```

### Terminal 3: verificación (topics + telemetría + detecciones + JSONL)

```bash
source /opt/ros/$ROS_DISTRO/setup.bash

# Rates (Ctrl+C para parar cada uno)
ros2 topic hz /image_raw
ros2 topic hz /telemetry/cv
ros2 topic hz /detections

# 1 muestra
ros2 topic echo /telemetry/cv --once
ros2 topic echo /detections --once

# JSONL crece y es válido
ls -la /tmp/cv_telemetry.jsonl
tail -n 3 /tmp/cv_telemetry.jsonl

python3 - <<'PY'
import json
p="/tmp/cv_telemetry.jsonl"
n=0
for line in open(p,'r',encoding='utf-8'):
    json.loads(line); n+=1
print("OK JSONL. lines:", n)
PY
```

Interpretación mínima:

/image_raw debería ir ~30 Hz (webcam)

/telemetry/cv y /detections bajarán según latencia de inferencia (ej: ~1–3 Hz si CPU y modelo pesado)

detections debe contener bounding boxes cuando el cono aparezca en cámara

/tmp/cv_telemetry.jsonl debe crecer y cada línea debe ser JSON válido

### B.7 Rosbag: grabar y reproducir (mínimo cámara + detecciones)

El paquete incluye scripts:

Grabar

En una terminal (con cámara + detector ya corriendo):

```bash
cd ~/junior-robotics-ops-cv-nav2-challenge/block_b_ros2_ws
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash

# Graba image + detections + telemetry
./install/cv_yolov9_detector_cpp/lib/cv_yolov9_detector_cpp/record_bag.sh \
  ~/bags/cv_bag_webcam \
  /image_raw /detections /telemetry/cv
Reproducir
cd ~/junior-robotics-ops-cv-nav2-challenge/block_b_ros2_ws
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash

./install/cv_yolov9_detector_cpp/lib/cv_yolov9_detector_cpp/play_bag.sh \
  ~/bags/cv_bag_webcam \
  1.0
```

Verificación durante el play (otra terminal):

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
ros2 topic hz /image_raw
ros2 topic hz /detections
ros2 topic echo /detections --once
```

### B.8 Telemetría (JSON) — campos mínimos

Cada mensaje de /telemetry/cv y cada línea en JSONL incluye:

- stamp_ns: timestamp monotónico

- image_topic: topic consumido

- img_w, img_h: tamaño del frame

- model_path, device, imgsz, conf_thres, iou_thres, max_det

- latency_ms: latencia end-to-end por frame

- fps: fps estimado (1/delta tiempo entre callbacks)

- num_detections: conteo detecciones

- mean_conf: confianza media del frame

### Notas de entrega / reproducibilidad

No se suben a git:

- datasets (block_a_cv/data/*)

- runs de entrenamiento (block_a_cv/runs/*)

- artefactos de build ROS2 (block_b_ros2_ws/build install log)

- rosbag (bags/*)

- telemetría generada (*.jsonl)

- modelos binarios (*.pt, *.onnx) (recomendado)

### Para reproducir:

- Block A: scripts + runs configurables, seeds en opt.yaml

- Block B: colcon build + parámetros del nodo + scripts rosbag