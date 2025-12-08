# Ejemplo de Uso - DeepLabScan

Este documento muestra ejemplos de uso del proyecto DeepLabScan.

## 1. Configuración Inicial

### Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Configurar API de Roboflow

Crea un archivo `.env` en la raíz del proyecto:

```bash
ROBOFLOW_API_KEY=tu_api_key_aqui
```

O edita `configs/config.yaml` para incluir tus credenciales de Roboflow.

## 2. Descargar Datos

### Desde Línea de Comandos

```bash
python scripts/download_data.py \
    --workspace "mi_workspace" \
    --project "mi_proyecto" \
    --version 1 \
    --location "./data/raw"
```

### Usando Configuración YAML

Edita `configs/config.yaml`:

```yaml
roboflow:
  workspace: 'mi_workspace'
  project: 'mi_proyecto'
  version: 1
```

Luego ejecuta:

```bash
python scripts/download_data.py --config configs/config.yaml
```

### Desde Python

```python
from src.data import RoboflowDataLoader

loader = RoboflowDataLoader(
    workspace="mi_workspace",
    project="mi_proyecto",
    version=1,
    api_key="tu_api_key"
)

dataset_path = loader.download_dataset(location="./data/raw")
print(f"Dataset descargado en: {dataset_path}")
```

## 3. Entrenar Modelo

### Entrenamiento Básico

```bash
python scripts/train.py \
    --data data/raw/data.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device cpu
```

### Con GPU (CUDA)

```bash
python scripts/train.py \
    --data data/raw/data.yaml \
    --epochs 100 \
    --batch 32 \
    --imgsz 640 \
    --device cuda
```

### Con Diferentes Modelos

```bash
# Modelo pequeño (más rápido)
python scripts/train.py --model yolov8n.pt --data data.yaml

# Modelo mediano (balance)
python scripts/train.py --model yolov8m.pt --data data.yaml

# Modelo grande (más preciso)
python scripts/train.py --model yolov8l.pt --data data.yaml
```

### Con Aumentación Personalizada

```bash
# Aumentación ligera
python scripts/train.py --data data.yaml --augmentation light

# Aumentación intensiva
python scripts/train.py --data data.yaml --augmentation heavy
```

### Desde Python

```python
from src.models import YOLOModel, YOLOTrainer
from src.data import DataAugmentation

# Crear modelo
model = YOLOModel(model_name='yolov8n.pt', pretrained=True)

# Configurar trainer
trainer = YOLOTrainer(model.get_model(), 'data/raw/data.yaml')

# Entrenar
results = trainer.train(
    epochs=100,
    imgsz=640,
    batch=16,
    device='cpu',
    patience=50,
    augmentation_params=DataAugmentation.get_default_augmentation()
)

print("Entrenamiento completado!")
```

## 4. Evaluar Modelo

### Evaluación Básica

```bash
python scripts/evaluate.py \
    --weights runs/train/exp/weights/best.pt \
    --data data/raw/data.yaml
```

### Con Visualizaciones

```bash
python scripts/evaluate.py \
    --weights runs/train/exp/weights/best.pt \
    --data data/raw/data.yaml \
    --save-plots \
    --output-dir results/evaluation
```

### Evaluar en Test Set

```bash
python scripts/evaluate.py \
    --weights runs/train/exp/weights/best.pt \
    --split test \
    --save-plots
```

### Desde Python

```python
from ultralytics import YOLO
from src.evaluation import MetricsCalculator
from src.utils import ResultsVisualizer

# Cargar modelo
model = YOLO('runs/train/exp/weights/best.pt')

# Evaluar
calculator = MetricsCalculator(model, 'data/raw/data.yaml')
metrics = calculator.evaluate(split='val', device='cpu')

# Generar reporte
report = calculator.generate_report(save_path='results/report.txt')
print(report)

# Visualizar
visualizer = ResultsVisualizer()
visualizer.create_results_summary(
    metrics,
    save_path='results/summary.png'
)
```

## 5. Realizar Predicciones

### En Imagen Individual

```bash
python scripts/predict.py \
    --weights runs/train/exp/weights/best.pt \
    --source path/to/image.jpg \
    --conf 0.25
```

### En Directorio de Imágenes

```bash
python scripts/predict.py \
    --weights runs/train/exp/weights/best.pt \
    --source path/to/images/ \
    --save \
    --save-txt
```

### En Video

```bash
python scripts/predict.py \
    --weights runs/train/exp/weights/best.pt \
    --source path/to/video.mp4 \
    --save
```

### Con Webcam

```bash
python scripts/predict.py \
    --weights runs/train/exp/weights/best.pt \
    --source 0 \
    --show
```

### Con Configuración Personalizada

```bash
python scripts/predict.py \
    --weights best.pt \
    --source images/ \
    --conf 0.5 \
    --iou 0.7 \
    --imgsz 1280 \
    --device cuda \
    --save \
    --save-txt \
    --save-crop
```

### Desde Python

```python
from ultralytics import YOLO

# Cargar modelo
model = YOLO('runs/train/exp/weights/best.pt')

# Predicción en imagen
results = model.predict(
    source='path/to/image.jpg',
    conf=0.25,
    save=True
)

# Predicción en múltiples imágenes
results = model.predict(
    source='path/to/images/',
    conf=0.25,
    save=True,
    project='runs/predict',
    name='exp'
)

# Mostrar resultados
for result in results:
    boxes = result.boxes
    print(f"Detecciones: {len(boxes)}")
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"  Clase: {result.names[cls]}, Confianza: {conf:.2f}")
```

## 6. Ejemplo Completo de Workflow

```bash
# 1. Descargar datos
python scripts/download_data.py \
    --workspace "mi_workspace" \
    --project "deteccion_objetos" \
    --version 1

# 2. Entrenar modelo
python scripts/train.py \
    --data data/raw/data.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --batch 16 \
    --device cpu \
    --augmentation default

# 3. Evaluar modelo
python scripts/evaluate.py \
    --weights runs/train/exp/weights/best.pt \
    --data data/raw/data.yaml \
    --save-plots \
    --output-dir results/evaluation

# 4. Hacer predicciones
python scripts/predict.py \
    --weights runs/train/exp/weights/best.pt \
    --source test_images/ \
    --conf 0.25 \
    --save \
    --save-txt
```

## 7. Consejos y Mejores Prácticas

### Aumentación de Datos

- **Light**: Para datasets grandes (>1000 imágenes)
- **Default**: Para datasets medianos (500-1000 imágenes)
- **Heavy**: Para datasets pequeños (<500 imágenes)

### Selección de Modelo

- **yolov8n**: Para dispositivos con recursos limitados
- **yolov8s**: Balance entre velocidad y precisión
- **yolov8m**: Para aplicaciones generales
- **yolov8l/x**: Cuando la precisión es crítica

### Hiperparámetros

```yaml
# Para objetos pequeños
imgsz: 1280
batch: 8

# Para entrenamiento rápido
imgsz: 416
batch: 32
epochs: 50

# Para máxima precisión
imgsz: 1280
batch: 4
epochs: 300
patience: 100
```

### Device Selection

```bash
# CPU
--device cpu

# GPU única
--device cuda
# o
--device 0

# GPU específica (múltiples GPUs)
--device 1

# Múltiples GPUs
--device 0,1,2,3

# Apple Silicon (Mac M1/M2)
--device mps
```

## 8. Solución de Problemas

### Memoria Insuficiente

Reduce el tamaño del batch:
```bash
python scripts/train.py --batch 8 --imgsz 416
```

### mAP Bajo

1. Aumenta épocas de entrenamiento
2. Usa un modelo más grande
3. Ajusta la aumentación de datos
4. Verifica la calidad del etiquetado

### Muchos Falsos Positivos

Aumenta el confidence threshold:
```bash
python scripts/predict.py --conf 0.5
```

### Detecciones Perdidas

Disminuye el confidence threshold:
```bash
python scripts/predict.py --conf 0.15
```
