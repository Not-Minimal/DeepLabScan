# DeepLabScan

Proyecto de detección de objetos usando YOLO con datos propios etiquetados en Roboflow.

## Descripción

Este proyecto implementa un pipeline completo de detección de objetos utilizando:
- Datos propios etiquetados en Roboflow
- Aumentación de datos para balanceo de clases
- Entrenamiento con YOLO (Ultralytics)
- Evaluación con métricas estándar (mAP, Precision, Recall)

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/Not-Minimal/DeepLabScan.git
cd DeepLabScan
```

### 2. Crear entorno virtual con Python 3.11

```bash
python3.11 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Actualizar pip y herramientas

```bash
pip install --upgrade pip setuptools wheel
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Uso

### 1. Descargar Dataset desde Roboflow

```bash
python scripts/download_roboflow_simple.py
```

Este script:
- Descarga el dataset desde Roboflow (URL configurada en el script)
- Descomprime automáticamente los archivos
- Organiza el dataset en `data/raw/` con la estructura YOLO esperada
- Mueve `data.yaml`, `train/`, `valid/`, `test/` a la ubicación correcta

**Nota:** Si necesitas cambiar la URL del dataset, edita la variable `DEFAULT_URL` en el script o pasa `--url` como argumento.

### 2. (Opcional) Aumentación de Datos

Balancea las clases minoritarias generando nuevas imágenes con aumentación (flip, rotación, brillo):

```bash
python scripts/augment_data.py --data-dir data/raw --limit 0.25
```

Parámetros:
- `--data-dir`: Directorio del dataset (default: `data/raw`)
- `--limit`: Límite de nuevas instancias como fracción del total (default: 0.25 = 25%)
- `--seed`: Semilla aleatoria para reproducibilidad (default: 42)
- `--save-plots`: Guardar gráficos en disco en lugar de mostrarlos

Este script:
- Analiza la distribución de clases
- Genera gráfico de distribución inicial
- Aumenta imágenes de clases minoritarias hasta igualar la mayoritaria
- Genera gráfico de distribución final
- Guarda las imágenes/etiquetas aumentadas en `train/images` y `train/labels`

### 3. Entrenar el Modelo

```bash
python scripts/train.py --data-dir data/raw --epochs 15
```

Parámetros disponibles:
- `--data-dir`: Directorio del dataset (default: `data/raw`)
- `--model`: Modelo YOLO a usar (default: `yolo11n.pt`)
- `--epochs`: Número de épocas (default: 15)
- `--batch`: Tamaño del batch, -1 para auto (default: -1)
- `--imgsz`: Tamaño de imágenes (default: 640)
- `--device`: Dispositivo (cpu/cuda/mps, auto-detect si no se especifica)
- `--project`: Directorio de salida (default: `runs/detect`)
- `--name`: Nombre del experimento (default: `train`)
- `--patience`: Épocas de paciencia para early stopping (default: 50)

El script:
- Detecta automáticamente GPU/CPU disponible
- Carga el modelo YOLO preentrenado
- Entrena con el dataset (aumentado si ejecutaste el paso 2)
- Guarda los pesos del mejor modelo en `runs/detect/train/weights/best.pt`

### 4. Evaluar el Modelo

```bash
python scripts/evaluate.py --weights runs/detect/train/weights/best.pt --data-dir data/raw
```

Genera:
- Métricas de Precision, Recall, mAP@0.5, mAP@0.5:0.95
- Matriz de confusión
- Curvas PR (Precision-Recall)
- Visualizaciones de predicciones

### 5. Realizar Predicciones

```bash
python scripts/predict.py --weights runs/detect/train/weights/best.pt --source path/to/images
```

## Estructura del Proyecto

```
DeepLabScan/
├── scripts/
│   ├── download_roboflow_simple.py   # Descarga dataset desde Roboflow
│   ├── augment_data.py               # Aumentación y balanceo de datos
│   ├── train.py                      # Entrenamiento del modelo
│   ├── evaluate.py                   # Evaluación del modelo
│   └── predict.py                    # Predicciones con el modelo
├── data/
│   └── raw/                          # Dataset descargado
│       ├── data.yaml                 # Configuración del dataset
│       ├── train/                    # Datos de entrenamiento
│       ├── valid/                    # Datos de validación
│       └── test/                     # Datos de prueba
├── runs/
│   └── detect/                       # Resultados de entrenamientos
├── requirements.txt                   # Dependencias del proyecto
├── .env                              # Variables de entorno (Roboflow API Key)
└── README.md
```

## Flujo de Trabajo Completo

```bash
# 1. Setup del entorno
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 2. Descargar datos
python scripts/download_roboflow_simple.py

# 3. (Opcional) Aumentar datos para balanceo
python scripts/augment_data.py --data-dir data/raw --limit 0.25

# 4. Entrenar modelo
python scripts/train.py --data-dir data/raw --epochs 15

# 5. Evaluar modelo
python scripts/evaluate.py --weights runs/detect/train/weights/best.pt --data-dir data/raw

# 6. Hacer predicciones
python scripts/predict.py --weights runs/detect/train/weights/best.pt --source test_image.jpg
```

## Métricas de Evaluación

- **Precision**: Proporción de detecciones correctas sobre todas las detecciones realizadas
- **Recall**: Proporción de objetos detectados sobre todos los objetos reales
- **mAP@0.5**: Mean Average Precision con IoU threshold de 0.5
- **mAP@0.5:0.95**: Mean Average Precision promediando IoU thresholds de 0.5 a 0.95

## Requisitos del Sistema

- Python 3.11+
- GPU recomendada (CUDA o Apple Silicon MPS) para entrenamiento rápido
- Al menos 8GB de RAM
- Espacio en disco para el dataset y modelos

## Dependencias Principales

- `ultralytics` - Framework YOLO
- `torch` y `torchvision` - PyTorch para deep learning
- `opencv-python` - Procesamiento de imágenes
- `pillow` - Manipulación de imágenes
- `numpy` - Operaciones numéricas
- `matplotlib` y `seaborn` - Visualización de datos
- `pandas` - Análisis de datos
- `pyyaml` - Configuración

## Troubleshooting

### Error: "No se encontró data.yaml"
- Asegúrate de haber ejecutado `download_roboflow_simple.py` primero
- Verifica que existe el archivo `data/raw/data.yaml`

### Error: "CUDA out of memory"
- Reduce el batch size: `--batch 8` o `--batch 4`
- Usa un modelo más pequeño: `--model yolov8n.pt`

### Error: "No module named 'ultralytics'"
- Reactiva el entorno virtual: `source venv/bin/activate`
- Reinstala dependencias: `pip install -r requirements.txt`

## Referencias

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Documentation](https://docs.roboflow.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Licencia

MIT License

## Autor

Not-Minimal