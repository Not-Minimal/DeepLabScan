# DeepLabScan

Proyecto de detección de objetos usando YOLO con datos propios etiquetados en Roboflow.

## Descripción

Este proyecto implementa un pipeline completo de detección de objetos utilizando:
- Datos propios etiquetados en Roboflow
- Aumentación de datos para balanceo de clases
- Entrenamiento con YOLO (Ultralytics)
- Evaluación con métricas estándar (mAP, Precision, Recall)
- **📊 Sistema de registro en Excel** para comparar experimentos

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

**Nota**: Incluye soporte para guardar resultados en Excel (`pandas`, `openpyxl`)

## Uso

### 1. Descargar Dataset desde Roboflow

```bash
python scripts/download_roboflow_simple.py
```

Este script:
- Descarga el dataset desde Roboflow (URL configurada en el script)
- Crea la estructura necesaria en `data/raw/`

### 2. Entrenar Modelo

```bash
# Entrenamiento básico (guarda automáticamente en Excel)
python scripts/train.py --data-dir data/raw --epochs 15

# Con configuración personalizada y notas
python scripts/train.py \
    --data-dir data/raw \
    --model yolo11n.pt \
    --epochs 30 \
    --name "exp_v1" \
    --notes "Primer entrenamiento con augmentation"
```

Los resultados se guardan automáticamente en `results/experiment_results.xlsx`.

### 3. Evaluar Modelo

```bash
# Evaluación básica
python scripts/evaluate.py --weights runs/detect/train/weights/best.pt

# Con nombre personalizado
python scripts/evaluate.py \
    --weights runs/detect/train/weights/best.pt \
    --exp-name "eval_test" \
    --notes "Evaluación en test set"
```

### 4. Realizar Predicciones

```bash
# Predicción en imagen
python scripts/predict.py \
    --weights runs/detect/train/weights/best.pt \
    --source test_image.jpg

# Predicción en directorio
python scripts/predict.py \
    --weights runs/detect/train/weights/best.pt \
    --source test_images/ \
    --conf 0.3 \
    --exp-name "pred_v1" \
    --notes "Predicciones con confidence 0.3"
```

## 📊 Sistema de Logging en Excel

### Características

- ✅ **Guardado automático** de todos los resultados
- ✅ **4 hojas organizadas**: Resumen, Training, Evaluation, Prediction
- ✅ **Comparación fácil** entre experimentos
- ✅ **Identificación automática** del mejor modelo
- ✅ **Formato profesional** con colores

### Ver Resultados

```bash
# Ver resumen de todos los experimentos
python scripts/view_results.py

# Ver últimos 10 experimentos
python scripts/view_results.py --summary --last 10

# Encontrar mejor modelo
python scripts/view_results.py --best-model

# Ver solo entrenamientos
python scripts/view_results.py --training

# Comparar experimentos
python scripts/view_results.py --compare

# Exportar a CSV
python scripts/view_results.py --export results/mi_analisis.csv
```

### Archivo Excel

El archivo `results/experiment_results.xlsx` contiene:

1. **Hoja Resumen**: Todos los experimentos para comparación rápida
2. **Hoja Training**: Detalles de entrenamientos (hiperparámetros, duración, métricas)
3. **Hoja Evaluation**: Resultados de evaluaciones (precision, recall, mAP)
4. **Hoja Prediction**: Historial de predicciones (detecciones, clases)

### Desactivar Excel Logging

Si no quieres guardar en Excel:

```bash
python scripts/train.py --data-dir data/raw --epochs 15 --no-excel
python scripts/evaluate.py --weights best.pt --no-excel
python scripts/predict.py --weights best.pt --source img.jpg --no-excel
```

### Probar el Sistema

Genera datos de ejemplo para ver cómo funciona:

```bash
python scripts/test_excel_logger.py
```

### Documentación Completa

Para más detalles, consulta:
- 📖 **Guía completa**: `EXCEL_LOGGING_GUIDE.md`
- 📁 **Documentación del sistema**: `results/README.md`

## Estructura del Proyecto

```
DeepLabScan/
├── scripts/
│   ├── train.py              # Entrenamiento con Excel logging
│   ├── evaluate.py           # Evaluación con Excel logging
│   ├── predict.py            # Predicción con Excel logging
│   ├── excel_logger.py       # Módulo de logging
│   ├── view_results.py       # Visualización de resultados
│   └── test_excel_logger.py  # Script de prueba
├── results/
│   ├── experiment_results.xlsx  # Archivo principal
│   └── README.md               # Documentación
├── data/
│   └── raw/                   # Dataset
├── runs/                      # Resultados de entrenamientos
├── EXCEL_LOGGING_GUIDE.md    # Guía completa del sistema
└── requirements.txt           # Dependencias

```

## Métricas y Resultados

### Métricas Registradas

- **mAP@0.5**: Mean Average Precision con IoU=0.5
- **mAP@0.5:0.95**: mAP promediado desde IoU 0.5 hasta 0.95
- **Precision**: Proporción de detecciones correctas
- **Recall**: Proporción de objetos detectados
- **F1-Score**: Media armónica entre Precision y Recall

### Interpretación

- **Excelente**: mAP ≥ 0.9
- **Bueno**: mAP 0.7-0.9
- **Aceptable**: mAP 0.5-0.7
- **Bajo**: mAP < 0.5 (requiere mejoras)

## Mejores Prácticas

### 1. Usa nombres descriptivos

```bash
python scripts/train.py \
    --name "yolo11n_aug_batch16_v1" \
    --notes "Con data augmentation, batch 16"
```

### 2. Documenta tus experimentos

```bash
python scripts/train.py \
    --notes "Baseline sin augmentation para comparación"
```

### 3. Compara regularmente

```bash
python scripts/view_results.py --best-model
python scripts/view_results.py --compare
```

### 4. Haz backups

```bash
cp results/experiment_results.xlsx results/backup_$(date +%Y%m%d).xlsx
- Descomprime automáticamente los archivos
- Organiza el dataset en `data/raw/` con la estructura YOLO esperada
- Mueve `data.yaml`, `train/`, `valid/`, `test/` a la ubicación correcta

**Nota:** Si necesitas cambiar la URL del dataset, edita la variable `DEFAULT_URL` en el script o pasa `--url` como argumento.

### 2. (Opcional) Aumentación de Datos

Balancea las clases minoritarias generando nuevas imágenes con aumentación (flip, rotación, brillo):

```bash
# Aumentación básica con límite del 25%
python scripts/augment_data.py --data-dir data/raw

# Con límite personalizado y semilla específica
python scripts/augment_data.py --data-dir data/raw --limit 0.25 --seed 42

# Guardando gráficos en disco
python scripts/augment_data.py --data-dir data/raw --save-plots
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
# Entrenamiento básico con 15 épocas
python scripts/train.py --data-dir data/raw --epochs 15

# Con modelo YOLO11n y batch size de 16
python scripts/train.py --data-dir data/raw --model yolo11n.pt --epochs 50 --batch 16

# Con modelo YOLOv8n, 100 épocas y CUDA
python scripts/train.py --data-dir data/raw --model yolov8n.pt --epochs 100 --imgsz 640 --device cuda
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
# Evaluación básica (muestra métricas en terminal)
python scripts/evaluate.py --weights runs/detect/train/weights/best.pt

# Con directorio del dataset especificado
python scripts/evaluate.py --weights runs/detect/train/weights/best.pt --data-dir data/raw

# Guardando gráficos en disco
python scripts/evaluate.py --weights runs/detect/train/weights/best.pt --save-plots

# Sin mostrar gráficos (headless)
python scripts/evaluate.py --weights runs/detect/train/weights/best.pt --no-show
```

Genera:
- Métricas de Precision, Recall, mAP@0.5, mAP@0.5:0.95
- Matriz de confusión
- Curvas PR (Precision-Recall)
- Visualizaciones de predicciones

### 5. Realizar Predicciones

```bash
# Predicción en una imagen
python scripts/predict.py --weights runs/detect/train2/weights/best.pt --source test.jpg

# Predicción en directorio con visualización
python scripts/predict.py --weights runs/detect/train2/weights/best.pt --source images/ --show

# Predicción en video
python scripts/predict.py --weights runs/detect/train2/weights/best.pt --source video.mp4

# Predicción con webcam
python scripts/predict.py --weights runs/detect/train2/weights/best.pt --source 0 --show

# Con ajustes de confidence
python scripts/predict.py --weights runs/detect/train2/weights/best.pt --source test.jpg --conf 0.5

# Guardando labels en formato YOLO
python scripts/predict.py --weights runs/detect/train2/weights/best.pt --source test.jpg --save-txt
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