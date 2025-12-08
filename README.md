# DeepLabScan

Proyecto semestral: Implementación de modelo YOLO para detección de poses/objetos o segmentación, usando datos propios etiquetados en Roboflow, con entrenamiento, evaluación (precisión, recall, mAP) e implementación final con pruebas.

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema de detección de objetos/poses o segmentación utilizando modelos YOLO (You Only Look Once). El proyecto incluye:
- Recolección y etiquetado de datos usando Roboflow
- Entrenamiento de modelos YOLO personalizados
- Evaluación con métricas estándar (Precision, Recall, mAP)
- Implementación y pruebas del modelo entrenado

## 📁 Estructura del Proyecto

```
DeepLabScan/
├── data/                    # Datos y anotaciones
│   ├── raw/                # Imágenes originales
│   ├── processed/          # Datos procesados
│   ├── annotations/        # Anotaciones manuales
│   └── roboflow/          # Datasets exportados de Roboflow
├── models/                 # Modelos y pesos
│   ├── pretrained/        # Pesos pre-entrenados
│   └── trained/           # Modelos entrenados
├── src/                    # Código fuente
│   ├── training/          # Scripts de entrenamiento
│   ├── inference/         # Scripts de inferencia
│   ├── evaluation/        # Scripts de evaluación
│   └── utils/             # Utilidades
├── configs/               # Archivos de configuración
├── notebooks/             # Jupyter notebooks
├── tests/                 # Tests unitarios
├── results/               # Resultados del entrenamiento
│   ├── metrics/          # Métricas (precision, recall, mAP)
│   ├── visualizations/   # Gráficos y visualizaciones
│   └── logs/             # Logs de entrenamiento
└── docs/                  # Documentación del proyecto
```

## 🚀 Inicio Rápido

### Prerrequisitos

- Python 3.8 o superior
- CUDA compatible GPU (recomendado para entrenamiento)
- Cuenta en Roboflow para etiquetado de datos

### Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/Not-Minimal/DeepLabScan.git
cd DeepLabScan
```

2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 📊 Flujo de Trabajo

### 1. Preparación de Datos

1. Recolectar imágenes para tu caso de uso
2. Subir a Roboflow y etiquetar los datos
3. Exportar en formato YOLO a `data/roboflow/`

```bash
# Ejemplo de estructura después de exportar:
data/roboflow/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 2. Configuración

Editar `configs/data_config.yaml` con las clases de tu dataset:
```yaml
names:
  0: clase_1
  1: clase_2
nc: 2  # número de clases
```

### 3. Entrenamiento

Entrenar el modelo (script a crear):
```bash
python src/training/train.py --config configs/training_config.yaml
```

### 4. Evaluación

Evaluar el modelo con métricas:
```bash
python src/evaluation/evaluate.py --model models/trained/best.pt
```

Métricas incluidas:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **mAP@0.5**: Mean Average Precision con IoU threshold de 0.5
- **mAP@0.5:0.95**: mAP promedio sobre IoU thresholds

### 5. Inferencia

Ejecutar inferencia en nuevas imágenes:
```bash
python src/inference/predict.py --model models/trained/best.pt --source path/to/images
```

## 📈 Resultados

Los resultados del entrenamiento se guardan en `results/`:
- Métricas en formato CSV y JSON en `results/metrics/`
- Visualizaciones de predicciones en `results/visualizations/`
- Logs de TensorBoard en `results/logs/`

Para visualizar con TensorBoard:
```bash
tensorboard --logdir results/logs
```

## 🧪 Testing

Ejecutar tests:
```bash
pytest tests/
```

Con cobertura:
```bash
pytest --cov=src tests/
```

## 📝 Documentación

La documentación completa del proyecto está en el directorio `docs/`:
- Propuesta del proyecto
- Metodología de recolección de datos
- Proceso de entrenamiento
- Resultados y análisis
- Guía de despliegue

## 🛠️ Tecnologías

- **Framework**: PyTorch
- **Modelo**: YOLO (YOLOv5/YOLOv8)
- **Etiquetado**: Roboflow
- **Visualización**: TensorBoard, Matplotlib
- **Testing**: pytest

## 👥 Contribución

Este es un proyecto académico. Para contribuir:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es parte de un proyecto semestral académico.

## 📧 Contacto

Proyecto Link: [https://github.com/Not-Minimal/DeepLabScan](https://github.com/Not-Minimal/DeepLabScan)

## 🙏 Agradecimientos

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8
- [Roboflow](https://roboflow.com/) - Plataforma de etiquetado
- Comunidad de PyTorch y Computer Vision