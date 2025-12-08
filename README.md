# DeepLabScan

Proyecto semestral de detección de objetos usando YOLO con datos propios etiquetados en Roboflow.

## Descripción del Proyecto

Este proyecto implementa un modelo YOLO (You Only Look Once) para detección de objetos utilizando:
- Datos propios etiquetados en Roboflow
- Entrenamiento con YOLOv8 de Ultralytics
- Evaluación con métricas de precisión, recall y mAP
- Implementación final con pruebas

## Estructura del Proyecto

```
DeepLabScan/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Carga de datos desde Roboflow
│   │   └── augmentation.py    # Aumentación de datos
│   ├── models/
│   │   ├── __init__.py
│   │   ├── yolo_model.py      # Configuración del modelo YOLO
│   │   └── trainer.py         # Entrenamiento del modelo
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py         # Métricas (precisión, recall, mAP)
│   └── utils/
│       ├── __init__.py
│       └── visualization.py   # Visualización de resultados
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_metrics.py
├── scripts/
│   ├── train.py               # Script de entrenamiento
│   ├── evaluate.py            # Script de evaluación
│   └── predict.py             # Script de predicción
├── configs/
│   └── config.yaml            # Configuración del proyecto
├── requirements.txt
├── .gitignore
└── README.md
```

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/Not-Minimal/DeepLabScan.git
cd DeepLabScan
```

2. Crear un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar Roboflow API:
```bash
# Crear archivo .env con tu API key de Roboflow
echo "ROBOFLOW_API_KEY=tu_api_key_aqui" > .env
```

## Uso

### 1. Descargar Datos desde Roboflow

```python
from src.data.loader import RoboflowDataLoader

loader = RoboflowDataLoader(
    workspace="tu_workspace",
    project="tu_proyecto",
    version=1
)
loader.download_dataset(location="./data/raw")
```

### 2. Entrenar el Modelo

```bash
python scripts/train.py --config configs/config.yaml
```

Parámetros disponibles:
- `--epochs`: Número de épocas (default: 100)
- `--batch-size`: Tamaño del batch (default: 16)
- `--img-size`: Tamaño de las imágenes (default: 640)
- `--device`: Dispositivo (cpu/cuda/mps)

### 3. Evaluar el Modelo

```bash
python scripts/evaluate.py --weights runs/train/exp/weights/best.pt
```

Esto generará:
- Métricas de precisión (Precision)
- Métricas de recall
- mAP@0.5 y mAP@0.5:0.95
- Matriz de confusión
- Curvas PR (Precision-Recall)

### 4. Realizar Predicciones

```bash
python scripts/predict.py --weights runs/train/exp/weights/best.pt --source path/to/images
```

## Métricas de Evaluación

El proyecto implementa las siguientes métricas:

- **Precision**: Proporción de detecciones correctas sobre todas las detecciones
- **Recall**: Proporción de objetos detectados sobre todos los objetos reales
- **mAP@0.5**: Mean Average Precision con IoU threshold de 0.5
- **mAP@0.5:0.95**: Mean Average Precision promediando IoU thresholds de 0.5 a 0.95

## Configuración

Editar `configs/config.yaml` para ajustar:
- Parámetros del modelo
- Hiperparámetros de entrenamiento
- Rutas de datos
- Configuración de aumentación

## Pruebas

Ejecutar las pruebas:
```bash
pytest tests/ -v
```

Ejecutar con cobertura:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Estructura de Datos Roboflow

Los datos deben estar en formato YOLO:
```
data/
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

## Resultados Esperados

Después del entrenamiento y evaluación, obtendrás:
- Pesos del modelo entrenado (.pt)
- Gráficas de métricas (precisión, recall, mAP)
- Matriz de confusión
- Ejemplos de predicciones
- Reporte de evaluación completo

## Dependencias Principales

- **ultralytics**: Framework YOLOv8
- **torch**: PyTorch para deep learning
- **roboflow**: Integración con Roboflow
- **opencv-python**: Procesamiento de imágenes
- **matplotlib/seaborn**: Visualización

## Contribución

Este es un proyecto semestral. Para contribuir:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Crea un Pull Request

## Licencia

MIT License

## Contacto

Para preguntas o soporte, abrir un issue en el repositorio.

## Referencias

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Documentation](https://docs.roboflow.com/)
- [YOLO Paper](https://arxiv.org/abs/1506.02640)