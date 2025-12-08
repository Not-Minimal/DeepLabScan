# Getting Started Guide

## Configuración Inicial del Proyecto

### 1. Configurar el Entorno de Desarrollo

#### Crear y activar entorno virtual:
```bash
python -m venv venv

# En Linux/Mac:
source venv/bin/activate

# En Windows:
venv\Scripts\activate
```

#### Instalar dependencias:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configurar Roboflow

#### Crear cuenta y proyecto:
1. Visita [Roboflow](https://roboflow.com/) y crea una cuenta
2. Crea un nuevo proyecto seleccionando el tipo de tarea:
   - Object Detection (para detección de objetos)
   - Instance Segmentation (para segmentación)
   - Pose Estimation (para detección de poses)

#### Subir y etiquetar datos:
1. Sube tus imágenes al proyecto
2. Etiqueta las imágenes usando las herramientas de anotación
3. Aplica augmentaciones si es necesario
4. Genera una versión del dataset

#### Exportar dataset:
1. Ve a la versión del dataset
2. Exportar en formato **YOLO v8** o **YOLO v5**
3. Copiar el código de descarga
4. Descargar a `data/roboflow/`

Ejemplo de código para descargar:
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(1).download("yolov8")
```

### 3. Configurar el Dataset

#### Actualizar data_config.yaml:
```yaml
path: data/roboflow
train: train/images
val: valid/images
test: test/images

names:
  0: clase_1
  1: clase_2
  # Añadir más clases según tu dataset

nc: 2  # número de clases
```

#### Verificar estructura de datos:
```bash
python -c "from src.utils.data_loader import validate_dataset, get_dataset_statistics; \
           valid, issues = validate_dataset('data/roboflow'); \
           print('Dataset válido:', valid); \
           print('Issues:', issues); \
           print('Stats:', get_dataset_statistics('data/roboflow'))"
```

### 4. Configurar el Entrenamiento

#### Editar training_config.yaml:

Parámetros importantes a ajustar:
- `epochs`: Número de épocas (100-300 típicamente)
- `batch_size`: Tamaño del batch (8, 16, 32 dependiendo de GPU)
- `img_size`: Tamaño de imagen (640 estándar)
- `learning_rate`: Learning rate inicial
- `num_classes`: Debe coincidir con tu dataset

### 5. Entrenar el Modelo

#### Opción A: Usando YOLOv8 (Ultralytics):
```python
from ultralytics import YOLO

# Cargar modelo pre-entrenado
model = YOLO('yolov8n.pt')  # nano, s, m, l, x

# Entrenar
results = model.train(
    data='configs/data_config.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='experiment_1'
)
```

#### Opción B: Usando script personalizado:
```bash
python src/training/train.py \
    --config configs/training_config.yaml \
    --data configs/data_config.yaml
```

### 6. Monitorear el Entrenamiento

#### TensorBoard:
```bash
tensorboard --logdir results/logs
```

Abre http://localhost:6006 en tu navegador

### 7. Evaluar el Modelo

```bash
python src/evaluation/evaluate.py \
    --model models/trained/best.pt \
    --data data/roboflow/test
```

Esto generará:
- Precision, Recall, F1-Score
- mAP@0.5 y mAP@0.5:0.95
- Confusion Matrix
- Visualizaciones en `results/visualizations/`

### 8. Hacer Predicciones

#### Imagen única:
```bash
python src/inference/predict.py \
    --model models/trained/best.pt \
    --source path/to/image.jpg \
    --conf 0.25
```

#### Directorio de imágenes:
```bash
python src/inference/predict.py \
    --model models/trained/best.pt \
    --source data/test_images/ \
    --output results/visualizations/predictions/
```

#### Video:
```bash
python src/inference/predict.py \
    --model models/trained/best.pt \
    --source path/to/video.mp4 \
    --output results/visualizations/video_output.mp4
```

### 9. Jupyter Notebooks

Inicia Jupyter para análisis exploratorio:
```bash
jupyter notebook notebooks/
```

Notebooks recomendados para crear:
1. `01_data_exploration.ipynb` - EDA del dataset
2. `02_training_experiments.ipynb` - Experimentos de entrenamiento
3. `03_results_analysis.ipynb` - Análisis de resultados
4. `04_inference_demo.ipynb` - Demo de inferencia

### 10. Documentación del Proyecto

A medida que avances, documenta en `docs/`:
- `project_proposal.md`: Propuesta inicial
- `methodology.md`: Metodología seguida
- `results.md`: Resultados y análisis
- `final_report.md`: Reporte final del semestre

## Checklist del Proyecto

- [ ] Configurar entorno de desarrollo
- [ ] Crear cuenta en Roboflow
- [ ] Recolectar y etiquetar datos
- [ ] Exportar dataset en formato YOLO
- [ ] Configurar archivos de configuración
- [ ] Entrenar modelo baseline
- [ ] Evaluar modelo (precision, recall, mAP)
- [ ] Experimentar con hiperparámetros
- [ ] Documentar resultados
- [ ] Hacer pruebas finales de inferencia
- [ ] Completar reporte final

## Recursos Adicionales

### Documentación Oficial:
- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [Roboflow Docs](https://docs.roboflow.com/)
- [PyTorch Docs](https://pytorch.org/docs/)

### Tutoriales:
- [YOLO Training Guide](https://docs.ultralytics.com/modes/train/)
- [Roboflow Tutorial](https://blog.roboflow.com/getting-started-with-roboflow/)

### Papers:
- YOLOv8: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Original YOLO: [You Only Look Once](https://arxiv.org/abs/1506.02640)

## Troubleshooting

### Problema: CUDA out of memory
**Solución**: Reducir `batch_size` en training_config.yaml

### Problema: Dataset no se carga correctamente
**Solución**: Verificar estructura con `validate_dataset()`

### Problema: mAP muy bajo
**Solución**: 
- Verificar calidad de etiquetado
- Aumentar épocas de entrenamiento
- Ajustar learning rate
- Probar con más data augmentation

### Problema: Overfitting
**Solución**:
- Aumentar data augmentation
- Usar early stopping
- Reducir complejidad del modelo
- Recolectar más datos
