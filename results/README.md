# Sistema de Registro de Resultados en Excel

Este directorio contiene los resultados de todos los experimentos de entrenamiento, evaluación y predicción guardados en formato Excel para facilitar la comparación y análisis.

## Archivo Principal

**`experiment_results.xlsx`** - Contiene todos los resultados organizados en múltiples hojas:

### 📊 Hojas del Excel

#### 1. **Resumen** (Vista General)
Contiene todos los experimentos en una sola hoja para comparación rápida:
- Fecha y Hora
- Tipo de experimento (Training/Evaluation/Prediction)
- Nombre del experimento
- Modelo utilizado
- Métricas clave (mAP, Precision, Recall, F1-Score)
- Notas adicionales

#### 2. **Training** (Entrenamientos)
Detalles completos de cada entrenamiento:
- Configuración del modelo
- Hiperparámetros (épocas, batch, imgsz)
- Dispositivo utilizado
- Duración del entrenamiento
- Métricas finales
- Ruta a los pesos guardados

#### 3. **Evaluation** (Evaluaciones)
Resultados de evaluación de modelos:
- Modelo evaluado (ruta a weights)
- Dataset y split utilizados
- Métricas de rendimiento
- Número de clases detectadas
- Visualizaciones generadas

#### 4. **Prediction** (Predicciones)
Historial de predicciones realizadas:
- Modelo utilizado
- Fuente de datos (imágenes/video)
- Parámetros (confidence, IoU)
- Total de detecciones
- Detecciones por clase
- Directorio de salida

## Uso Automático

Los scripts ya están configurados para guardar automáticamente en Excel:

### Entrenamiento
```bash
# Los resultados se guardan automáticamente al finalizar
python scripts/train.py --data-dir data/raw --epochs 15 --notes "Mi primer entrenamiento"

# Sin guardar en Excel
python scripts/train.py --data-dir data/raw --epochs 15 --no-excel
```

### Evaluación
```bash
# Guardar evaluación con nombre personalizado
python scripts/evaluate.py --weights runs/detect/train/weights/best.pt --exp-name "eval_v1" --notes "Evaluación inicial"

# Sin guardar en Excel
python scripts/evaluate.py --weights best.pt --no-excel
```

### Predicción
```bash
# Guardar predicciones
python scripts/predict.py --weights best.pt --source imagen.jpg --exp-name "pred_test" --notes "Prueba de detección"

# Sin guardar en Excel
python scripts/predict.py --weights best.pt --source imagen.jpg --no-excel
```

## Visualización y Análisis

### Ver Resumen de Todos los Experimentos
```bash
python scripts/view_results.py
```

### Ver Solo Últimos 5 Experimentos
```bash
python scripts/view_results.py --summary --last 5
```

### Encontrar Mejor Modelo
```bash
python scripts/view_results.py --best-model
```

### Ver Detalles de Entrenamientos
```bash
python scripts/view_results.py --training
```

### Ver Detalles de Evaluaciones
```bash
python scripts/view_results.py --evaluation
```

### Ver Detalles de Predicciones
```bash
python scripts/view_results.py --prediction
```

### Comparar Experimentos
```bash
python scripts/view_results.py --compare
```

### Exportar a CSV
```bash
python scripts/view_results.py --export results/mi_resumen.csv
```

## Uso Programático

También puedes usar el logger desde tus propios scripts:

```python
from scripts.excel_logger import ExcelLogger

# Crear logger
logger = ExcelLogger("results/experiment_results.xlsx")

# Guardar entrenamiento
logger.log_training(
    experiment_name="mi_exp_1",
    model="yolo11n.pt",
    dataset="data/raw",
    epochs=15,
    batch=16,
    imgsz=640,
    device="cuda",
    duration_minutes=25.5,
    best_map50=0.85,
    best_map50_95=0.65,
    best_precision=0.82,
    best_recall=0.78,
    final_loss=0.15,
    weights_path="runs/detect/train/weights/best.pt",
    notes="Experimento de prueba"
)

# Guardar evaluación
logger.log_evaluation(
    experiment_name="eval_1",
    weights_path="runs/detect/train/weights/best.pt",
    dataset="data/raw",
    split="val",
    device="cuda",
    precision=0.82,
    recall=0.78,
    map50=0.85,
    map50_95=0.65,
    classes_detected=3,
    notes="Evaluación en validation set"
)

# Guardar predicción
logger.log_prediction(
    experiment_name="pred_1",
    weights_path="runs/detect/train/weights/best.pt",
    source="test_images/",
    confidence=0.25,
    iou=0.7,
    device="cuda",
    total_images=10,
    total_detections=45,
    class_counts={"clase1": 20, "clase2": 25},
    output_dir="runs/predict/exp",
    notes="Predicciones en imágenes de prueba"
)

# Obtener resumen
df = logger.get_summary_dataframe()
print(df)

# Encontrar mejor modelo
best = logger.get_best_model(metric="mAP@0.5")
print(f"Mejor modelo: {best['Experimento']} con mAP@0.5 = {best['mAP@0.5']}")
```

## Métricas Registradas

### Métricas Principales
- **mAP@0.5**: Mean Average Precision con IoU threshold de 0.5
- **mAP@0.5:0.95**: Mean Average Precision promediado desde IoU 0.5 hasta 0.95
- **Precision**: Proporción de detecciones correctas sobre todas las detecciones
- **Recall**: Proporción de objetos detectados sobre todos los objetos reales
- **F1-Score**: Media armónica entre Precision y Recall

### Interpretación de Valores
- **Excelente**: mAP ≥ 0.9
- **Bueno**: mAP entre 0.7 - 0.9
- **Aceptable**: mAP entre 0.5 - 0.7
- **Bajo**: mAP < 0.5 (requiere más entrenamiento)

## Estructura del Excel

```
experiment_results.xlsx
├── Resumen (Vista comparativa de todos los experimentos)
├── Training (Detalles de entrenamientos)
├── Evaluation (Detalles de evaluaciones)
└── Prediction (Detalles de predicciones)
```

## Consejos

### 📝 Usar Nombres Descriptivos
```bash
# Malo
python scripts/train.py --data-dir data/raw --epochs 15

# Bueno
python scripts/train.py --data-dir data/raw --epochs 15 --name "exp_yolo11n_aug_v1" --notes "Primer entrenamiento con data augmentation"
```

### 📊 Comparar Experimentos
1. Abre `experiment_results.xlsx` en Excel
2. Ve a la hoja "Resumen"
3. Usa filtros y ordenamiento para comparar
4. Crea gráficos personalizados según tus necesidades

### 🔍 Analizar Tendencias
```bash
# Ver estadísticas de todos los entrenamientos
python scripts/view_results.py --training

# Ver mejor modelo por diferentes métricas
python scripts/view_results.py --best-model
```

### 💾 Backup
Haz copias de seguridad periódicas del archivo Excel:
```bash
cp results/experiment_results.xlsx results/backup_$(date +%Y%m%d).xlsx
```

## Solución de Problemas

### Error: "Excel logger no disponible"
Instala las dependencias:
```bash
pip install pandas openpyxl
```

### El archivo Excel no se actualiza
1. Cierra el archivo Excel si está abierto
2. Verifica permisos de escritura en el directorio `results/`
3. Usa `--no-excel` temporalmente si hay problemas

### Métricas no aparecen
- Asegúrate de que el entrenamiento se completó exitosamente
- Verifica que existe el archivo `results.csv` en el directorio del experimento
- Revisa las notas en el Excel para más detalles

## Referencias

- **Documentación YOLO**: https://docs.ultralytics.com/
- **Pandas Excel**: https://pandas.pydata.org/docs/reference/api/pandas.ExcelWriter.html
- **OpenPyXL**: https://openpyxl.readthedocs.io/

---

**Nota**: Este sistema de registro facilita el seguimiento de experimentos y la comparación de diferentes configuraciones de modelos. ¡Úsalo para optimizar tus modelos de detección!