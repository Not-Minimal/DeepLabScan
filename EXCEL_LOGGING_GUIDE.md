# 📊 Guía Completa del Sistema de Logging en Excel

Esta guía explica cómo usar el sistema de registro de resultados en Excel para DeepLabScan, que permite guardar y comparar resultados de entrenamientos, evaluaciones y predicciones.

## 🎯 Características Principales

- ✅ **Guardado automático** de resultados en Excel
- ✅ **4 hojas organizadas**: Resumen, Training, Evaluation, Prediction
- ✅ **Comparación fácil** entre experimentos
- ✅ **Identificación automática** del mejor modelo
- ✅ **Formato profesional** con colores y columnas ajustadas
- ✅ **Exportación a CSV** para análisis adicional

## 📁 Estructura de Archivos

```
DeepLabScan/
├── scripts/
│   ├── train.py              # Script de entrenamiento (con Excel logging)
│   ├── evaluate.py           # Script de evaluación (con Excel logging)
│   ├── predict.py            # Script de predicción (con Excel logging)
│   ├── excel_logger.py       # Módulo de logging
│   ├── view_results.py       # Visualización de resultados
│   └── test_excel_logger.py  # Script de prueba
├── results/
│   ├── experiment_results.xlsx  # Archivo principal con todos los resultados
│   └── README.md               # Documentación del sistema
└── EXCEL_LOGGING_GUIDE.md      # Esta guía
```

## 🚀 Inicio Rápido

### 1. Instalar Dependencias

```bash
pip install pandas openpyxl
```

O instalar todo el proyecto:

```bash
pip install -r requirements.txt
```

### 2. Entrenar y Guardar Resultados

```bash
# Entrenamiento básico (guarda automáticamente en Excel)
python scripts/train.py --data-dir data/raw --epochs 15

# Entrenamiento con nombre personalizado y notas
python scripts/train.py \
    --data-dir data/raw \
    --model yolo11n.pt \
    --epochs 30 \
    --name "exp_yolo11n_v1" \
    --notes "Primer entrenamiento con yolo11n"
```

### 3. Evaluar y Guardar Resultados

```bash
# Evaluación básica
python scripts/evaluate.py --weights runs/detect/train/weights/best.pt

# Evaluación con nombre personalizado
python scripts/evaluate.py \
    --weights runs/detect/train/weights/best.pt \
    --exp-name "eval_modelo_v1" \
    --notes "Evaluación inicial del modelo"
```

### 4. Predecir y Guardar Resultados

```bash
# Predicción básica
python scripts/predict.py \
    --weights runs/detect/train/weights/best.pt \
    --source test_image.jpg

# Predicción con configuración personalizada
python scripts/predict.py \
    --weights runs/detect/train/weights/best.pt \
    --source test_images/ \
    --conf 0.3 \
    --exp-name "pred_test_v1" \
    --notes "Predicciones con confidence 0.3"
```

### 5. Ver Resultados

```bash
# Ver resumen de todos los experimentos
python scripts/view_results.py

# Ver mejor modelo
python scripts/view_results.py --best-model

# Ver últimos 5 experimentos
python scripts/view_results.py --summary --last 5
```

## 📊 Hojas del Excel

### 🔍 Hoja "Resumen"

Vista comparativa de todos los experimentos en una sola hoja:

| Fecha | Hora | Tipo | Experimento | Modelo | Épocas | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1-Score | Detecciones | Notas |
|-------|------|------|-------------|--------|--------|---------|--------------|-----------|--------|----------|-------------|-------|
| 2024-01-15 | 14:30:00 | Training | train_v1 | yolo11n.pt | 30 | 0.8542 | 0.6234 | 0.8123 | 0.7856 | 0.7988 | - | Primer modelo |
| 2024-01-15 | 15:45:00 | Evaluation | eval_v1 | train | - | 0.8521 | 0.6198 | 0.8101 | 0.7834 | 0.7966 | - | Evaluación val |
| 2024-01-15 | 16:00:00 | Prediction | pred_test | train | - | - | - | - | - | - | 45 | Test set |

**Uso**: Ideal para comparar rápidamente todos los experimentos y encontrar tendencias.

### 🏋️ Hoja "Training"

Detalles completos de entrenamientos:

- Configuración del modelo
- Hiperparámetros (épocas, batch, imgsz, device)
- Duración del entrenamiento en minutos
- Métricas finales (mAP, Precision, Recall, Loss)
- Ruta a los pesos guardados

**Uso**: Analizar qué configuraciones de entrenamiento funcionan mejor.

### 📈 Hoja "Evaluation"

Resultados de evaluaciones:

- Modelo evaluado (weights path)
- Dataset y split (val/test)
- Métricas de rendimiento detalladas
- Número de clases detectadas
- Cantidad de visualizaciones generadas

**Uso**: Comparar el rendimiento del mismo modelo en diferentes datasets.

### 🎯 Hoja "Prediction"

Historial de predicciones:

- Configuración (confidence, IoU)
- Total de imágenes procesadas
- Total de detecciones realizadas
- Detecciones por clase
- Directorio de salida

**Uso**: Rastrear predicciones realizadas y sus resultados.

## 🔧 Opciones Avanzadas

### Desactivar Excel Logging

Si no quieres guardar en Excel temporalmente:

```bash
# Sin guardar en Excel
python scripts/train.py --data-dir data/raw --epochs 15 --no-excel
python scripts/evaluate.py --weights best.pt --no-excel
python scripts/predict.py --weights best.pt --source img.jpg --no-excel
```

### Nombres de Experimentos Personalizados

```bash
# Training con nombre personalizado
python scripts/train.py \
    --data-dir data/raw \
    --epochs 30 \
    --name "yolo11n_aug_batch16" \
    --notes "Con data augmentation, batch 16"

# Evaluation con nombre personalizado
python scripts/evaluate.py \
    --weights best.pt \
    --exp-name "eval_test_set" \
    --notes "Evaluación en test set"

# Prediction con nombre personalizado
python scripts/predict.py \
    --weights best.pt \
    --source images/ \
    --exp-name "pred_production" \
    --notes "Predicciones en producción"
```

### Análisis Detallado

```bash
# Ver solo entrenamientos
python scripts/view_results.py --training

# Ver solo evaluaciones
python scripts/view_results.py --evaluation

# Ver solo predicciones
python scripts/view_results.py --prediction

# Comparar experimentos con estadísticas
python scripts/view_results.py --compare

# Exportar a CSV
python scripts/view_results.py --export results/mi_analisis.csv
```

## 💻 Uso Programático

Puedes usar el logger directamente en tus propios scripts:

```python
from scripts.excel_logger import ExcelLogger

# Crear logger
logger = ExcelLogger("results/experiment_results.xlsx")

# Registrar entrenamiento
logger.log_training(
    experiment_name="mi_experimento",
    model="yolo11n.pt",
    dataset="data/raw",
    epochs=30,
    batch=16,
    imgsz=640,
    device="cuda",
    duration_minutes=45.5,
    best_map50=0.85,
    best_map50_95=0.65,
    best_precision=0.82,
    best_recall=0.78,
    final_loss=0.12,
    weights_path="runs/detect/train/weights/best.pt",
    notes="Experimento con nuevos datos"
)

# Registrar evaluación
logger.log_evaluation(
    experiment_name="eval_mi_modelo",
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

# Registrar predicción
logger.log_prediction(
    experiment_name="pred_produccion",
    weights_path="runs/detect/train/weights/best.pt",
    source="images/produccion/",
    confidence=0.25,
    iou=0.7,
    device="cuda",
    total_images=100,
    total_detections=450,
    class_counts={"objeto_a": 200, "objeto_b": 250},
    output_dir="runs/predict/produccion",
    notes="Predicciones en ambiente de producción"
)

# Obtener mejor modelo
best_model = logger.get_best_model(metric="mAP@0.5")
print(f"Mejor modelo: {best_model['Experimento']}")
print(f"mAP@0.5: {best_model['mAP@0.5']}")

# Obtener DataFrame con todos los resultados
df = logger.get_summary_dataframe()
print(df.head())

# Filtrar por tipo de experimento
df_training = df[df["Tipo"] == "Training"]
print(f"Total entrenamientos: {len(df_training)}")
```

## 🧪 Probar el Sistema

Ejecuta el script de prueba para generar datos de ejemplo:

```bash
# Generar datos de prueba (5 de cada tipo)
python scripts/test_excel_logger.py

# Generar más datos
python scripts/test_excel_logger.py --num-train 10 --num-eval 10 --num-predict 10

# Usar archivo de prueba diferente
python scripts/test_excel_logger.py --excel-path results/test.xlsx

# Limpiar y empezar de nuevo
python scripts/test_excel_logger.py --clean
```

Luego visualiza los resultados de prueba:

```bash
python scripts/view_results.py --excel-path results/test_experiment_results.xlsx
```

## 📈 Interpretación de Métricas

### mAP (Mean Average Precision)

- **mAP@0.5**: Precisión promedio con umbral de IoU de 0.5
  - ≥ 0.9: Excelente
  - 0.7-0.9: Bueno
  - 0.5-0.7: Aceptable
  - < 0.5: Requiere mejoras

- **mAP@0.5:0.95**: Promedio de mAP desde IoU 0.5 hasta 0.95
  - Métrica más estricta y realista
  - Típicamente 30-40% menor que mAP@0.5

### Precision y Recall

- **Precision**: De todas las detecciones, ¿cuántas son correctas?
  - Alta precision = Pocos falsos positivos
  - Baja precision = Muchos falsos positivos

- **Recall**: De todos los objetos reales, ¿cuántos detectamos?
  - Alto recall = Pocos falsos negativos
  - Bajo recall = Muchos falsos negativos

### F1-Score

- Media armónica entre Precision y Recall
- Balance entre ambas métricas
- Útil cuando quieres optimizar ambas simultáneamente

## 💡 Mejores Prácticas

### 1. Nombres Descriptivos

```bash
# ❌ Malo
python scripts/train.py --data-dir data/raw --epochs 15

# ✅ Bueno
python scripts/train.py \
    --data-dir data/raw \
    --epochs 15 \
    --name "yolo11n_baseline_20240115" \
    --notes "Baseline sin augmentation para comparación"
```

### 2. Documentar Cambios

Usa el campo `--notes` para documentar:
- Cambios en el dataset
- Modificaciones de hiperparámetros
- Experimentos A/B
- Observaciones importantes

```bash
python scripts/train.py \
    --data-dir data/raw \
    --epochs 30 \
    --notes "Aumentado rotation=15deg, flip=horizontal, brightness=0.2"
```

### 3. Versionado de Experimentos

```bash
# Versión 1: Baseline
python scripts/train.py --name "v1_baseline" --notes "Sin augmentation"

# Versión 2: Con augmentation
python scripts/train.py --name "v2_augmented" --notes "Con data augmentation"

# Versión 3: Más épocas
python scripts/train.py --name "v3_more_epochs" --epochs 50 --notes "50 épocas"
```

### 4. Backup Regular

```bash
# Hacer backup del Excel
cp results/experiment_results.xlsx results/backup_$(date +%Y%m%d).xlsx

# O usar script automatizado
#!/bin/bash
BACKUP_DIR="results/backups"
mkdir -p $BACKUP_DIR
cp results/experiment_results.xlsx \
   $BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).xlsx
echo "Backup creado en $BACKUP_DIR"
```

### 5. Análisis Periódico

```bash
# Cada semana, revisa tus experimentos
python scripts/view_results.py --best-model
python scripts/view_results.py --compare

# Exporta para análisis más profundo
python scripts/view_results.py --export results/weekly_report.csv
```

## 🔍 Casos de Uso

### Caso 1: Optimización de Hiperparámetros

```bash
# Probar diferentes batch sizes
python scripts/train.py --batch 8 --name "exp_batch8" --notes "batch=8"
python scripts/train.py --batch 16 --name "exp_batch16" --notes "batch=16"
python scripts/train.py --batch 32 --name "exp_batch32" --notes "batch=32"

# Comparar resultados
python scripts/view_results.py --training
```

### Caso 2: Comparación de Modelos

```bash
# Entrenar diferentes modelos
python scripts/train.py --model yolo11n.pt --name "exp_yolo11n"
python scripts/train.py --model yolo11s.pt --name "exp_yolo11s"
python scripts/train.py --model yolov8n.pt --name "exp_yolov8n"

# Encontrar el mejor
python scripts/view_results.py --best-model
```

### Caso 3: Evaluación en Múltiples Datasets

```bash
# Evaluar en validation y test
python scripts/evaluate.py --weights best.pt --split val --exp-name "eval_val"
python scripts/evaluate.py --weights best.pt --split test --exp-name "eval_test"

# Comparar resultados
python scripts/view_results.py --evaluation
```

## ❓ Solución de Problemas

### Problema: "Excel logger no disponible"

**Solución**:
```bash
pip install pandas openpyxl
```

### Problema: El archivo Excel no se actualiza

**Causas posibles**:
1. El archivo está abierto en Excel → Ciérralo
2. Permisos insuficientes → Verifica permisos en `results/`
3. Disco lleno → Libera espacio

**Solución temporal**:
```bash
python scripts/train.py --no-excel  # Entrenar sin guardar en Excel
```

### Problema: Métricas aparecen como 0 o vacías

**Causas**:
- El entrenamiento no se completó correctamente
- No existe el archivo `results.csv` en el directorio del experimento
- Error al leer las métricas de YOLO

**Solución**:
1. Verifica que el entrenamiento se completó sin errores
2. Revisa las notas en el Excel para más detalles
3. Busca el archivo `results.csv` en `runs/detect/train*/`

### Problema: "Permission denied" al escribir Excel

**Solución**:
```bash
# Cambiar permisos del directorio
chmod -R 755 results/

# O crear nuevo archivo
python scripts/train.py --excel-path results/nuevo_results.xlsx
```

## 📚 Referencias

- [Documentación Ultralytics YOLO](https://docs.ultralytics.com/)
- [Pandas Excel Writer](https://pandas.pydata.org/docs/reference/api/pandas.ExcelWriter.html)
- [OpenPyXL Documentation](https://openpyxl.readthedocs.io/)

## 🤝 Contribuir

Si encuentras bugs o quieres agregar funcionalidades:

1. Reporta el issue con descripción detallada
2. Propón mejoras con ejemplos de uso
3. Comparte tus análisis y mejores prácticas

## 📝 Changelog

### v1.0.0 (2024)
- ✨ Sistema inicial de logging en Excel
- ✨ 4 hojas: Resumen, Training, Evaluation, Prediction
- ✨ Script de visualización de resultados
- ✨ Script de prueba con datos de ejemplo
- ✨ Formato automático con colores
- ✨ Identificación del mejor modelo
- ✨ Exportación a CSV

---

**¡Listo!** Ya puedes empezar a registrar y comparar tus experimentos de manera profesional. 🚀

Para preguntas o soporte, revisa la documentación en `results/README.md` o ejecuta:

```bash
python scripts/view_results.py --help
python scripts/test_excel_logger.py --help
```
