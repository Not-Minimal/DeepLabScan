# Proyecto Semestral YOLO - Checklist

## Fase 1: Configuración Inicial ⚙️
- [ ] Clonar repositorio
- [ ] Ejecutar script de setup (`./setup.sh` o `setup.bat`)
- [ ] Verificar instalación de dependencias
- [ ] Crear cuenta en Roboflow
- [ ] Familiarizarse con estructura del proyecto

## Fase 2: Recolección de Datos 📸
- [ ] Definir el problema (detección de objetos/poses/segmentación)
- [ ] Determinar las clases a detectar
- [ ] Recolectar imágenes (mínimo 100-500 imágenes por clase)
- [ ] Organizar imágenes crudas en `data/raw/`
- [ ] Documentar el proceso en `docs/data_collection.md`

## Fase 3: Etiquetado en Roboflow 🏷️
- [ ] Crear proyecto en Roboflow
- [ ] Subir imágenes al proyecto
- [ ] Etiquetar todas las imágenes
  - [ ] Definir bounding boxes para detección
  - [ ] Definir keypoints para poses (si aplica)
  - [ ] Definir máscaras para segmentación (si aplica)
- [ ] Revisar calidad de las anotaciones
- [ ] Aplicar data augmentation (opcional)
  - [ ] Rotación
  - [ ] Flip horizontal/vertical
  - [ ] Ajustes de brillo/contraste
  - [ ] Recortes
- [ ] Dividir dataset (Train: 70%, Valid: 20%, Test: 10%)
- [ ] Generar versión del dataset
- [ ] Exportar en formato YOLO

## Fase 4: Preparación del Dataset 📊
- [ ] Descargar dataset de Roboflow a `data/roboflow/`
- [ ] Verificar estructura de archivos
  ```
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
- [ ] Actualizar `configs/data_config.yaml`
  - [ ] Agregar nombres de clases
  - [ ] Actualizar número de clases (nc)
  - [ ] Verificar paths
- [ ] Ejecutar notebook de exploración de datos
- [ ] Validar dataset con script de utilidad
  ```bash
  python -c "from src.utils.data_loader import validate_dataset; print(validate_dataset('data/roboflow'))"
  ```

## Fase 5: Configuración del Entrenamiento 🔧
- [ ] Revisar y ajustar `configs/training_config.yaml`
  - [ ] Elegir modelo (yolov8n, yolov8s, yolov8m, etc.)
  - [ ] Configurar epochs (100-300)
  - [ ] Ajustar batch_size según GPU
  - [ ] Configurar learning rate
  - [ ] Configurar augmentation
- [ ] Descargar pesos pre-entrenados a `models/pretrained/`
- [ ] Configurar TensorBoard para monitoreo

## Fase 6: Entrenamiento del Modelo 🏋️
- [ ] Entrenar modelo baseline
  ```bash
  python src/training/train.py --config configs/training_config.yaml
  ```
- [ ] Monitorear entrenamiento con TensorBoard
  ```bash
  tensorboard --logdir results/logs
  ```
- [ ] Guardar resultados del training en `results/`
- [ ] Documentar hiperparámetros usados
- [ ] Realizar experimentos adicionales (opcional)
  - [ ] Experimento 2: Ajustar learning rate
  - [ ] Experimento 3: Aumentar augmentation
  - [ ] Experimento 4: Probar otro modelo
- [ ] Seleccionar mejor modelo basado en métricas

## Fase 7: Evaluación del Modelo 📈
- [ ] Evaluar modelo en test set
  ```bash
  python src/evaluation/evaluate.py --model models/trained/best.pt --data data/roboflow/test
  ```
- [ ] Calcular métricas principales:
  - [ ] Precision
  - [ ] Recall
  - [ ] F1-Score
  - [ ] mAP@0.5
  - [ ] mAP@0.5:0.95
- [ ] Generar matriz de confusión
- [ ] Analizar errores comunes
- [ ] Visualizar predicciones en test set
- [ ] Crear gráficas de métricas
- [ ] Guardar resultados en `results/metrics/`
- [ ] Documentar resultados en `docs/evaluation_results.md`

## Fase 8: Pruebas de Inferencia 🔍
- [ ] Probar inferencia en imágenes individuales
  ```bash
  python src/inference/predict.py --model models/trained/best.pt --source test_image.jpg
  ```
- [ ] Probar inferencia en batch de imágenes
- [ ] Probar inferencia en video (si aplica)
- [ ] Ajustar threshold de confianza
- [ ] Validar tiempo de inferencia
- [ ] Guardar ejemplos de predicciones en `results/visualizations/`
- [ ] Crear demo interactivo (notebook)

## Fase 9: Documentación 📝
- [ ] Completar `docs/project_proposal.md`
  - [ ] Introducción y motivación
  - [ ] Objetivos del proyecto
  - [ ] Alcance
- [ ] Completar `docs/methodology.md`
  - [ ] Descripción del dataset
  - [ ] Proceso de etiquetado
  - [ ] Arquitectura del modelo
  - [ ] Hiperparámetros
- [ ] Completar `docs/evaluation_results.md`
  - [ ] Métricas obtenidas
  - [ ] Gráficas y visualizaciones
  - [ ] Análisis de resultados
  - [ ] Comparación de experimentos
- [ ] Crear presentación del proyecto
- [ ] Preparar demos para presentación

## Fase 10: Reporte Final 📄
- [ ] Escribir reporte final en `docs/final_report.md`
  - [ ] Resumen ejecutivo
  - [ ] Introducción
  - [ ] Marco teórico (YOLO)
  - [ ] Metodología
  - [ ] Resultados experimentales
  - [ ] Discusión
  - [ ] Conclusiones
  - [ ] Trabajo futuro
  - [ ] Referencias
- [ ] Incluir todas las figuras y tablas
- [ ] Revisar formato y ortografía
- [ ] Exportar a PDF

## Fase 11: Presentación 🎤
- [ ] Preparar slides de presentación
- [ ] Incluir demo en vivo
- [ ] Preparar respuestas a preguntas frecuentes
- [ ] Practicar presentación
- [ ] Subir materiales al repositorio

## Notas Importantes 📌

### Métricas Mínimas Esperadas
- Precision: > 0.7
- Recall: > 0.7
- mAP@0.5: > 0.5

### Tiempo Estimado por Fase
- Fase 1-2: 1 semana
- Fase 3-4: 2 semanas
- Fase 5-6: 2-3 semanas
- Fase 7-8: 1 semana
- Fase 9-11: 1-2 semanas

### Recordatorios
- Hacer commits frecuentes con mensajes descriptivos
- Documentar cada experimento
- Guardar todos los resultados
- Hacer backup del dataset y modelos entrenados
- Pedir feedback temprano al profesor/tutor

## Recursos Adicionales 🔗
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Universe](https://universe.roboflow.com/)
- [Papers with Code - Object Detection](https://paperswithcode.com/task/object-detection)
- [TensorBoard Tutorial](https://www.tensorflow.org/tensorboard/get_started)

---
**Última actualización:** Diciembre 2024
