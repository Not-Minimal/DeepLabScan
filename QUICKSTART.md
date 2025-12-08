# Guía de Inicio Rápido - DeepLabScan

Esta guía te ayudará a empezar con DeepLabScan en menos de 10 minutos.

## 1. Instalación Rápida

```bash
# Clonar el repositorio
git clone https://github.com/Not-Minimal/DeepLabScan.git
cd DeepLabScan

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## 2. Configurar Roboflow

Obtén tu API key de [Roboflow](https://roboflow.com):

```bash
# Crear archivo .env
echo "ROBOFLOW_API_KEY=tu_api_key_aqui" > .env
```

## 3. Descargar Datos

```bash
python scripts/download_data.py \
    --workspace "tu_workspace" \
    --project "tu_proyecto" \
    --version 1
```

## 4. Entrenar tu Primer Modelo

```bash
python scripts/train.py \
    --data data/raw/data.yaml \
    --epochs 50 \
    --batch 16 \
    --device cpu
```

**Nota**: Usa `--device cuda` si tienes GPU NVIDIA, o `--device mps` para Apple Silicon.

## 5. Evaluar el Modelo

```bash
python scripts/evaluate.py \
    --weights runs/train/exp/weights/best.pt \
    --save-plots
```

## 6. Hacer Predicciones

```bash
python scripts/predict.py \
    --weights runs/train/exp/weights/best.pt \
    --source path/to/images/ \
    --save
```

## Próximos Pasos

- Lee [EXAMPLES.md](EXAMPLES.md) para ejemplos más detallados
- Consulta [README.md](README.md) para documentación completa
- Ajusta parámetros en [configs/config.yaml](configs/config.yaml)

## Solución Rápida de Problemas

### Error: No se encontró data.yaml
```bash
# Asegúrate de haber descargado los datos primero
python scripts/download_data.py --workspace tu_workspace --project tu_proyecto --version 1
```

### Error: Memoria insuficiente
```bash
# Reduce el tamaño del batch
python scripts/train.py --batch 8 --imgsz 416
```

### Predicciones con muchos falsos positivos
```bash
# Aumenta el confidence threshold
python scripts/predict.py --conf 0.5
```

## Comandos Útiles

```bash
# Ver información del modelo
python -c "from src.models import YOLOModel; print(YOLOModel.get_available_models())"

# Verificar instalación
python -c "import ultralytics; print(f'YOLOv8 version: {ultralytics.__version__}')"

# Ejecutar tests
pytest tests/ -v
```

## Recursos Adicionales

- [Documentación de YOLOv8](https://docs.ultralytics.com/)
- [Documentación de Roboflow](https://docs.roboflow.com/)
- [Tutorial de Computer Vision](https://www.pyimagesearch.com/)

¡Listo para comenzar! 🚀
