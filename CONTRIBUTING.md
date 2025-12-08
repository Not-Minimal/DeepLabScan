# Guía de Contribución

¡Gracias por tu interés en contribuir a DeepLabScan!

## Cómo Contribuir

### Reportar Bugs

Si encuentras un bug, por favor abre un issue con:
- Descripción clara del problema
- Pasos para reproducirlo
- Comportamiento esperado vs actual
- Versiones de Python, PyTorch, y otras dependencias
- Screenshots si es aplicable

### Sugerir Mejoras

Para sugerir nuevas características:
1. Verifica que no exista ya un issue similar
2. Abre un nuevo issue describiendo la mejora
3. Explica por qué sería útil
4. Proporciona ejemplos de uso si es posible

### Pull Requests

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Estilo de Código

- Sigue PEP 8 para código Python
- Usa docstrings en español para todas las funciones y clases
- Añade type hints cuando sea posible
- Escribe tests para nuevas funcionalidades

### Tests

Ejecuta los tests antes de hacer un PR:

```bash
pytest tests/ -v
```

Para verificar cobertura:

```bash
pytest tests/ --cov=src --cov-report=html
```

### Documentación

- Actualiza README.md si cambias funcionalidad
- Añade ejemplos en EXAMPLES.md para nuevas features
- Documenta parámetros y valores de retorno

## Proceso de Revisión

1. Al menos un maintainer revisará tu PR
2. Se pueden solicitar cambios
3. Una vez aprobado, será merged

## Código de Conducta

- Sé respetuoso con otros contribuidores
- Acepta críticas constructivas
- Enfócate en lo mejor para el proyecto

## Preguntas

Si tienes preguntas, abre un issue con la etiqueta "question".

¡Gracias por contribuir! 🎉
