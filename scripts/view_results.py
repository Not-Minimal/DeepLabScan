#!/usr/bin/env python3
"""
Script para Visualizar y Analizar Resultados en Excel

Este script permite:
1. Ver resumen de todos los experimentos
2. Comparar diferentes modelos
3. Encontrar el mejor modelo según una métrica
4. Generar estadísticas de los experimentos

Uso:
    python scripts/view_results.py
    python scripts/view_results.py --summary
    python scripts/view_results.py --best-model
    python scripts/view_results.py --compare
    python scripts/view_results.py --export-summary
"""

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    from excel_logger import ExcelLogger
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Instala con: pip install pandas openpyxl")
    sys.exit(1)


def print_summary(logger, last_n=10):
    """Muestra resumen de últimos experimentos"""
    print("\n" + "=" * 100)
    print(f"RESUMEN DE ÚLTIMOS {last_n} EXPERIMENTOS")
    print("=" * 100)

    df = logger.get_summary_dataframe()

    if len(df) == 0:
        print("No hay experimentos registrados todavía.")
        return

    # Mostrar últimos n
    recent = df.tail(last_n)

    # Configurar display
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 40)

    print(recent.to_string(index=True))
    print("=" * 100)
    print(f"\nTotal de experimentos: {len(df)}")


def print_best_models(logger):
    """Muestra los mejores modelos según diferentes métricas"""
    print("\n" + "=" * 100)
    print("MEJORES MODELOS POR MÉTRICA")
    print("=" * 100)

    df = logger.get_summary_dataframe()

    if len(df) == 0:
        print("No hay experimentos registrados todavía.")
        return

    # Filtrar solo Training y Evaluation
    df_models = df[df["Tipo"].isin(["Training", "Evaluation"])].copy()

    if len(df_models) == 0:
        print("No hay modelos entrenados o evaluados todavía.")
        return

    # Convertir a numérico
    for col in ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1-Score"]:
        df_models[col] = pd.to_numeric(df_models[col], errors="coerce")

    metrics = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1-Score"]

    for metric in metrics:
        valid_data = df_models[df_models[metric].notna()]
        if len(valid_data) > 0:
            best_idx = valid_data[metric].idxmax()
            best = valid_data.loc[best_idx]

            print(f"\n🏆 Mejor {metric}: {best[metric]:.4f}")
            print(f"   - Experimento: {best['Experimento']}")
            print(f"   - Tipo: {best['Tipo']}")
            print(f"   - Modelo: {best['Modelo']}")
            print(f"   - Fecha: {best['Fecha']} {best['Hora']}")


def compare_experiments(logger, exp_names=None):
    """Compara experimentos específicos o todos"""
    print("\n" + "=" * 100)
    print("COMPARACIÓN DE EXPERIMENTOS")
    print("=" * 100)

    df = logger.get_summary_dataframe()

    if len(df) == 0:
        print("No hay experimentos registrados todavía.")
        return

    # Filtrar experimentos si se especificaron
    if exp_names:
        df = df[df["Experimento"].isin(exp_names)]
        if len(df) == 0:
            print(f"No se encontraron los experimentos: {exp_names}")
            return

    # Agrupar por tipo y calcular estadísticas
    print("\n📊 ESTADÍSTICAS POR TIPO DE EXPERIMENTO:\n")

    for exp_type in df["Tipo"].unique():
        df_type = df[df["Tipo"] == exp_type]

        print(f"\n{exp_type}:")
        print(f"  Total: {len(df_type)}")

        # Calcular estadísticas para métricas numéricas
        numeric_cols = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1-Score"]

        for col in numeric_cols:
            values = pd.to_numeric(df_type[col], errors="coerce").dropna()
            if len(values) > 0:
                print(f"  {col}:")
                print(f"    - Media: {values.mean():.4f}")
                print(f"    - Mediana: {values.median():.4f}")
                print(f"    - Min: {values.min():.4f}")
                print(f"    - Max: {values.max():.4f}")
                print(f"    - Std: {values.std():.4f}")


def export_summary(logger, output_path="results/summary_report.csv"):
    """Exporta resumen a CSV"""
    df = logger.get_summary_dataframe()

    if len(df) == 0:
        print("No hay experimentos registrados todavía.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\n✓ Resumen exportado a: {output_path}")


def show_training_details(logger):
    """Muestra detalles de entrenamientos"""
    print("\n" + "=" * 100)
    print("DETALLES DE ENTRENAMIENTOS")
    print("=" * 100)

    try:
        df = pd.read_excel(logger.excel_path, sheet_name=logger.TRAIN_SHEET)

        if len(df) == 0:
            print("No hay entrenamientos registrados todavía.")
            return

        print(f"\nTotal de entrenamientos: {len(df)}\n")

        # Configurar display
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", 40)

        print(df.to_string(index=False))

        # Estadísticas
        print("\n" + "=" * 100)
        print("ESTADÍSTICAS DE ENTRENAMIENTOS")
        print("=" * 100)

        print(f"\nDuración promedio: {df['Duración (min)'].mean():.2f} minutos")
        print(f"Duración total: {df['Duración (min)'].sum():.2f} minutos")

        # Contar modelos usados
        print(f"\nModelos utilizados:")
        for modelo, count in df["Modelo"].value_counts().items():
            print(f"  - {modelo}: {count} entrenamientos")

    except Exception as e:
        print(f"❌ Error al leer entrenamientos: {e}")


def show_evaluation_details(logger):
    """Muestra detalles de evaluaciones"""
    print("\n" + "=" * 100)
    print("DETALLES DE EVALUACIONES")
    print("=" * 100)

    try:
        df = pd.read_excel(logger.excel_path, sheet_name=logger.EVAL_SHEET)

        if len(df) == 0:
            print("No hay evaluaciones registradas todavía.")
            return

        print(f"\nTotal de evaluaciones: {len(df)}\n")

        # Configurar display
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", 40)

        print(df.to_string(index=False))

    except Exception as e:
        print(f"❌ Error al leer evaluaciones: {e}")


def show_prediction_details(logger):
    """Muestra detalles de predicciones"""
    print("\n" + "=" * 100)
    print("DETALLES DE PREDICCIONES")
    print("=" * 100)

    try:
        df = pd.read_excel(logger.excel_path, sheet_name=logger.PREDICT_SHEET)

        if len(df) == 0:
            print("No hay predicciones registradas todavía.")
            return

        print(f"\nTotal de predicciones: {len(df)}\n")

        # Configurar display
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", 40)

        print(df.to_string(index=False))

        # Estadísticas
        print("\n" + "=" * 100)
        print("ESTADÍSTICAS DE PREDICCIONES")
        print("=" * 100)

        total_imgs = df["Total Imágenes"].sum()
        total_dets = df["Total Detecciones"].sum()
        avg_dets = total_dets / total_imgs if total_imgs > 0 else 0

        print(f"\nTotal de imágenes procesadas: {total_imgs}")
        print(f"Total de detecciones: {total_dets}")
        print(f"Promedio de detecciones por imagen: {avg_dets:.2f}")

    except Exception as e:
        print(f"❌ Error al leer predicciones: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualizar y analizar resultados de experimentos"
    )

    parser.add_argument(
        "--excel-path",
        type=str,
        default="results/experiment_results.xlsx",
        help="Ruta al archivo Excel con resultados",
    )

    parser.add_argument(
        "--summary", action="store_true", help="Mostrar resumen de últimos experimentos"
    )

    parser.add_argument(
        "--best-model", action="store_true", help="Mostrar mejores modelos por métrica"
    )

    parser.add_argument("--compare", action="store_true", help="Comparar experimentos")

    parser.add_argument(
        "--training", action="store_true", help="Mostrar detalles de entrenamientos"
    )

    parser.add_argument(
        "--evaluation", action="store_true", help="Mostrar detalles de evaluaciones"
    )

    parser.add_argument(
        "--prediction", action="store_true", help="Mostrar detalles de predicciones"
    )

    parser.add_argument(
        "--export", type=str, default=None, help="Exportar resumen a CSV"
    )

    parser.add_argument(
        "--last",
        type=int,
        default=10,
        help="Número de últimos experimentos a mostrar (default: 10)",
    )

    parser.add_argument(
        "--all", action="store_true", help="Mostrar toda la información disponible"
    )

    args = parser.parse_args()

    # Verificar que existe el archivo Excel
    excel_path = Path(args.excel_path)
    if not excel_path.exists():
        print(f"❌ Error: No se encontró el archivo {excel_path}")
        print("\nAsegúrate de haber ejecutado al menos un experimento:")
        print("  python scripts/train.py --data-dir data/raw --epochs 15")
        sys.exit(1)

    print("=" * 100)
    print(f"ANÁLISIS DE RESULTADOS - DeepLabScan")
    print("=" * 100)
    print(f"Archivo: {excel_path}")

    # Crear logger
    logger = ExcelLogger(str(excel_path))

    # Si no se especificó ninguna opción, mostrar todo
    if not any(
        [
            args.summary,
            args.best_model,
            args.compare,
            args.training,
            args.evaluation,
            args.prediction,
            args.export,
            args.all,
        ]
    ):
        args.all = True

    # Ejecutar acciones
    if args.all or args.summary:
        print_summary(logger, last_n=args.last)

    if args.all or args.best_model:
        print_best_models(logger)

    if args.all or args.compare:
        compare_experiments(logger)

    if args.all or args.training:
        show_training_details(logger)

    if args.all or args.evaluation:
        show_evaluation_details(logger)

    if args.all or args.prediction:
        show_prediction_details(logger)

    if args.export:
        export_summary(logger, args.export)

    print("\n" + "=" * 100)
    print("✓ ANÁLISIS COMPLETADO")
    print("=" * 100)
    print(f"\n📊 Para abrir el Excel: {excel_path}")
    print(f"💡 Usa las diferentes hojas para ver detalles por tipo de experimento")


if __name__ == "__main__":
    main()
