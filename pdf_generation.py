# pdf_generation.py

from fpdf import FPDF
from datetime import datetime, timedelta
import numpy as np
import unicodedata
import os
import matplotlib.pyplot as plt # Importar para poder guardar figuras temporalmente
import tempfile # Para archivos temporales al guardar figuras

def _clean_text_for_pdf(text):
    """Normaliza y limpia texto para asegurar que se renderice correctamente en PDF."""
    if text is None:
        return ""
    return unicodedata.normalize("NFKD", str(text)).encode("ASCII", "ignore").decode("ASCII")

def _print_field_to_pdf(pdf_obj, label, value, unit=""):
    """Helper para imprimir un campo en el PDF solo si tiene un valor válido."""
    if value is not None and str(value).strip() != "" and str(value).lower() != "no especificado" and str(value).lower() != "nan":
        pdf_obj.cell(200, 10, f"{label}: {_clean_text_for_pdf(value)}{unit}", ln=True)

def _print_params_and_config(pdf_obj, params_dict, title):
    """Helper para imprimir parámetros de configuración/estimulación en el PDF."""
    pdf_obj.set_font("Arial", 'B', 12)
    pdf_obj.cell(0, 10, title, ln=True)
    pdf_obj.set_font("Arial", size=10)

    # Definir los parámetros de configuración y sus unidades para imprimir
    # Aquí se incluyen Mano y Dedo, asumimos que están en params_dict
    params_to_print_with_unit = {
        "Mano": "", "Dedo": "",
        "ECP": "", "GPI": "", "NST": "", "Polaridad": "",
        "Duracion": " ms", "Pulso": " µs", "Corriente": " mA",
        "Voltaje": " V", "Frecuencia": " Hz"
    }

    for param_key, unit in params_to_print_with_unit.items():
        value = params_dict.get(param_key)
        _print_field_to_pdf(pdf_obj, param_key, value, unit)
    pdf_obj.ln(5)

def _print_results_table(pdf_obj, df_res, title):
    """Helper para imprimir una tabla de resultados de temblor en el PDF."""
    pdf_obj.set_font("Arial", 'B', 14)
    pdf_obj.cell(0, 10, title, ln=True)
    pdf_obj.set_font("Arial", 'B', 12)
    pdf_obj.cell(30, 10, "Test", 1)
    pdf_obj.cell(40, 10, "Frecuencia (Hz)", 1)
    pdf_obj.cell(30, 10, "RMS", 1)
    pdf_obj.cell(50, 10, "Amplitud (cm)", 1)
    pdf_obj.ln(10)
    pdf_obj.set_font("Arial", "", 10)

    for _, row in df_res.iterrows():
        pdf_obj.cell(30, 10, _clean_text_for_pdf(row['Test']), 1)
        pdf_obj.cell(40, 10, f"{row['Frecuencia Dominante (Hz)']:.2f}", 1)
        pdf_obj.cell(30, 10, f"{row['RMS (m/s2)']:.4f}", 1)
        pdf_obj.cell(50, 10, f"{row['Amplitud Temblor (cm)']:.2f}", 1)
        pdf_obj.ln(10)
    pdf_obj.ln(5)

def generate_tremor_report_pdf(patient_data_dict, results_df=None, comparison_results_df1=None, comparison_results_df2=None,
                                conclusion_text="", figures=None, filename="informe_temblor.pdf"):
    """
    Genera un informe PDF completo con datos del paciente, resultados de análisis
    y gráficos. Puede generar un informe para un solo análisis o comparativo.
    """
    fecha_hora = (datetime.now() - timedelta(hours=3)).strftime("%d/%m/%Y %H:%M")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)

    if comparison_results_df1 is not None and comparison_results_df2 is not None:
        pdf.cell(200, 10, "Informe Comparativo de Análisis de Temblor", ln=True, align='C')
    else:
        pdf.cell(200, 10, "Informe de Análisis de Temblor", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Fecha y hora del análisis: {fecha_hora}", ln=True)
    pdf.ln(5)

    # Datos del Paciente (generales)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Datos del Paciente", ln=True)
    pdf.set_font("Arial", size=12)
    _print_field_to_pdf(pdf, "Nombre", patient_data_dict.get("Nombre"))
    _print_field_to_pdf(pdf, "Apellido", patient_data_dict.get("Apellido"))

    edad_val = patient_data_dict.get("Edad")
    edad_str_to_print = None
    try:
        if pd.notna(edad_val):
            edad_int = int(float(edad_val))
            edad_str_to_print = str(edad_int)
    except ValueError:
        pass
    _print_field_to_pdf(pdf, "Edad", edad_str_to_print)

    _print_field_to_pdf(pdf, "Sexo", patient_data_dict.get("Sexo"))
    _print_field_to_pdf(pdf, "Diagnóstico", patient_data_dict.get("Diagnostico"))
    _print_field_to_pdf(pdf, "Tipo", patient_data_dict.get("Tipo"))
    _print_field_to_pdf(pdf, "Antecedente", patient_data_dict.get("Antecedente"))
    _print_field_to_pdf(pdf, "Medicacion", patient_data_dict.get("Medicacion"))
    pdf.ln(5)

    # Configuración de mediciones (para comparativa o para singular)
    if comparison_results_df1 is not None and comparison_results_df2 is not None:
        _print_params_and_config(pdf, patient_data_dict, "Configuración Medición 1")
        # Asumimos que patient_data_dict contiene los datos para la medición 1
        # Si la medición 2 tiene parámetros diferentes, deberían pasarse también
        # Por simplicidad, aquí se usa el mismo patient_data_dict para la config 2,
        # pero esto podría necesitar una mejora si las configs son REALMENTE distintas.
        # Por ahora, los datos de la Configuración 2 vendrían de df_config2_meta si se extraen
        # y se pasan como un segundo diccionario de datos del paciente.
        # Por simplicidad en este refactoring, solo imprime del primer patient_data_dict
        # si no se pasa explícitamente otro para la medición 2.
    else: # Single measurement analysis
        _print_params_and_config(pdf, patient_data_dict, "Configuración de la Medición")
    pdf.ln(5)

    # Tablas de Resultados
    if results_df is not None:
        _print_results_table(pdf, results_df, "Resultados del Análisis de Temblor")

    if comparison_results_df1 is not None:
        _print_results_table(pdf, comparison_results_df1, "Resultados Medición 1")
    if comparison_results_df2 is not None:
        _print_results_table(pdf, comparison_results_df2, "Resultados Medición 2")

    # Interpretación o Conclusión
    if results_df is not None: # For single analysis
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "Interpretación clínica:", ln=True)
        pdf.set_font("Arial", size=10)
        # Este texto es fijo en tu código original, podrías hacerlo dinámico o pasarlo como parámetro
        static_clinical_text = """
        Este informe analiza tres tipos de temblores: en reposo, postural y de acción.

        Los valores de referencia considerados son:
          Para las frecuencias (Hz):
        - Temblor Parkinsoniano: 3-6 Hz en reposo.
        - Temblor Esencial: 8-10 Hz en acción o postura.

          Para las amplitudes:
        - Mayores a 0.5 cm pueden ser clínicamente relevantes.

          Para el RMS (m/s2):
        - Normal/sano: menor a 0.5 m/s2.
        - PK leve: entre 0.5 y 1.5 m/s2.
        - TE o PK severo: mayor a 2 m/s2.

        Nota clínica: Los valores de referencia presentados a continuación se basan en literatura científica.
        """
        pdf.multi_cell(0, 8, _clean_text_for_pdf(static_clinical_text))
        pdf.set_font("Arial", 'B', 12)

    if conclusion_text: # For comparative analysis or specific prediction text
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Conclusión", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 10, _clean_text_for_pdf(conclusion_text))

    # Añadir figuras
    if figures:
        for fig_item in figures:
            if fig_item is not None:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    fig_item.savefig(tmpfile.name, format='png', bbox_inches='tight')
                    pdf.image(tmpfile.name, x=15, w=180)
                    os.remove(tmpfile.name)
                pdf.ln(10)
                plt.close(fig_item) # Importante para liberar memoria

    # Output PDF as bytes
    pdf_output = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)
    return pdf_output
