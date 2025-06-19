# pdf_generation.py
from fpdf import FPDF
from datetime import datetime, timedelta
import unicodedata
import os
import io
from io import BytesIO # Needed for BytesIO if used

def limpiar_texto_para_pdf(texto):
    """Normaliza y limpia texto para asegurar compatibilidad con PDF."""
    return unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("ASCII")

def generar_pdf(datos_paciente_dict, df_resultados, nombre_archivo="informe_temblor.pdf", diagnostico="", fig=None, comparison_mode=False, config1_params=None, config2_params=None):
    """
    Genera un informe PDF con los resultados del análisis de temblor.
    Acepta un diccionario de datos del paciente y un DataFrame de resultados.
    Puede generar informes para una sola medición o comparativos.
    """
    fecha_hora = (datetime.now() - timedelta(hours=3)).strftime("%d/%m/%Y %H:%M")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Informe de Análisis de Temblor", ln=True, align='C')
    
    if comparison_mode:
        pdf.cell(200, 10, "Comparativo de Mediciones", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Fecha y hora del análisis: {fecha_hora}", ln=True)

    # Helper para imprimir campos solo si tienen valor
    def _imprimir_campo_pdf(pdf_obj, etiqueta, valor, unit=""):
        if valor is not None and str(valor).strip() != "" and str(valor).lower() != "no especificado":
            pdf_obj.cell(200, 10, f"{etiqueta}: {valor}{unit}", ln=True)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Datos del Paciente", ln=True)
    pdf.set_font("Arial", size=12)

    _imprimir_campo_pdf(pdf, "Nombre", datos_paciente_dict.get("Nombre"))
    _imprimir_campo_pdf(pdf, "Apellido", datos_paciente_dict.get("Apellido"))
    
    edad_val = datos_paciente_dict.get("edad") # Use 'edad' from the dict
    edad_str_to_print = None
    try:
        if edad_val is not None and str(edad_val).strip() != "":
            edad_int = int(float(edad_val))
            edad_str_to_print = str(edad_int)
    except (ValueError, TypeError):
        pass
    _imprimir_campo_pdf(pdf, "Edad", edad_str_to_print)
    
    _imprimir_campo_pdf(pdf, "Sexo", datos_paciente_dict.get("sexo")) # Use 'sexo' from the dict
    _imprimir_campo_pdf(pdf, "Diagnóstico", datos_paciente_dict.get("Diagnostico"))
    _imprimir_campo_pdf(pdf, "Tipo", datos_paciente_dict.get("Tipo"))
    _imprimir_campo_pdf(pdf, "Antecedente", datos_paciente_dict.get("Antecedente"))
    _imprimir_campo_pdf(pdf, "Medicacion", datos_paciente_dict.get("Medicacion"))
    
    # In comparative mode, Mano and Dedo are part of config params, not patient info
    if not comparison_mode:
        _imprimir_campo_pdf(pdf, "Mano", datos_paciente_dict.get("mano_medida"))
        _imprimir_campo_pdf(pdf, "Dedo", datos_paciente_dict.get("dedo_medido"))
    
    pdf.ln(5)

    parametros_estimulacion_units = {
        "Mano": "", "Dedo": "", # Included here for config sections
        "ECP": "", "GPI": "", "NST": "", "Polaridad": "",
        "Duracion": " ms", "Pulso": " µs", "Corriente": " mA",
        "Voltaje": " V", "Frecuencia": " Hz"
    }

    # Function to print config/stimulation parameters
    def _print_config_section(pdf_obj, params_dict, title):
        if any(params_dict.get(k) is not None and str(params_dict.get(k)).strip() != "" for k in parametros_estimulacion_units.keys()):
            pdf_obj.set_font("Arial", 'B', 14)
            pdf_obj.cell(0, 10, title, ln=True)
            pdf_obj.set_font("Arial", size=12)
            for param_key, unit in parametros_estimulacion_units.items():
                _imprimir_campo_pdf(pdf_obj, param_key, params_dict.get(param_key), unit)
            pdf_obj.ln(5)

    if comparison_mode:
        if config1_params:
            _print_config_section(pdf, config1_params, "Configuración Medición 1")
        if config2_params:
            _print_config_section(pdf, config2_params, "Configuración Medición 2")
    else: # Single measurement mode
        _print_config_section(pdf, datos_paciente_dict, "Configuración")

    # Table for results
    def _print_results_table(pdf_obj, df_res, title="Resultados del Análisis"):
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
            pdf_obj.cell(30, 10, row['Test'], 1)
            pdf_obj.cell(40, 10, f"{row['Frecuencia Dominante (Hz)']:.2f}", 1)
            pdf_obj.cell(30, 10, f"{row['RMS (m/s2)']:.4f}", 1)
            pdf_obj.cell(50, 10, f"{row['Amplitud Temblor (cm)']:.2f}", 1)
            pdf_obj.ln(10)
        pdf_obj.ln(5)

    if comparison_mode:
        _print_results_table(pdf, df_resultados[df_resultados['Measurement'] == 1].drop(columns='Measurement'), "Resultados Medición 1")
        _print_results_table(pdf, df_resultados[df_resultados['Measurement'] == 2].drop(columns='Measurement'), "Resultados Medición 2")
    else:
        _print_results_table(pdf, df_resultados)

    # Clinical interpretation / Conclusion
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Interpretación clínica:", ln=True)
    pdf.set_font("Arial", size=10)
    
    if comparison_mode:
        pdf.multi_cell(0, 8, limpiar_texto_para_pdf(diagnostico)) # In comparison, 'diagnostico' is the conclusion
    else:
        texto_original = f"""
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

        Diagnóstico Automático: {diagnostico}
        """
        pdf.multi_cell(0, 8, limpiar_texto_para_pdf(texto_original))

    if fig is not None:
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig.savefig(tmpfile.name, format='png', bbox_inches='tight')
                pdf.image(tmpfile.name, x=15, w=180)
            os.remove(tmpfile.name)
        except Exception as e:
            print(f"Error al añadir la imagen al PDF: {e}") # For debugging in local dev

    pdf.output(nombre_archivo)
