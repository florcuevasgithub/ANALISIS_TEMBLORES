# pdf_generation.py
from fpdf import FPDF
from datetime import datetime, timedelta
import unicodedata
import os
import io
from io import BytesIO

def limpiar_texto_para_pdf(texto):
    """Normaliza y limpia texto para asegurar compatibilidad con PDF."""
    return unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("ASCII")

def generar_pdf(datos_paciente_dict, df_resultados, nombre_archivo="informe_temblor.pdf",
                diagnostico="", figs=None, comparison_mode=False,
                config1_params=None, config2_params=None, prediction_info=None): # Añadido prediction_info

    fecha_hora = (datetime.now() - timedelta(hours=3)).strftime("%d/%m/%Y %H:%M")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Informe de Análisis de Temblor", ln=True, align='C')

    if comparison_mode:
        pdf.cell(200, 10, "Comparativo de Mediciones", ln=True, align='C')
    elif prediction_info: # Si es modo predicción
        pdf.cell(200, 10, "Informe de Predicción de Temblor", ln=True, align='C')


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

    edad_val = datos_paciente_dict.get("edad")
    edad_str_to_print = None
    try:
        if edad_val is not None and str(edad_val).strip() != "":
            edad_int = int(float(edad_val))
            edad_str_to_print = str(edad_int)
    except (ValueError, TypeError):
        pass
    _imprimir_campo_pdf(pdf, "Edad", edad_str_to_print)

    _imprimir_campo_pdf(pdf, "Sexo", datos_paciente_dict.get("sexo"))
    _imprimir_campo_pdf(pdf, "Diagnóstico", datos_paciente_dict.get("Diagnostico"))
    _imprimir_campo_pdf(pdf, "Tipo", datos_paciente_dict.get("Tipo"))
    _imprimir_campo_pdf(pdf, "Antecedente", datos_paciente_dict.get("Antecedente"))
    _imprimir_campo_pdf(pdf, "Medicacion", datos_paciente_dict.get("Medicacion"))

    pdf.ln(5)

    parametros_estimulacion_units = {
        "Mano": "", "Dedo": "", # Ahora Mano y Dedo siempre se imprimen si están en los diccionarios de config
        "ECP": "", "GPI": "", "NST": "", "Polaridad": "",
        "Duracion": " ms", "Pulso": " µs", "Corriente": " mA",
        "Voltaje": " V", "Frecuencia": " Hz"
    }

    # Function to print config/stimulation parameters
    def _print_config_section(pdf_obj, params_dict, title):
        # Only print title if there are actual parameters to display in this section
        has_params_to_display = False
        for param_key in parametros_estimulacion_units.keys():
            if params_dict.get(param_key) is not None and str(params_dict.get(param_key)).strip() != "":
                has_params_to_display = True
                break
        
        if has_params_to_display:
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
    elif prediction_info: # Para la predicción, también mostrar la configuración de la medición
        _print_config_section(pdf, datos_paciente_dict, "Configuración de la Medición")
    else: # Modo de medición única
        _print_config_section(pdf, datos_paciente_dict, "Configuración de la Medición")


    # Table for results
    def _print_results_table(pdf_obj, df_res, title="Resultados del Análisis"):
        if df_res.empty:
            pdf_obj.set_font("Arial", size=12)
            pdf_obj.cell(0, 10, f"{title}: No hay resultados disponibles.", ln=True)
            pdf_obj.ln(5)
            return

        pdf_obj.set_font("Arial", 'B', 14)
        pdf_obj.cell(0, 10, title, ln=True)
        pdf_obj.set_font("Arial", 'B', 12)

        headers = ["Test", "Frecuencia (Hz)", "RMS", "Amplitud (cm)"]
        col_widths = [30, 40, 30, 50] # Adjusted width for Amplitud

        for i, header in enumerate(headers):
            pdf_obj.cell(col_widths[i], 10, header, 1)
        pdf_obj.ln(10)
        pdf_obj.set_font("Arial", "", 10)

        for _, row in df_res.iterrows():
            pdf_obj.cell(col_widths[0], 10, limpiar_texto_para_pdf(row['Test']), 1)
            pdf_obj.cell(col_widths[1], 10, f"{row['Frecuencia Dominante (Hz)']:.2f}", 1)
            pdf_obj.cell(col_widths[2], 10, f"{row['RMS (m/s2)']:.4f}", 1)
            pdf_obj.cell(col_widths[3], 10, f"{row['Amplitud Temblor (cm)']:.2f}", 1)
            pdf_obj.ln(10)
        pdf_obj.ln(5)

    if comparison_mode:
        _print_results_table(pdf, df_resultados[df_resultados['Measurement'] == 1].drop(columns='Measurement', errors='ignore'), "Resultados Medición 1")
        _print_results_table(pdf, df_resultados[df_resultados['Measurement'] == 2].drop(columns='Measurement', errors='ignore'), "Resultados Medición 2")
    elif prediction_info: # Para la predicción, usar los resultados calculados
        _print_results_table(pdf, df_resultados, "Métricas de Temblor Calculadas")
    else:
        _print_results_table(pdf, df_resultados)

    # Prediction Info Section
    if prediction_info:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Resultados de la Predicción del Modelo", ln=True)
        pdf.set_font("Arial", size=12)
        _imprimir_campo_pdf(pdf, "Predicción del Modelo", prediction_info["prediction"])
        
        if prediction_info["probabilities"]:
            pdf.ln(2)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Probabilidades por Clase:", ln=True)
            pdf.set_font("Arial", size=10)
            for label, prob in prediction_info["probabilities"].items():
                pdf.cell(0, 7, f"- {label}: {prob:.2f}%", ln=True)
        pdf.ln(5)


    # Clinical interpretation / Conclusion
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    
    if prediction_info: # Si es modo predicción, el diagnóstico es la predicción misma
        pdf.cell(200, 10, "Diagnóstico / Predicción:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 8, limpiar_texto_para_pdf(diagnostico)) # diagnostico aquí es el texto de la predicción
    elif comparison_mode:
        pdf.cell(200, 10, "Conclusión del Análisis Comparativo:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 8, limpiar_texto_para_pdf(diagnostico)) # In comparison, 'diagnostico' is the conclusion
    else:
        pdf.cell(200, 10, "Interpretación clínica:", ln=True)
        pdf.set_font("Arial", size=10)
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

    # Handle figures (now `figs` can be a list)
    if figs:
        # Ensure figs is always a list for iteration
        if not isinstance(figs, list):
            figs = [figs]
            
        for i, current_fig in enumerate(figs):
            if current_fig is not None:
                # Add a new page for each figure if there's more content or to keep them separate
                if i > 0 and (pdf.get_y() + 80 > pdf.h - 20): # Check if enough space or if not first figure
                     pdf.add_page()
                else: # Add some space before the first image
                    pdf.ln(10) 

                try:
                    # Save figure to a BytesIO object instead of a temporary file for in-memory handling
                    img_buffer = BytesIO()
                    current_fig.savefig(img_buffer, format='png', bbox_inches='tight')
                    img_buffer.seek(0) # Rewind the buffer to the beginning

                    pdf.image(img_buffer, x=15, w=180) # Adjust w as needed
                    img_buffer.close()
                except Exception as e:
                    print(f"Error al añadir la imagen al PDF: {e}") # For debugging

    pdf.output(nombre_archivo)
