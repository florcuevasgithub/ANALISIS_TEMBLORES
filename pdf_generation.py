# pdf_generation.py
from fpdf import FPDF
from datetime import datetime, timedelta
import unicodedata
import os
import io
from io import BytesIO

def limpiar_texto_para_pdf(texto):
    """Normaliza y limpia texto para asegurar compatibilidad con PDF."""
    if texto is None: # Añadido para manejar None
        return ""
    return unicodedata.normalize("NFKD", str(texto)).encode("ASCII", "ignore").decode("ASCII")

class PDF(FPDF):
    def header(self):
        # Header (optional, can be customized)
        # self.set_font('Arial', 'B', 15)
        # self.cell(0, 10, 'Informe de Análisis de Temblor', 0, 1, 'C')
        pass

    def footer(self):
        # Page footer
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', 0, 0, 'C')

def generar_pdf(datos_paciente_dict, df_resultados, nombre_archivo="informe_temblor.pdf",
                diagnostico="", figs=None, comparison_mode=False,
                config1_params=None, config2_params=None, prediction_info=None): # Añadido prediction_info

    fecha_hora = (datetime.now() - timedelta(hours=3)).strftime("%d/%m/%Y %H:%M")
    
    # Initialize PDF with alias for total pages
    pdf = PDF() # Usamos la clase PDF personalizada para el pie de página
    pdf.alias_nb_pages() 
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15) # Asegura saltos de página automáticos
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Informe de Análisis de Temblor", ln=True, align='C')

    if comparison_mode:
        pdf.cell(0, 10, "Comparativo de Mediciones", ln=True, align='C')
    elif prediction_info:
        pdf.cell(0, 10, "Informe de Predicción de Temblor", ln=True, align='C')

    pdf.set_font("Arial", size=10) # Fuente un poco más pequeña para el texto general
    pdf.ln(5) # Espacio reducido
    pdf.cell(0, 7, f"Fecha y hora del análisis: {fecha_hora}", ln=True) # Altura de celda reducida

    # Helper para imprimir campos solo si tienen valor
    def _imprimir_campo_pdf(pdf_obj, etiqueta, valor, unit=""):
        cleaned_valor = limpiar_texto_para_pdf(valor)
        if cleaned_valor and cleaned_valor.lower() != "no especificado":
            pdf_obj.cell(0, 7, f"{etiqueta}: {cleaned_valor}{unit}", ln=True) # Altura de celda reducida

    pdf.set_font("Arial", 'B', 12) # Títulos de sección un poco más pequeños
    pdf.cell(0, 10, "Datos del Paciente", ln=True)
    pdf.set_font("Arial", size=10) # Contenido de datos del paciente

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

    pdf.ln(3) # Espacio reducido

    # Parámetros que queremos mostrar en la sección de configuración
    parametros_estimulacion_keys = [
        "Mano", "Dedo", "ECP", "GPI", "NST", "Polaridad",
        "Duracion", "Pulso", "Corriente", "Voltaje", "Frecuencia"
    ]
    parametros_estimulacion_units = { # Unidades para display
        "Mano": "", "Dedo": "", 
        "Duracion": " ms", "Pulso": " µs", "Corriente": " mA",
        "Voltaje": " V", "Frecuencia": " Hz"
    }

    # Function to print config/stimulation parameters
    def _print_config_section(pdf_obj, params_dict, title):
        # Check if there's any non-empty parameter to display in this section
        has_params_to_display = False
        for param_key in parametros_estimulacion_keys:
            if params_dict and params_dict.get(param_key) is not None and str(params_dict.get(param_key)).strip() != "":
                has_params_to_display = True
                break
        
        if has_params_to_display:
            pdf_obj.set_font("Arial", 'B', 12)
            pdf_obj.cell(0, 10, limpiar_texto_para_pdf(title), ln=True)
            pdf_obj.set_font("Arial", size=10) # Contenido de configuración
            for param_key in parametros_estimulacion_keys:
                value = params_dict.get(param_key)
                unit = parametros_estimulacion_units.get(param_key, "")
                _imprimir_campo_pdf(pdf_obj, param_key, value, unit)
            pdf_obj.ln(3)

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
            pdf_obj.set_font("Arial", size=10)
            pdf_obj.cell(0, 7, f"{title}: No hay resultados disponibles.", ln=True)
            pdf_obj.ln(3)
            return

        pdf_obj.set_font("Arial", 'B', 12)
        pdf_obj.cell(0, 10, limpiar_texto_para_pdf(title), ln=True)
        pdf_obj.set_font("Arial", 'B', 10) # Encabezados de tabla

        headers = ["Test", "Frecuencia (Hz)", "RMS", "Amplitud (cm)"]
        col_widths = [30, 40, 30, 50]

        # Ajuste para el ancho de las celdas de encabezado para que se ajusten
        current_x = pdf_obj.get_x()
        for i, header in enumerate(headers):
            pdf_obj.cell(col_widths[i], 7, limpiar_texto_para_pdf(header), 1, 0, 'C') # Centrado
        pdf_obj.ln(7) # Salto de línea con altura de celda reducida
        
        pdf_obj.set_font("Arial", "", 9) # Contenido de tabla

        for _, row in df_res.iterrows():
            pdf_obj.cell(col_widths[0], 6, limpiar_texto_para_pdf(row['Test']), 1)
            pdf_obj.cell(col_widths[1], 6, f"{row['Frecuencia Dominante (Hz)']:.2f}", 1, 0, 'C')
            pdf_obj.cell(col_widths[2], 6, f"{row['RMS (m/s2)']:.4f}", 1, 0, 'C')
            pdf_obj.cell(col_widths[3], 6, f"{row['Amplitud Temblor (cm)']:.2f}", 1, 0, 'C')
            pdf_obj.ln(6)
        pdf_obj.ln(3)

    if comparison_mode:
        # Asegurarse de que los DataFrames no estén vacíos antes de filtrar
        if not df_resultados.empty:
            _print_results_table(pdf, df_resultados[df_resultados['Measurement'] == 1].drop(columns='Measurement', errors='ignore'), "Resultados Medición 1")
            _print_results_table(pdf, df_resultados[df_resultados['Measurement'] == 2].drop(columns='Measurement', errors='ignore'), "Resultados Medición 2")
        else:
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 7, "No hay resultados para la comparación de mediciones.", ln=True)
            pdf.ln(3)
    elif prediction_info:
        _print_results_table(pdf, df_resultados, "Métricas de Temblor Calculadas")
    else:
        _print_results_table(pdf, df_resultados)

    # Prediction Info Section
    if prediction_info:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Resultados de la Predicción del Modelo", ln=True)
        pdf.set_font("Arial", size=10)
        _imprimir_campo_pdf(pdf, "Predicción del Modelo", prediction_info["prediction"])
        
        if prediction_info["probabilities"]:
            pdf.ln(2)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 7, "Probabilidades por Clase:", ln=True)
            pdf.set_font("Arial", size=9)
            for label, prob in prediction_info["probabilities"].items():
                pdf.cell(0, 6, f"- {limpiar_texto_para_pdf(label)}: {prob:.2f}%", ln=True)
        pdf.ln(3)


    # Clinical interpretation / Conclusion
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    
    if prediction_info:
        pdf.cell(0, 10, "Diagnóstico / Predicción:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 6, limpiar_texto_para_pdf(diagnostico))
    elif comparison_mode:
        pdf.cell(0, 10, "Conclusión del Análisis Comparativo:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 6, limpiar_texto_para_pdf(diagnostico))
    else:
        pdf.cell(0, 10, "Interpretación clínica:", ln=True)
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
        pdf.multi_cell(0, 6, limpiar_texto_para_pdf(texto_original))

    # Handle figures (now `figs` can be a list)
    if figs:
        # Ensure figs is always a list for iteration
        if not isinstance(figs, list):
            figs = [figs]
            
        for i, current_fig in enumerate(figs):
            if current_fig is not None:
                # Add a new page for each figure for better spacing, or check if there's enough space
                # A 180mm width image needs about 100-110mm height with standard aspect ratio for matplotlib
                # We'll assume a max height of 100mm for the plot itself (plus padding)
                required_height = 100 + 10 # Image height + some padding
                
                if pdf.get_y() + required_height > pdf.h - pdf.b_margin - 5: # Check if enough space is available, add a small buffer
                     pdf.add_page()
                else: 
                    pdf.ln(5) # Just a little space if it fits on the same page

                try:
                    img_buffer = BytesIO()
                    current_fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300) # Increased DPI for quality
                    img_buffer.seek(0)

                    # Calculate width to fit page, maintaining aspect ratio. 
                    # Page width is 210mm. Margins are 10mm each side (default). Usable width 190mm.
                    # Set image width to 180mm. Let FPDF calculate height.
                    pdf.image(img_buffer, x=15, w=180) 
                    img_buffer.close()
                except Exception as e:
                    print(f"Error al añadir la imagen {i+1} al PDF: {e}") # For debugging
                    # Optionally add a text placeholder in the PDF if image fails
                    pdf.set_font("Arial", 'I', 10)
                    pdf.cell(0, 10, f"Error al cargar gráfico {i+1}", ln=True, align='C')


    pdf.output(nombre_archivo)
