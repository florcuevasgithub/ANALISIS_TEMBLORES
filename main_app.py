# main_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io

# Importar constantes y funciones de nuestros módulos
from config import FS, VENTANA_DURACION_SEG, SOLAPAMIENTO_VENTANA, EXPECTED_FEATURES_FOR_MODEL, SENSOR_COLS
from data_processing import load_csv, extract_patient_data, clean_sensor_data
from signal_analysis import analyze_tremor_windows
from pdf_generation import generate_tremor_report_pdf
from ml_model import load_prediction_model, make_tremor_prediction

# --- CONFIGURACIÓN GENERAL DE STREAMLIT ---
st.set_page_config(layout="wide", page_title="Análisis y Predicción de Temblor", page_icon="📈")

# Inicializar una variable en el estado de sesión para controlar el reinicio
if "reiniciar" not in st.session_state:
    st.session_state.reiniciar = False

# Función para manejar el reinicio de la aplicación
def manejar_reinicio():
    if st.session_state.get("reiniciar", False):
        # Eliminar archivos temporales si es necesario (ej. PDFs generados)
        # (Esto puede ser mejorado para buscar archivos específicos si no se eliminan al final de la descarga)
        # Por ahora, solo resetea el estado de la sesión
        pass # La generación de PDF ahora maneja archivos temporales de figura de forma local

    st.session_state.clear()
    st.experimental_rerun()

# --- Estilos CSS generales (otros estilos específicos de uploader se mueven a sus secciones) ---
st.markdown("""
    <style>
    .prueba-titulo {
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


# --- Funciones específicas de la aplicación (diagnóstico rule-based) ---
def diagnosticar_rule_based(df_results_table):
    """
    Realiza un diagnóstico basado en reglas simples de frecuencia y amplitud.
    Recibe un DataFrame con los resultados promedio de los tests.
    """
    def get_max_amplitude(test_name):
        # Filtra el DataFrame para el test_name específico y obtiene la Amplitud Temblor (cm)
        fila = df_results_table[df_results_table['Test'] == test_name]
        return fila['Amplitud Temblor (cm)'].max() if not fila.empty else 0

    def get_mean_frequency(test_name):
        # Filtra el DataFrame para el test_name específico y obtiene la Frecuencia Dominante (Hz)
        fila = df_results_table[df_results_table['Test'] == test_name]
        return fila['Frecuencia Dominante (Hz)'].mean() if not fila.empty else 0

    # Obtener métricas para cada test
    amp_reposo = get_max_amplitude('Reposo')
    freq_reposo = get_mean_frequency('Reposo')
    amp_postural = get_max_amplitude('Postural')
    freq_postural = get_mean_frequency('Postural')
    amp_accion = get_max_amplitude('Acción')
    freq_accion = get_mean_frequency('Acción')

    # Reglas de diagnóstico
    # Parkinson: temblor en reposo > 0.3 cm y frecuencia entre 3-6.5 Hz
    if amp_reposo > 0.3 and 3 <= freq_reposo <= 6.5:
        return "Probable Parkinson"
    # Temblor Esencial: temblor postural/acción > 0.3 cm y frecuencia entre 7.5-12 Hz
    elif (amp_postural > 0.3 or amp_accion > 0.3) and \
         (7.5 <= freq_postural <= 12 or 7.5 <= freq_accion <= 12):
        return "Probable Temblor Esencial"
    else:
        return "Temblor dentro de parámetros normales"


# ------------------ MODO PRINCIPAL DE LA APLICACIÓN --------------------

st.title("🧠 Análisis de Temblor")
opcion = st.sidebar.radio("Selecciona una opción:", ["1️⃣ Análisis de una medición", "2️⃣ Comparar dos mediciones", "3️⃣ Predicción de Temblor"])

if st.sidebar.button("🔄 Nuevo análisis"):
    manejar_reinicio()

# --- Opción 1: Análisis de una medición ---
if opcion == "1️⃣ Análisis de una medición":
    st.title("📈 Análisis de una medición")

    st.markdown('<div class="prueba-titulo">Subir archivo CSV para prueba en REPOSO</div>', unsafe_allow_html=True)
    reposo_file = st.file_uploader("", type=["csv"], key="reposo")

    st.markdown('<div class="prueba-titulo">Subir archivo CSV para prueba POSTURAL</div>', unsafe_allow_html=True)
    postural_file = st.file_uploader("", type=["csv"], key="postural")

    st.markdown('<div class="prueba-titulo">Subir archivo CSV para prueba en ACCIÓN</div>', unsafe_allow_html=True)
    accion_file = st.file_uploader("", type=["csv"], key="accion")

    # Estilos CSS personalizados para los uploaders en esta sección
    st.markdown("""
        <style>
        div[data-testid="stFileUploaderDropzoneInstructions"] span {
            display: none !important;
        }
        div[data-testid="stFileUploaderDropzoneInstructions"]::before {
            content: "Arrastrar archivo aquí";
            font-weight: bold;
            font-size: 16px;
            color: #444;
            display: block;
            margin-bottom: 0.5rem;
        }
        div[data-testid="stFileUploader"] button[kind="secondary"] {
            visibility: hidden;
        }
        div[data-testid="stFileUploader"] button[kind="secondary"]::before {
            float: right;
            margin-right: 0;
            content: "Cargar archivos";
            visibility: visible;
            display: inline-block;
            background-color: #FF5722;
            color: white;
            padding: 0.5em 1em;
            border-radius: 6px;
            border: 2px solid white;
            cursor: pointer;
        }
        /* Alinea todo a la derecha */
        div[data-testid="stFileUploader"] > div:first-child {
            display: flex;
            justify-content: flex-end;
            align-items: center;
        }
        div[data-testid="stFileUploader"] > div {
            display: flex;
            justify-content: flex-end;
            align-items: center;
        }
        </style>
    """, unsafe_allow_html=True)


    uploaded_files = {
        "Reposo": reposo_file,
        "Postural": postural_file,
        "Acción": accion_file,
    }

    if st.button("Iniciar análisis"):
        mediciones_tests_raw = {}
        for test, file_obj in uploaded_files.items():
            if file_obj is not None:
                mediciones_tests_raw[test] = load_csv(file_obj)

        if not mediciones_tests_raw:
            st.warning("Por favor, sube al menos un archivo para iniciar el análisis.")
        else:
            # Extraer datos del paciente del primer archivo válido
            first_df_raw = None
            for df_raw_val in mediciones_tests_raw.values():
                if not df_raw_val.empty:
                    first_df_raw = df_raw_val
                    break
            
            if first_df_raw is not None:
                patient_data_for_report = extract_patient_data(first_df_raw)
            else:
                patient_data_for_report = {} # Empty dict if no valid file

            results_single_analysis = []
            figures_for_report = []
            min_ventanas_count = float('inf')
            temp_window_dfs = [] # Para almacenar df_ventanas temporales para el gráfico

            for test_type, raw_df in mediciones_tests_raw.items():
                if not raw_df.empty:
                    st.info(f"Procesando {test_type}...")
                    # Limpieza y análisis de la señal
                    cleaned_df = clean_sensor_data(raw_df)
                    if cleaned_df.empty:
                        st.warning(f"No hay datos de sensor válidos en el archivo de {test_type} después de la limpieza. Saltando análisis de {test_type}.")
                        continue

                    df_promedio, df_ventanas = analyze_tremor_windows(cleaned_df)

                    if not df_promedio.empty:
                        result_row = df_promedio.iloc[0].to_dict()
                        result_row['Test'] = test_type
                        results_single_analysis.append(result_row)
                    
                    if not df_ventanas.empty:
                        df_ventanas_copy = df_ventanas.copy()
                        df_ventanas_copy["Test"] = test_type
                        temp_window_dfs.append(df_ventanas_copy)
                        if len(df_ventanas_copy) < min_ventanas_count:
                            min_ventanas_count = len(df_ventanas_copy)

            if temp_window_dfs:
                fig, ax = plt.subplots(figsize=(10, 6))
                for df_plot in temp_window_dfs:
                    test_name = df_plot["Test"].iloc[0]
                    # Ajustar la longitud de los datos a graficar si hay duraciones diferentes
                    if min_ventanas_count != float('inf') and len(df_plot) > min_ventanas_count:
                        df_to_plot = df_plot.iloc[:min_ventanas_count].copy()
                    else:
                        df_to_plot = df_plot.copy()
                    
                    df_to_plot["Tiempo (segundos)"] = df_to_plot["Ventana"] * VENTANA_DURACION_SEG * (1 - SOLAPAMIENTO_VENTANA)
                    ax.plot(df_to_plot["Tiempo (segundos)"], df_to_plot["Amplitud Temblor (cm)"], label=f"{test_name}")

                ax.set_title("Amplitud de Temblor por Ventana de Tiempo")
                ax.set_xlabel("Tiempo (segundos)")
                ax.set_ylabel("Amplitud (cm)")
                xue)
 ax.legend()
          ax.grid(True)
     d (  tg)
  st.pyplot(fig)
                      figures_for_report.append(fig) # Add figure to list for PDF

            else:
                st.warning("No se genpara e datos de ventanas para el gráfico.")
     _lysis:
      if results_single_analysis:
                      df_results_final = pd.DataFrame(results_single_analysis)
                df_results_final_display = df_results_final.set_index('Te For display
 Forle        s play
 
          diagnostico_auto = diagnosticar_rule_based(df_results_final)

                st.subheader("Resultadoslts_inal)

esultados del  Análisis de Temblor")
                st.dataframe(df_results_final_display)
                st.write(f"Diagnóstico automático: **{diag
   }**"trar  # Generar PDF
PDF
    a      rt_bytes = genor_ = generate_tremor_report_pdf(p  pdf_ou
,
         patient_data_for_report,
    atient_daort,
                       results_df=df_results_final,
                    figures=figures_for_report,
                    conclusion_text=f"Diagnóstico automático: {diagnostico_auto}"
                )
                
                st.download_button("📄 Descargar informe PDF", pdf_output_bytes.getvalue(), file_name="informe_temblor.pdf", mime="application/pdf")
                st.info("El archivo se descargará en tu carpeta de descargasor.pdf", mime="application/pdf")forme_temblor.pdf",on/pdf")
                st.i mime="application/pdf")argas predeterminada o el navegador te pedirá la ubicación.")
            else:
                st.warning("No hay suficientes datos de ventanas para graficar los archivos de predicción.")
