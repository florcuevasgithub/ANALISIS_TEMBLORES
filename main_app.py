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
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                figures_for_report.append(fig) # Add figure to list for PDF

            else:
                st.warning("No se generaron datos de ventanas para el gráfico.")
            
            if results_single_analysis:
                df_results_final = pd.DataFrame(results_single_analysis)
                df_results_final_display = df_results_final.set_index('Test') # For display
                
                diagnostico_auto = diagnosticar_rule_based(df_results_final)

                st.subheader("Resultados del Análisis de Temblor")
                st.dataframe(df_results_final_display)
                st.write(f"Diagnóstico automático: **{diagnostico_auto}**")

                # Generar PDF
                pdf_output_bytes = generate_tremor_report_pdf(
                    patient_data_for_report,
                    results_df=df_results_final,
                    figures=figures_for_report,
                    conclusion_text=f"Diagnóstico automático: {diagnostico_auto}"
                )
                
                st.download_button("📄 Descargar informe PDF", pdf_output_bytes.getvalue(), file_name="informe_temblor.pdf", mime="application/pdf")
                st.info("El archivo se descargará en tu carpeta de descargas predeterminada o el navegador te pedirá la ubicación.")
            else:
                st.warning("No se encontraron datos suficientes para el análisis.")


# --- Opción 2: Comparar dos mediciones ---
elif opcion == "2️⃣ Comparar dos mediciones":
    st.title("📊 Comparar dos mediciones")

    st.markdown("### Cargar archivos de la **medición 1**")
    config1_archivos_raw = {
        "Reposo": st.file_uploader("Archivo de REPOSO medición 1", type="csv", key="reposo1"),
        "Postural": st.file_uploader("Archivo de POSTURAL medición 1", type="csv", key="postural1"),
        "Acción": st.file_uploader("Archivo de ACCION medición 1", type="csv", key="accion1")
    }

    st.markdown("### Cargar archivos de la **medición 2**")
    config2_archivos_raw = {
        "Reposo": st.file_uploader("Archivo de REPOSO medición 2", type="csv", key="reposo2"),
        "Postural": st.file_uploader("Archivo de POSTURAL medición 2", type="csv", key="postural2"),
        "Acción": st.file_uploader("Archivo de ACCION medición 2", type="csv", key="accion2")
    }

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


    if st.button("Comparar Mediciones"):
        # Cargar todos los archivos raw para ambas configuraciones
        config1_loaded_dfs = {test: load_csv(file_obj) for test, file_obj in config1_archivos_raw.items()}
        config2_loaded_dfs = {test: load_csv(file_obj) for test, file_obj in config2_archivos_raw.items()}

        # Verificar que se hayan cargado todos los archivos necesarios para la comparación
        all_config1_loaded = all(df is not None and not df.empty for df in config1_loaded_dfs.values())
        all_config2_loaded = all(df is not None and not df.empty for df in config2_loaded_dfs.values())

        if not all_config1_loaded or not all_config2_loaded:
            st.warning("Por favor, cargue los 3 archivos para ambas mediciones.")
        else:
            # Extraer datos del paciente del primer archivo válido de la Configuración 1
            patient_data_for_report = extract_patient_data(config1_loaded_dfs["Reposo"])
            
            results_config1 = []
            results_config2 = []
            figures_for_report = []

            for test_type in ["Reposo", "Postural", "Acción"]:
                st.info(f"Procesando {test_type} para Medición 1 y Medición 2...")

                # Medición 1
                cleaned_df1 = clean_sensor_data(config1_loaded_dfs[test_type])
                if not cleaned_df1.empty:
                    df1_promedio, df1_ventanas = analyze_tremor_windows(cleaned_df1)
                    if not df1_promedio.empty:
                        result_row1 = df1_promedio.iloc[0].to_dict()
                        result_row1['Test'] = test_type
                        results_config1.append(result_row1)
                else:
                    st.warning(f"No hay datos válidos en el archivo {test_type} de Medición 1 después de la limpieza.")
                    
                # Medición 2
                cleaned_df2 = clean_sensor_data(config2_loaded_dfs[test_type])
                if not cleaned_df2.empty:
                    df2_promedio, df2_ventanas = analyze_tremor_windows(cleaned_df2)
                    if not df2_promedio.empty:
                        result_row2 = df2_promedio.iloc[0].to_dict()
                        result_row2['Test'] = test_type
                        results_config2.append(result_row2)
                else:
                    st.warning(f"No hay datos válidos en el archivo {test_type} de Medición 2 después de la limpieza.")

                # Gráfico comparativo por test
                if not cleaned_df1.empty and not cleaned_df2.empty and not df1_ventanas.empty and not df2_ventanas.empty:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    df1_ventanas["Tiempo (segundos)"] = df1_ventanas["Ventana"] * VENTANA_DURACION_SEG * (1 - SOLAPAMIENTO_VENTANA)
                    df2_ventanas["Tiempo (segundos)"] = df2_ventanas["Ventana"] * VENTANA_DURACION_SEG * (1 - SOLAPAMIENTO_VENTANA)

                    # Asegurarse de que las series tengan la misma longitud para un gráfico significativo
                    min_len = min(len(df1_ventanas), len(df2_ventanas))
                    ax.plot(df1_ventanas["Tiempo (segundos)"].iloc[:min_len], df1_ventanas["Amplitud Temblor (cm)"].iloc[:min_len], label="Medición 1", color="blue")
                    ax.plot(df2_ventanas["Tiempo (segundos)"].iloc[:min_len], df2_ventanas["Amplitud Temblor (cm)"].iloc[:min_len], label="Medición 2", color="orange")
                    
                    ax.set_title(f"Amplitud por Ventana - {test_type}")
                    ax.set_xlabel("Tiempo (segundos)")
                    ax.set_ylabel("Amplitud (cm)")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                    figures_for_report.append(fig) # Add figure to list for PDF
                else:
                    st.warning(f"No se pudieron generar gráficos por ventana para el test {test_type} debido a datos insuficientes.")

            if results_config1 and results_config2:
                df_results_config1 = pd.DataFrame(results_config1)
                df_results_config2 = pd.DataFrame(results_config2)

                st.subheader("Resultados Medición 1")
                st.dataframe(df_results_config1.set_index('Test'))

                st.subheader("Resultados Medición 2")
                st.dataframe(df_results_config2.set_index('Test'))

                amp_avg_config1 = df_results_config1['Amplitud Temblor (cm)'].mean()
                amp_avg_config2 = df_results_config2['Amplitud Temblor (cm)'].mean()
                
                conclusion_text = ""
                if amp_avg_config1 < amp_avg_config2:
                    conclusion_text = (
                        f"La Medición 1 muestra una amplitud de temblor promedio ({amp_avg_config1:.2f} cm) "
                        f"más baja que la Medición 2 ({amp_avg_config2:.2f} cm), lo que sugiere una mayor reducción del temblor."
                    )
                elif amp_avg_config2 < amp_avg_config1:
                    conclusion_text = (
                        f"La Medición 2 muestra una amplitud de temblor promedio ({amp_avg_config2:.2f} cm) "
                        f"más baja que la Medición 1 ({amp_avg_config1:.2f} cm), lo que sugiere una mayor reducción del temblor."
                    )
                else:
                    conclusion_text = (
                        f"Ambas mediciones muestran amplitudes de temblor promedio muy similares ({amp_avg_config1:.2f} cm). "
                    )
                st.subheader("Conclusión del Análisis Comparativo")
                st.write(conclusion_text)

                # Generar PDF comparativo
                pdf_output_bytes = generate_tremor_report_pdf(
                    patient_data_for_report,
                    comparison_results_df1=df_results_config1,
                    comparison_results_df2=df_results_config2,
                    conclusion_text=conclusion_text,
                    figures=figures_for_report,
                    filename="informe_comparativo_temblor.pdf"
                )

                st.download_button(
                    label="Descargar Informe Comparativo PDF",
                    data=pdf_output_bytes.getvalue(),
                    file_name="informe_comparativo_temblor.pdf",
                    mime="application/pdf"
                )
                st.info("El archivo se descargará en tu carpeta de descargas predeterminada o el navegador te pedirá la ubicación.")
            else:
                st.warning("No se pudieron comparar las mediciones. Asegúrate de que los archivos contengan datos válidos para todas las pruebas.")


# --- Opción 3: Predicción de Temblor ---
elif opcion == "3️⃣ Predicción de Temblor":
    st.title("🔮 Predicción de Temblor")
    st.markdown("### Cargar archivos CSV para la Predicción")

    prediccion_reposo_file = st.file_uploader("Archivo de REPOSO para Predicción", type="csv", key="prediccion_reposo")
    prediccion_postural_file = st.file_uploader("Archivo de POSTURAL para Predicción", type="csv", key="prediccion_postural")
    prediccion_accion_file = st.file_uploader("Archivo de ACCION para Predicción", type="csv", key="prediccion_accion")

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


    prediccion_files_raw = {
        "Reposo": prediccion_reposo_file,
        "Postural": prediccion_postural_file,
        "Acción": prediccion_accion_file
    }

    if st.button("Realizar Predicción"):
        avg_tremor_metrics = {}
        patient_data = {} # Para almacenar los datos del paciente para el modelo
        figures_for_report = [] # Para el gráfico opcional

        # Cargar y limpiar archivos, extraer datos del paciente
        first_valid_df_raw = None
        for test_type, file_obj in prediccion_files_raw.items():
            if file_obj is not None:
                raw_df = load_csv(file_obj)
                if not raw_df.empty:
                    # Extraer datos del paciente una sola vez del primer archivo válido
                    if not patient_data:
                        patient_data = extract_patient_data(raw_df)
                    
                    # Limpiar datos del sensor
                    cleaned_df = clean_sensor_data(raw_df)
                    if cleaned_df.empty:
                        st.warning(f"No hay datos de sensor válidos en el archivo de {test_type} después de la limpieza. Saltando análisis de {test_type}.")
                        avg_tremor_metrics[test_type] = {
                            'Frecuencia Dominante (Hz)': np.nan, 'RMS (m/s2)': np.nan, 'Amplitud Temblor (cm)': np.nan
                        }
                        continue

                    # Analizar temblor
                    df_promedio, df_ventanas = analyze_tremor_windows(cleaned_df)

                    if not df_promedio.empty:
                        avg_tremor_metrics[test_type] = df_promedio.iloc[0].to_dict()
                    else:
                        st.warning(f"No se pudieron calcular métricas de temblor para {test_type}. Se usarán NaN.")
                        avg_tremor_metrics[test_type] = {
                            'Frecuencia Dominante (Hz)': np.nan, 'RMS (m/s2)': np.nan, 'Amplitud Temblor (cm)': np.nan
                        }
                    
                    # Preparar datos para el gráfico de amplitud por ventana
                    if not df_ventanas.empty:
                        df_ventanas_copy = df_ventanas.copy()
                        df_ventanas_copy["Test"] = test_type
                        figures_for_report.append((test_type, df_ventanas_copy)) # Almacenar para graficar

        if not avg_tremor_metrics or all(pd.isna(v['Frecuencia Dominante (Hz)']) for v in avg_tremor_metrics.values()):
            st.error("No se pudo procesar ningún archivo cargado para la predicción o los datos son insuficientes. Asegúrate de que los archivos contengan datos válidos.")
        else:
            st.subheader("Datos de Temblor Calculados para la Predicción:")
            df_metrics_display = pd.DataFrame.from_dict(avg_tremor_metrics, orient='index')
            df_metrics_display.index.name = "Test"
            st.dataframe(df_metrics_display)

            # Preparar el diccionario de características para el modelo
            features_for_model = {}
            # Datos demográficos del paciente
            features_for_model['edad'] = patient_data.get('edad', np.nan)
            features_for_model['sexo'] = patient_data.get('sexo', 'no especificado').lower()
            features_for_model['mano_medida'] = patient_data.get('mano_medida', 'no especificada').lower()
            features_for_model['dedo_medido'] = patient_data.get('dedo_medido', 'no especificado').lower()

            # Métricas de temblor por tipo de prueba
            feature_name_map = {"Reposo": "Reposo", "Postural": "Postural", "Acción": "Accion"}
            for original_test_type, model_feature_prefix in feature_name_map.items():
                metrics = avg_tremor_metrics.get(original_test_type, {})
                features_for_model[f'Frec_{model_feature_prefix}'] = metrics.get('Frecuencia Dominante (Hz)', np.nan)
                features_for_model[f'RMS_{model_feature_prefix}'] = metrics.get('RMS (m/s2)', np.nan)
                features_for_model[f'Amp_{model_feature_prefix}'] = metrics.get('Amplitud Temblor (cm)', np.nan)

            st.subheader("Características preparadas para el Modelo de Predicción:")
            st.json(features_for_model)
            st.write("Claves presentes:", list(features_for_model.keys()))

            # Cargar y usar el modelo de predicción
            model = load_prediction_model()
            if model:
                prediction, probabilities = make_tremor_prediction(model, features_for_model, EXPECTED_FEATURES_FOR_MODEL)

                if prediction is not None:
                    st.subheader("Resultado de la Predicción:")
                    st.success(f"La predicción del modelo es: **{prediction}**")

                    if probabilities is not None:
                        st.write("Probabilidades por clase:")
                        if hasattr(model, 'classes_'):
                            for i, class_label in enumerate(model.classes_):
                                st.write(f"- **{class_label}**: {probabilities[0][i]*100:.2f}%")
                        else:
                            st.info("El modelo no tiene el atributo 'classes_'. No se pueden mostrar las etiquetas de clase para las probabilidades.")
                else:
                    st.error("No se pudo obtener una predicción del modelo.")
            
            # Generar el gráfico de amplitud por ventana para la predicción
            if figures_for_report:
                # Encontrar la longitud mínima para graficar
                min_plot_len = float('inf')
                for _, df_win in figures_for_report:
                    if len(df_win) < min_plot_len:
                        min_plot_len = len(df_win)
                
                plot_figs = []
                st.subheader("Amplitud de Temblor por Ventana (Archivos de Predicción)")
                for test_type, df_plot in figures_for_report:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df_to_plot = df_plot.iloc[:min_plot_len].copy()
                    df_to_plot["Tiempo (segundos)"] = df_to_plot["Ventana"] * VENTANA_DURACION_SEG * (1 - SOLAPAMIENTO_VENTANA)
                    ax.plot(df_to_plot["Tiempo (segundos)"], df_to_plot["Amplitud Temblor (cm)"], label=f"{test_type}")
                    ax.set_title(f"Amplitud por Ventana - {test_type}")
                    ax.set_xlabel("Tiempo (segundos)")
                    ax.set_ylabel("Amplitud (cm)")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                    plot_figs.append(fig) # Añadir la figura generada para el informe PDF
                
                # Generar PDF de predicción (podría ser una función diferente en pdf_generation)
                pdf_output_bytes = generate_tremor_report_pdf(
                    patient_data,
                    results_df=df_metrics_display.reset_index().rename(columns={'index': 'Test'}), # Asegurar formato de tabla
                    conclusion_text=f"Predicción del Modelo: {prediction}",
                    figures=plot_figs,
                    filename="informe_prediccion_temblor.pdf"
                )
                st.download_button("📄 Descargar informe de predicción PDF", pdf_output_bytes.getvalue(), file_name="informe_prediccion_temblor.pdf", mime="application/pdf")
                st.info("El archivo se descargará en tu carpeta de descargas predeterminada o el navegador te pedirá la ubicación.")
            else:
                st.warning("No hay suficientes datos de ventanas para graficar los archivos de predicción.")
