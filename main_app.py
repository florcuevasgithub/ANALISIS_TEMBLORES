# main_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import io
from io import BytesIO

# Import functions and configurations from other files
from config import VENTANA_DURACION_SEG
from data_processing import extraer_datos_paciente, diagnosticar
from signal_analysis import analizar_temblor_por_ventanas_resultante
from pdf_generation import generar_pdf
from ml_model import load_tremor_model, prepare_data_for_prediction


# Inicializar una variable en el estado de sesión para controlar el reinicio
if "reiniciar" not in st.session_state:
    st.session_state.reiniciar = False

st.markdown("""
    <style>
    /* Oculta el texto 'Limit 200MB per file • CSV' */
    div[data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
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

def manejar_reinicio():
    if st.session_state.get("reiniciar", False):
        st.session_state.clear()
        st.experimental_rerun()


# ------------------ Modo principal --------------------

st.title("🧠 Análisis de Temblor")
opcion = st.sidebar.radio("Selecciona una opción:", ["1️⃣ Análisis de una medición", "2️⃣ Comparar dos mediciones", "3️⃣ Predicción de Temblor"])
if st.sidebar.button("🔄 Nuevo análisis"):
    st.session_state.reiniciar = True # Set flag to true
    manejar_reinicio() # Call the handler

if opcion == "1️⃣ Análisis de una medición":
    st.title("📈 Análisis de una medición")

    st.markdown('<div class="prueba-titulo">Subir archivo CSV para prueba en REPOSO</div>', unsafe_allow_html=True)
    reposo_file = st.file_uploader("", type=["csv"], key="reposo")

    st.markdown('<div class="prueba-titulo">Subir archivo CSV para prueba POSTURAL</div>', unsafe_allow_html=True)
    postural_file = st.file_uploader("", type=["csv"], key="postural")

    st.markdown('<div class="prueba-titulo">Subir archivo CSV para prueba en ACCIÓN</div>', unsafe_allow_html=True)
    accion_file = st.file_uploader("", type=["csv"], key="accion")

    st.markdown("""
        <style>
        /* Ocultar el texto original de "Drag and drop file here" */
        div[data-testid="stFileUploaderDropzoneInstructions"] span {
            display: none !important;
        }

        /* Añadir nuestro propio texto arriba del botón */
        div[data-testid="stFileUploaderDropzoneInstructions"]::before {
            content: "Arrastrar archivo aquí";
            font-weight: bold;
            font-size: 16px;
            color: #444;
            display: block;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)


    uploaded_files = {
        "Reposo": reposo_file,
        "Postural": postural_file,
        "Acción": accion_file,
    }

    # Inicializa estas variables FUERA del bloque del botón.
    resultados_globales = []
    datos_paciente_para_pdf = {}
    ventanas_para_grafico = []
    min_ventanas_count = float('inf')
    fig_single_analysis = None # Cambiado el nombre de la variable para evitar conflictos

    if st.button("Iniciar análisis"):
        mediciones_tests = {}
        for test, file in uploaded_files.items():
            if file is not None:
                file.seek(0) # Reset file pointer for re-reading
                mediciones_tests[test] = pd.read_csv(file, encoding='latin1')

        if not mediciones_tests:
            st.warning("Por favor, sube al menos un archivo para iniciar el análisis.")
        else:
            primer_df_cargado = None
            for test, datos in mediciones_tests.items():
                if datos is not None and not datos.empty:
                    primer_df_cargado = datos
                    break

            if primer_df_cargado is not None:
                datos_paciente_para_pdf = extraer_datos_paciente(primer_df_cargado)
            
            for test, datos in mediciones_tests.items():
                if datos is not None and not datos.empty:
                    df_promedio, df_ventanas = analizar_temblor_por_ventanas_resultante(datos, fs=100)

                    if not df_promedio.empty:
                        fila = df_promedio.iloc[0].to_dict()
                        fila['Test'] = test
                        resultados_globales.append(fila)

                    if not df_ventanas.empty:
                        df_ventanas_copy = df_ventanas.copy()
                        df_ventanas_copy["Test"] = test
                        ventanas_para_grafico.append(df_ventanas_copy)
                        if len(df_ventanas_copy) < min_ventanas_count:
                            min_ventanas_count = len(df_ventanas_copy)
                else:
                    st.info(f"No se cargó ningún archivo para el test de '{test}'. Se omitirá este análisis.")


            if ventanas_para_grafico:
                fig_single_analysis, ax = plt.subplots(figsize=(10, 6)) # Asigna a la variable específica
                for df in ventanas_para_grafico:
                    test_name = df["Test"].iloc[0]
                    if min_ventanas_count != float('inf') and len(df) > min_ventanas_count:
                        df_to_plot = df.iloc[:min_ventanas_count].copy()
                    else:
                        df_to_plot = df.copy()
                    
                    df_to_plot["Tiempo (segundos)"] = df_to_plot["Ventana"] * VENTANA_DURACION_SEG
                    ax.plot(df_to_plot["Tiempo (segundos)"], df_to_plot["Amplitud Temblor (cm)"], label=f"{test_name}")

                ax.set_title("Amplitud de Temblor por Ventana de Tiempo (Comparación Visual)")
                ax.set_xlabel("Tiempo (segundos)")
                ax.set_ylabel("Amplitud (cm)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig_single_analysis) # Muestra la variable específica
            else:
                st.warning("No se generaron datos de ventanas para el gráfico.")

            if resultados_globales:
                df_resultados_final = pd.DataFrame(resultados_globales)
                diagnostico_auto = diagnosticar(df_resultados_final)

                st.subheader("Resultados del Análisis de Temblor")
                st.dataframe(df_resultados_final.set_index('Test'))

                generar_pdf(
                    datos_paciente_para_pdf,
                    df_resultados_final,
                    nombre_archivo="informe_temblor.pdf",
                    diagnostico=diagnostico_auto,
                    figs=fig_single_analysis, # Pasa la variable específica (puede ser None o una figura)
                    comparison_mode=False,
                    config1_params=None, # No aplicable para análisis único
                    config2_params=None # No aplicable para análisis único
                )

                with open("informe_temblor.pdf", "rb") as f:
                    st.download_button("📄 Descargar informe PDF", f, file_name="informe_temblor.pdf")
                    st.info("El archivo se descargará en tu carpeta de descargas predeterminada o el navegador te pedirá la ubicación, dependiendo de tu configuración.")
            else:
                st.warning("No se encontraron datos suficientes para el análisis.")

elif opcion == "2️⃣ Comparar dos mediciones":
    st.title("📊 Comparar dos mediciones")

    st.markdown("### Cargar archivos de la **medición 1**")
    config1_archivos = {
        "Reposo": st.file_uploader("Archivo de REPOSO medición 1", type="csv", key="reposo1"),
        "Postural": st.file_uploader("Archivo de POSTURAL medición 1", type="csv", key="postural1"),
        "Acción": st.file_uploader("Archivo de ACCION medición 1", type="csv", key="accion1")
    }

    st.markdown("### Cargar archivos de la **medición 2**")
    config2_archivos = {
        "Reposo": st.file_uploader("Archivo de REPOSO medición 2", type="csv", key="reposo2"),
        "Postural": st.file_uploader("Archivo de POSTURAL medición 2", type="csv", key="postural2"),
        "Acción": st.file_uploader("Archivo de ACCION medición 2", type="csv", key="accion2")
    }

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
        </style>
    """, unsafe_allow_html=True)

    def analizar_configuracion_comparacion(archivos, fs=100):
        resultados = []
        for test, archivo in archivos.items():
            if archivo is not None:
                archivo.seek(0)
                df = pd.read_csv(archivo, encoding='latin1')
                df_promedio, df_ventana = analizar_temblor_por_ventanas_resultante(df, fs=fs)
                if isinstance(df_ventana, pd.DataFrame) and not df_ventana.empty:
                    prom = df_promedio.iloc[0] if not df_promedio.empty else None
                    if prom is not None:
                        freq = prom['Frecuencia Dominante (Hz)']
                        amp = prom['Amplitud Temblor (cm)']
                        rms = prom['RMS (m/s2)']
                        resultados.append({
                            'Test': test,
                            'Frecuencia Dominante (Hz)': round(freq, 2),
                            'RMS (m/s2)': round(rms, 4),
                            'Amplitud Temblor (cm)': round(amp, 2)
                        })
            else:
                st.warning(f"Archivo de {test} no cargado para esta configuración. Se omitirá del análisis.")

        return pd.DataFrame(resultados)

    if st.button("Comparar Mediciones"):
        any_files_uploaded_config1 = any(f is not None for f in config1_archivos.values())
        any_files_uploaded_config2 = any(f is not None for f in config2_archivos.values())
        
        if not any_files_uploaded_config1 and not any_files_uploaded_config2:
            st.warning("Por favor, cargue al menos un archivo para cada medición para iniciar la comparación.")
        else:
            # Read the first available file from each configuration to extract metadata
            df_config1_meta = None
            for f in config1_archivos.values():
                if f is not None:
                    f.seek(0)
                    df_config1_meta = pd.read_csv(f, encoding='latin1')
                    break

            df_config2_meta = None
            for f in config2_archivos.values():
                if f is not None:
                    f.seek(0)
                    df_config2_meta = pd.read_csv(f, encoding='latin1')
                    break
            
            datos_personales = {}
            if df_config1_meta is not None:
                datos_personales = extraer_datos_paciente(df_config1_meta)
            elif df_config2_meta is not None:
                datos_personales = extraer_datos_paciente(df_config2_meta)

            parametros_config1 = {}
            if df_config1_meta is not None:
                parametros_config1 = extraer_datos_paciente(df_config1_meta)

            parametros_config2 = {}
            if df_config2_meta is not None:
                parametros_config2 = extraer_datos_paciente(df_config2_meta)


            df_resultados_config1 = analizar_configuracion_comparacion(config1_archivos)
            df_resultados_config2 = analizar_configuracion_comparacion(config2_archivos)

            amp_avg_config1 = df_resultados_config1['Amplitud Temblor (cm)'].mean() if not df_resultados_config1.empty else 0
            amp_avg_config2 = df_resultados_config2['Amplitud Temblor (cm)'].mean() if not df_resultados_config2.empty else 0

            rms_avg_config1 = df_resultados_config1['RMS (m/s2)'].mean() if not df_resultados_config1.empty else 0
            rms_avg_config2 = df_resultados_config2['RMS (m/s2)'].mean() if not df_resultados_config2.empty else 0

            conclusion = ""
            if amp_avg_config1 < amp_avg_config2:
                conclusion = (
                    f"La Medición 1 muestra una amplitud de temblor promedio ({amp_avg_config1:.2f} cm) "
                    f"más baja que la Medición 2 ({amp_avg_config2:.2f} cm), lo que sugiere una mayor reducción del temblor."
                )
            elif amp_avg_config2 < amp_avg_config1:
                conclusion = (
                    f"La Medición 2 muestra una amplitud de temblor promedio ({amp_avg_config2:.2f} cm) "
                    f"más baja que la Medición 1 ({amp_avg_config1:.2f} cm), lo que sugiere una mayor reducción del temblor."
                )
            else:
                conclusion = (
                    f"Ambas mediciones muestran amplitudes de temblor promedio muy similares ({amp_avg_config1:.2f} cm). "
                )

            st.subheader("Resultados Medición 1")
            st.dataframe(df_resultados_config1)

            st.subheader("Resultados Medición 2")
            st.dataframe(df_resultados_config2)

            st.subheader("Comparación Gráfica de Amplitud por Ventana")
            nombres_test = ["Reposo", "Postural", "Acción"]
            
            # Lista para almacenar todas las figuras para el PDF
            figs_comparison = []

            for test in nombres_test:
                archivo1 = config1_archivos.get(test)
                archivo2 = config2_archivos.get(test)

                if archivo1 is not None and archivo2 is not None:
                    archivo1.seek(0)
                    archivo2.seek(0)
                    df1 = pd.read_csv(archivo1, encoding='latin1')
                    df2 = pd.read_csv(archivo2, encoding='latin1')

                    df1_promedio, df1_ventanas = analizar_temblor_por_ventanas_resultante(df1, fs=100)
                    df2_promedio, df2_ventanas = analizar_temblor_por_ventanas_resultante(df2, fs=100)

                    if not df1_ventanas.empty and not df2_ventanas.empty:
                        fig, ax = plt.subplots(figsize=(10, 5))

                        df1_ventanas["Tiempo (segundos)"] = df1_ventanas["Ventana"] * VENTANA_DURACION_SEG
                        df2_ventanas["Tiempo (segundos)"] = df2_ventanas["Ventana"] * VENTANA_DURACION_SEG

                        ax.plot(df1_ventanas["Tiempo (segundos)"], df1_ventanas["Amplitud Temblor (cm)"], label="Medición 1", color="blue")
                        ax.plot(df2_ventanas["Tiempo (segundos)"], df2_ventanas["Amplitud Temblor (cm)"], label="Medición 2", color="orange")
                        ax.set_title(f"Amplitud por Ventana - {test}")
                        ax.set_xlabel("Tiempo (segundos)")
                        ax.set_ylabel("Amplitud (cm)")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig) # Muestra en Streamlit
                        figs_comparison.append(fig) # Añade la figura a la lista para el PDF
                        plt.close(fig) # Cierra la figura para liberar memoria
                    else:
                        st.warning(f"No hay suficientes datos de ventanas para graficar el test: {test}")
                else:
                    st.info(f"Archivos no cargados para el test {test} en ambas mediciones. Se omitirá este gráfico.")
            
            st.subheader("Conclusión del Análisis Comparativo")
            st.write(conclusion)

            combined_df_for_pdf = pd.DataFrame()
            if not df_resultados_config1.empty:
                df_resultados_config1['Measurement'] = 1
                combined_df_for_pdf = pd.concat([combined_df_for_pdf, df_resultados_config1])
            if not df_resultados_config2.empty:
                df_resultados_config2['Measurement'] = 2
                combined_df_for_pdf = pd.concat([combined_df_for_pdf, df_resultados_config2])
            
            if not combined_df_for_pdf.empty:
                generar_pdf(
                    datos_paciente_dict=datos_personales,
                    df_resultados=combined_df_for_pdf,
                    nombre_archivo="informe_comparativo_temblor.pdf",
                    diagnostico=conclusion,
                    figs=figs_comparison, # Pasa la lista de figuras
                    comparison_mode=True,
                    config1_params=parametros_config1,
                    config2_params=parametros_config2
                )

                with open("informe_comparativo_temblor.pdf", "rb") as f:
                    st.download_button(
                        label="Descargar Informe PDF",
                        data=f.read(),
                        file_name="informe_comparativo_temblor.pdf",
                        mime="application/pdf"
                    )
                st.info("El archivo se descargará en tu carpeta de descargas predeterminada o el navegador te pedirá la ubicación, dependiendo de tu configuración.")
            else:
                st.warning("No hay datos suficientes para generar un informe comparativo PDF.")


elif opcion == "3️⃣ Predicción de Temblor":

    st.title("🔮 Predicción de Temblor")
    st.markdown("### Cargar archivos CSV para la Predicción")

    prediccion_reposo_file = st.file_uploader("Archivo de REPOSO para Predicción", type="csv", key="prediccion_reposo")
    prediccion_postural_file = st.file_uploader("Archivo de POSTURAL para Predicción", type="csv", key="prediccion_postural")
    prediccion_accion_file = st.file_uploader("Archivo de ACCION para Predicción", type="csv", key="prediccion_accion")

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
        </style>
    """, unsafe_allow_html=True)

    if st.button("Realizar Predicción"):
        prediccion_files_correctas = {
            "Reposo": prediccion_reposo_file,
            "Postural": prediccion_postural_file,
            "Acción": prediccion_accion_file
        }

        any_file_uploaded = any(file is not None for file in prediccion_files_correctas.values())

        if not any_file_uploaded:
            st.warning("Por favor, sube al menos un archivo CSV para realizar la predicción.")
        else:
            avg_tremor_metrics = {}
            datos_paciente = {} # Initialize datos_paciente here

            # Extract patient data from the first successfully loaded file
            first_df_loaded = None
            for test_type, uploaded_file in prediccion_files_correctas.items():
                if uploaded_file is not None:
                    uploaded_file.seek(0)
                    first_df_loaded = pd.read_csv(uploaded_file, encoding='latin1')
                    datos_paciente = extraer_datos_paciente(first_df_loaded)
                    break # Get patient data from the first file and then proceed

            if not datos_paciente: # If no file was successfully loaded to get patient data
                st.error("No se pudo extraer información del paciente. Asegúrate de que los archivos contengan datos válidos.")
                st.stop() # Stop execution if no patient data can be extracted

            for test_type, uploaded_file in prediccion_files_correctas.items():
                if uploaded_file is not None:
                    uploaded_file.seek(0)
                    df_current_test = pd.read_csv(uploaded_file, encoding='latin1')

                    df_promedio, _ = analizar_temblor_por_ventanas_resultante(df_current_test, fs=100)

                    if not df_promedio.empty:
                        avg_tremor_metrics[test_type] = df_promedio.iloc[0].to_dict()
                    else:
                        st.warning(f"No se pudieron calcular métricas de temblor para {test_type}. Se usarán NaN.")
                        avg_tremor_metrics[test_type] = {
                            'Frecuencia Dominante (Hz)': np.nan,
                            'RMS (m/s2)': np.nan,
                            'Amplitud Temblor (cm)': np.nan
                        }

            if not avg_tremor_metrics:
                st.error("No se pudo procesar ningún archivo cargado para la predicción. Asegúrate de que los archivos contengan datos válidos.")
            else:
                st.subheader("Datos de Temblor Calculados para la Predicción:")
                df_metrics_display = pd.DataFrame.from_dict(avg_tremor_metrics, orient='index')
                df_metrics_display.index.name = "Test"
                st.dataframe(df_metrics_display)

                df_for_prediction = prepare_data_for_prediction(datos_paciente, avg_tremor_metrics)

                st.subheader("DataFrame preparado para el Modelo de Predicción:")
                st.dataframe(df_for_prediction)

                model_filename = 'tremor_prediction_model_V2.joblib'
                
                prediction_result_str = "No se pudo realizar la predicción."
                prediction_probabilities_dict = {}

                try:
                    modelo_cargado = load_tremor_model(model_filename)
                    prediction = modelo_cargado.predict(df_for_prediction)
                    prediction_result_str = prediction[0]

                    st.subheader("Resultado de la Predicción:")
                    st.success(f"La predicción del modelo es: **{prediction_result_str}**")

                    if hasattr(modelo_cargado, 'predict_proba'):
                        probabilities = modelo_cargado.predict_proba(df_for_prediction)
                        st.write("Probabilidades por clase:")
                        if hasattr(modelo_cargado, 'classes_'):
                            for i, class_label in enumerate(modelo_cargado.classes_):
                                st.write(f"- **{class_label}**: {probabilities[0][i]*100:.2f}%")
                                prediction_probabilities_dict[class_label] = probabilities[0][i]*100
                        else:
                            st.info("El modelo no tiene el atributo 'classes_'. No se pueden mostrar las etiquetas de clase.")

                except FileNotFoundError as e:
                    st.error(f"Error: {e}")
                    st.error("Asegúrate de que el archivo del modelo esté en la misma carpeta que este script.")
                except Exception as e:
                    st.error(f"Ocurrió un error al usar el modelo: {e}")
                    st.error("Verifica que el DataFrame `df_for_prediction` coincida con lo que espera el modelo.")
                
                # Prepare prediction info for PDF
                prediction_info_for_pdf = {
                    "prediction": prediction_result_str,
                    "probabilities": prediction_probabilities_dict
                }


                # Optional graph generation for prediction
                all_ventanas_for_plot = []
                current_min_ventanas = float('inf')
                figs_prediction = [] # Lista para las figuras de predicción para el PDF

                for test_type, uploaded_file in prediccion_files_correctas.items():
                    if uploaded_file is not None:
                        uploaded_file.seek(0)
                        df_temp = pd.read_csv(uploaded_file, encoding='latin1')
                        _, df_ventanas_temp = analizar_temblor_por_ventanas_resultante(df_temp, fs=100)

                        if not df_ventanas_temp.empty:
                            df_ventanas_temp_copy = df_ventanas_temp.copy()
                            df_ventanas_temp_copy["Test"] = test_type
                            all_ventanas_for_plot.append(df_ventanas_temp_copy)

                            if len(df_ventanas_temp_copy) < current_min_ventanas:
                                current_min_ventanas = len(df_ventanas_temp_copy)

                if all_ventanas_for_plot:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for df_plot in all_ventanas_for_plot:
                        test_name = df_plot["Test"].iloc[0]
                        if current_min_ventanas != float('inf') and len(df_plot) > current_min_ventanas:
                            df_to_plot = df_plot.iloc[:current_min_ventanas].copy()
                        else:
                            df_to_plot = df_plot.copy()

                        df_to_plot["Tiempo (segundos)"] = df_to_plot["Ventana"] * VENTANA_DURACION_SEG
                        ax.plot(df_to_plot["Tiempo (segundos)"], df_to_plot["Amplitud Temblor (cm)"], label=f"{test_name}")

                    ax.set_title("Amplitud de Temblor por Ventana de Tiempo (Archivos de Predicción)")
                    ax.set_xlabel("Tiempo (segundos)")
                    ax.set_ylabel("Amplitud (cm)")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig) # Muestra en Streamlit
                    figs_prediction.append(fig) # Añade la figura a la lista para el PDF
                    plt.close(fig)
                else:
                    st.warning("No hay suficientes datos de ventanas para graficar los archivos de predicción.")

                # Generar PDF para la opción 3
                if not df_metrics_display.empty: # Asegurarse de que hay métricas para la tabla
                    generar_pdf(
                        datos_paciente_dict=datos_paciente, # Información personal
                        df_resultados=df_metrics_display, # Cuadro comparativo de métricas calculadas
                        nombre_archivo="informe_prediccion_temblor.pdf",
                        diagnostico=prediction_result_str, # La predicción como diagnóstico/conclusión
                        figs=figs_prediction, # Las figuras generadas
                        comparison_mode=False, # No es modo comparación
                        config1_params=None, # No aplicable
                        config2_params=None, # No aplicable
                        prediction_info=prediction_info_for_pdf # Información de la predicción
                    )

                    with open("informe_prediccion_temblor.pdf", "rb") as f:
                        st.download_button(
                            label="Descargar Informe PDF de Predicción",
                            data=f.read(),
                            file_name="informe_prediccion_temblor.pdf",
                            mime="application/pdf"
                        )
                    st.info("El archivo se descargará en tu carpeta de descargas predeterminada o el navegador te pedirá la ubicación, dependiendo de tu configuración.")
                else:
                    st.warning("No hay datos suficientes para generar un informe de predicción PDF.")
