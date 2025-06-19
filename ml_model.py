# ml_model.py
import joblib
import pandas as pd
import numpy as np

def load_tremor_model(model_filename='tremor_prediction_model_V2.joblib'):
    """Carga el modelo de predicción de temblor."""
    try:
        modelo_cargado = joblib.load(model_filename)
        return modelo_cargado
    except FileNotFoundError:
        raise FileNotFoundError(f"El archivo del modelo '{model_filename}' no se encontró. Asegúrate de que esté en la misma carpeta que este script.")
    except Exception as e:
        raise Exception(f"Error al cargar el modelo: {e}")

def prepare_data_for_prediction(datos_paciente, avg_tremor_metrics):
    """
    Prepara un DataFrame con las características necesarias para la predicción
    del modelo de temblor.
    """
    data_for_model = {}
    
    # Patient Demographics
    edad_val = datos_paciente.get('edad', np.nan)
    try:
        data_for_model['edad'] = int(float(edad_val)) if pd.notna(edad_val) else np.nan
    except (ValueError, TypeError):
        data_for_model['edad'] = np.nan

    data_for_model['sexo'] = datos_paciente.get('sexo', 'no especificado').lower()
    data_for_model['mano_medida'] = datos_paciente.get('mano_medida', 'no especificada').lower()
    data_for_model['dedo_medido'] = datos_paciente.get('dedo_medido', 'no especificado').lower()

    # Tremor Metrics per Test Type
    feature_name_map = {
        "Reposo": "Reposo",
        "Postural": "Postural",
        "Acción": "Accion"
    }

    for original_test_type, model_feature_prefix in feature_name_map.items():
        metrics = avg_tremor_metrics.get(original_test_type, {})
        data_for_model[f'Frec_{model_feature_prefix}'] = metrics.get('Frecuencia Dominante (Hz)', np.nan)
        data_for_model[f'RMS_{model_feature_prefix}'] = metrics.get('RMS (m/s2)', np.nan)
        data_for_model[f'Amp_{model_feature_prefix}'] = metrics.get('Amplitud Temblor (cm)', np.nan)

    expected_features_for_model = [
        'edad',
        'Frec_Reposo', 'RMS_Reposo', 'Amp_Reposo',
        'Frec_Postural', 'RMS_Postural', 'Amp_Postural',
        'Frec_Accion', 'RMS_Accion', 'Amp_Accion',
        'sexo', 'mano_medida', 'dedo_medido'
    ]
    
    # Create DataFrame, ensuring all expected columns are present, even if NaN
    df_for_prediction = pd.DataFrame([data_for_model]).reindex(columns=expected_features_for_model)

    return df_for_prediction
