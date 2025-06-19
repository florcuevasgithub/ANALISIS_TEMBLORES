# data_processing.py
import pandas as pd

def extraer_datos_paciente(df):
    """
    Extrae datos personales del paciente y parámetros de configuración desde un DataFrame,
    sin modificar las columnas originales del DataFrame.
    """
    # Crear un mapeo normalizado de nombres: {columna_lower: columna_original}
    col_map = {col.lower().strip(): col for col in df.columns}

    datos = {
        "sexo": "No especificado",
        "edad": 0,
        "mano_medida": "No especificada",
        "dedo_medido": "No especificado",
        "Nombre": None,
        "Apellido": None,
        "Diagnostico": None,
        "Antecedente": None,
        "Medicacion": None,
        "Tipo": None,
        "ECP": None, "GPI": None, "NST": None, "Polaridad": None,
        "Duracion": None, "Pulso": None, "Corriente": None,
        "Voltaje": None, "Frecuencia": None
    }

    if not df.empty:
        # Extraer datos personales usando el mapeo
        if "sexo" in col_map and pd.notna(df.at[0, col_map["sexo"]]):
            datos["sexo"] = str(df.at[0, col_map["sexo"]]).strip()

        if "edad" in col_map and pd.notna(df.at[0, col_map["edad"]]):
            try:
                datos["edad"] = int(float(str(df.at[0, col_map["edad"]]).replace(',', '.')))
            except (ValueError, TypeError):
                datos["edad"] = 0 # Keep as 0 or None if conversion fails

        if "mano" in col_map and pd.notna(df.at[0, col_map["mano"]]):
            datos["mano_medida"] = str(df.at[0, col_map["mano"]]).strip()

        if "dedo" in col_map and pd.notna(df.at[0, col_map["dedo"]]):
            datos["dedo_medido"] = str(df.at[0, col_map["dedo"]]).strip()

        # Extraer otros metadatos generales
        for key in ["Nombre", "Apellido", "Diagnostico", "Antecedente", "Medicacion", "Tipo"]:
            key_l = key.lower()
            if key_l in col_map and pd.notna(df.at[0, col_map[key_l]]):
                datos[key] = str(df.at[0, col_map[key_l]])

        # Extraer campos de estimulación/configuración
        for key in ["ECP", "GPI", "NST", "Polaridad", "Duracion", "Pulso", "Corriente", "Voltaje", "Frecuencia"]:
            key_l = key.lower()
            if key_l in col_map and pd.notna(df.at[0, col_map[key_l]]):
                val = str(df.at[0, col_map[key_l]]).replace(',', '.')
                try:
                    if key in ["Duracion", "Pulso", "Corriente", "Voltaje", "Frecuencia"]:
                        datos[key] = float(val)
                    else:
                        datos[key] = val
                except ValueError:
                    datos[key] = val  # Leave as text if not convertible

    return datos

def diagnosticar(df):
    """
    Realiza un diagnóstico de temblor basado en los resultados de las pruebas.
    Asume que df contiene las columnas 'Test', 'Amplitud Temblor (cm)', 'Frecuencia Dominante (Hz)'.
    """
    def max_amp(test):
        fila = df[df['Test'] == test]
        return fila['Amplitud Temblor (cm)'].max() if not fila.empty else 0

    def mean_freq(test):
        fila = df[df['Test'] == test]
        return fila['Frecuencia Dominante (Hz)'].mean() if not fila.empty else 0

    if max_amp('Reposo') > 0.3 and 3 <= mean_freq('Reposo') <= 6.5:
        return "Probable Parkinson"
    elif (max_amp('Postural') > 0.3 or max_amp('Acción') > 0.3) and \
         (7.5 <= mean_freq('Postural') <= 12 or 7.5 <= mean_freq('Acción') <= 12):
        return "Probable Temblor Esencial"
    else:
        return "Temblor dentro de parámetros normales"

