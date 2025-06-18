# config.py

# --- Variables Globales de Configuración ---

# Frecuencia de muestreo (Hz) - AJUSTAR SI ES DIFERENTE EN TUS DATOS
FS = 100

# Duración de cada ventana en segundos para el análisis de temblor
VENTANA_DURACION_SEG = 2

# Solapamiento entre ventanas (ej. 0.5 para 50% de solapamiento)
SOLAPAMIENTO_VENTANA = 0.5

# Características esperadas por tu modelo ML
# ¡IMPORTANTE! Asegúrate que 'Accion' está SIN tilde si así lo espera tu modelo entrenado.
EXPECTED_FEATURES_FOR_MODEL = [
    'edad',
    'Frec_Reposo', 'RMS_Reposo', 'Amp_Reposo',
    'Frec_Postural', 'RMS_Postural', 'Amp_Postural',
    'Frec_Accion', 'RMS_Accion', 'Amp_Accion',
    'sexo',
    'mano_medida',
    'dedo_medido'
]

# Columnas del sensor esperadas en los archivos CSV
# ¡IMPORTANTE! VERIFICA ESTOS NOMBRES EXACTOS EN TUS CSVs (mayúsculas/minúsculas, guiones, etc.)
SENSOR_COLS = ['Acel_X', 'Acel_Y', 'Acel_Z', 'GiroX', 'GiroY', 'GiroZ']

# Rango de frecuencia para el filtro de temblor
TREMOR_FILTER_BAND = [1, 15] # Hz

# Orden del filtro Butterworth
FILTER_ORDER = 4
