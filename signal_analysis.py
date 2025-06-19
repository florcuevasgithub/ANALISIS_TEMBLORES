# signal_analysis.py
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch
from ahrs.filters import Mahony

# Import the global configuration
from config import VENTANA_DURACION_SEG

def q_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),         2*(x*y - z*w),           2*(x*z + y*w)],
        [2*(x*y + z*w),               1 - 2*(x**2 + z**2),       2*(y*z - x*w)],
        [2*(x*z - y*w),               2*(y*z + x*w),           1 - 2*(x**2 + y**2)]
    ])

def filtrar_temblor(signal, fs=100):
    b, a = butter(N=4, Wn=[1, 15], btype='bandpass', fs=fs)
    return filtfilt(b, a, signal)

def analizar_temblor_por_ventanas_resultante(df, fs=100, ventana_seg=VENTANA_DURACION_SEG):
    required_cols = ['Acel_X', 'Acel_Y', 'Acel_Z', 'GiroX', 'GiroY', 'GiroZ']
    # Ensure all required columns exist and drop NaNs only from these specific columns for AHRS
    df_filtered = df[required_cols].dropna() 

    if df_filtered.empty:
        return pd.DataFrame(), pd.DataFrame() # Return empty if no valid data after dropping NaNs

    acc = df_filtered[['Acel_X', 'Acel_Y', 'Acel_Z']].to_numpy()
    gyr = np.radians(df_filtered[['GiroX', 'GiroY', 'GiroZ']].to_numpy())
    
    # Mahony filter requires at least 2 data points for quaternion calculation in some cases
    if len(acc) < 2: 
        return pd.DataFrame(), pd.DataFrame()

    mahony = Mahony(gyr=gyr, acc=acc, frequency=fs)
    Q = mahony.Q
    
    linear_accelerations_magnitude = []
    g_world_vector = np.array([0.0, 0.0, 9.81])

    for i in range(len(acc)):
        if i >= len(Q): # Safety check if Q is shorter than acc due to internal AHRS handling
            break
        q = Q[i]
        acc_measured = acc[i]
        R_W_B = q_to_matrix(q)
        gravity_in_sensor_frame = R_W_B @ g_world_vector
        linear_acc_sensor_frame = acc_measured - gravity_in_sensor_frame
        linear_accelerations_magnitude.append(np.linalg.norm(linear_acc_sensor_frame))

    movimiento_lineal = np.array(linear_accelerations_magnitude)
    
    if len(movimiento_lineal) < 1: # Ensure there's enough data after gravity removal
        return pd.DataFrame(), pd.DataFrame()

    señal_filtrada = filtrar_temblor(movimiento_lineal, fs)

    resultados_por_ventana = []
    tamaño_ventana = int(fs * ventana_seg)
    
    if len(señal_filtrada) < tamaño_ventana:
        # Not enough data for even one full window, return empty
        return pd.DataFrame(), pd.DataFrame()

    num_ventanas = len(señal_filtrada) // tamaño_ventana

    for i in range(num_ventanas):
        segmento = señal_filtrada[i*tamaño_ventana:(i+1)*tamaño_ventana]
        segmento = segmento - np.mean(segmento) # Detrend segment

        # Avoid Welch calculation on empty or too short segments
        if len(segmento) < 1:
            continue

        # nperseg must be less than or equal to the segment length
        nperseg_val = min(tamaño_ventana, len(segmento))
        if nperseg_val == 0:
            freq_dominante = 0.0
            Pxx_max = 0.0
        else:
            f, Pxx = welch(segmento, fs=fs, nperseg=nperseg_val)
            if len(Pxx) > 0:
                freq_dominante = f[np.argmax(Pxx)]
                Pxx_max = np.max(Pxx)
            else:
                freq_dominante = 0.0
                Pxx_max = 0.0


        varianza = np.var(segmento)
        rms = np.sqrt(np.mean(segmento**2))
        amp_g = (np.max(segmento) - np.min(segmento))/2

        if freq_dominante > 1.5: # Apply amplitude calculation only if dominant freq is meaningful
            amp_cm = ((amp_g * 100) / ((2 * np.pi * freq_dominante) ** 2))*2
        else:
            amp_cm = 0.0

        resultados_por_ventana.append({
           'Ventana': i,
           'Frecuencia Dominante (Hz)': freq_dominante,
           'RMS (m/s2)': rms,
           'Amplitud Temblor (g)': amp_g,
           'Amplitud Temblor (cm)': amp_cm
         })

    df_por_ventana = pd.DataFrame(resultados_por_ventana)

    if not df_por_ventana.empty:
        # Ensure 'Ventana' is not included in numeric_only mean for the average calculation if it's not a numeric metric
        # Remove 'Ventana' from columns before calculating mean if it's just an index
        cols_for_mean = [col for col in df_por_ventana.columns if col not in ['Ventana', 'Amplitud Temblor (g)']]
        promedio = df_por_ventana[cols_for_mean].mean().to_dict()
        df_promedio = pd.DataFrame([{
            'Frecuencia Dominante (Hz)': promedio.get('Frecuencia Dominante (Hz)', 0.0),
            'RMS (m/s2)': promedio.get('RMS (m/s2)', 0.0),
            'Amplitud Temblor (cm)': promedio.get('Amplitud Temblor (cm)', 0.0)
        }])
    else:
        df_promedio = pd.DataFrame()

    return df_promedio, df_por_ventana
