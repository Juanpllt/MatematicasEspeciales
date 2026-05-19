import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import subprocess
import tempfile
import os


audio = None
sr = None


def convertir_a_wav(ruta_audio):


    archivo_temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    )
    ruta_wav = archivo_temp.name
    archivo_temp.close()

    comando = [
        "ffmpeg",
        "-i", ruta_audio,
        "-ar", "16000",
        "-ac", "1",
        ruta_wav,
        "-y"
    ]

    resultado = subprocess.run(
        comando,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )


    if resultado.returncode != 0:
        raise RuntimeError(
            f"ffmpeg falló:\n{resultado.stderr.decode()}"
        )


    if not os.path.exists(ruta_wav):
        raise FileNotFoundError(
            f"No se generó el archivo WAV en: {ruta_wav}"
        )

    return ruta_wav


def cargar_audio():

    global audio, sr

    ruta = filedialog.askopenfilename(
        title="Seleccionar audio",
        filetypes=[("Audios", "*.m4a *.mp4 *.wav *.mp3")]
    )

    if ruta == "":
        return

    ruta_wav = None

    try:
        ruta_wav = convertir_a_wav(ruta)

        audio, sr = librosa.load(
            ruta_wav,
            sr=16000,
            mono=True
        )

        label_archivo.config(
            text=f"Archivo cargado:\n{ruta}"
        )

        analizar_audio()

    except Exception as e:
        messagebox.showerror(
            "Error",
            f"No se pudo cargar el audio:\n{e}"
        )

    finally:

        if ruta_wav and os.path.exists(ruta_wav):
            try:
                os.remove(ruta_wav)
            except:
                pass


def analizar_audio():

    global audio, sr


    fft = np.abs(np.fft.rfft(audio))
    frecuencias = np.fft.rfftfreq(len(audio), 1/sr)


    centroide = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    promedio_centroide = np.mean(centroide)


    banda_baja  = np.logical_and(frecuencias >= 80,   frecuencias <= 500)
    banda_media = np.logical_and(frecuencias > 500,   frecuencias <= 4000)
    banda_alta  = np.logical_and(frecuencias > 4000,  frecuencias <= 8000)

    energia_baja  = np.sum(fft[banda_baja]**2)
    energia_media = np.sum(fft[banda_media]**2)
    energia_alta  = np.sum(fft[banda_alta]**2)


    f0, _, _ = librosa.pyin(audio, fmin=50, fmax=500)
    f0_limpia = f0[~np.isnan(f0)]

    if len(f0_limpia) > 0:
        promedio_f0   = np.mean(f0_limpia)
        desviacion_f0 = np.std(f0_limpia)
    else:
        promedio_f0   = 0
        desviacion_f0 = 0


    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_promedio = np.mean(mfcc, axis=1)


    energia_total = np.sum(audio**2)
    ruido = audio - librosa.effects.preemphasis(audio)
    energia_ruido = np.sum(ruido**2)

    hnr = 10 * np.log10(energia_total / energia_ruido) if energia_ruido != 0 else 0


    texto.delete("1.0", tk.END)
    texto.insert(tk.END, "========== RESULTADOS ==========\n\n")
    texto.insert(tk.END, f"Frecuencia de muestreo: {sr} Hz\n\n")
    texto.insert(tk.END, f"Centroide espectral: {promedio_centroide:.2f} Hz\n\n")
    texto.insert(tk.END, f"Energía baja:  {energia_baja:.2f}\n")
    texto.insert(tk.END, f"Energía media: {energia_media:.2f}\n")
    texto.insert(tk.END, f"Energía alta:  {energia_alta:.2f}\n\n")
    texto.insert(tk.END, f"F0 promedio:   {promedio_f0:.2f} Hz\n")
    texto.insert(tk.END, f"Variación F0:  {desviacion_f0:.2f}\n\n")
    texto.insert(tk.END, f"HNR: {hnr:.2f} dB\n\n")
    texto.insert(tk.END, "========== MFCC ==========\n\n")

    for i, valor in enumerate(mfcc_promedio):
        texto.insert(tk.END, f"MFCC {i+1}: {valor:.2f}\n")


    figura.clear()

    ax1 = figura.add_subplot(311)
    librosa.display.waveshow(audio, sr=sr, ax=ax1)
    ax1.set_title("Señal de Audio")

    ax2 = figura.add_subplot(312)
    ax2.plot(frecuencias, fft)
    ax2.set_title("FFT")

    ax3 = figura.add_subplot(313)
    librosa.display.specshow(mfcc, x_axis='time', ax=ax3)
    ax3.set_title("MFCC")

    figura.tight_layout()
    canvas.draw()


ventana = tk.Tk()
ventana.title("Analizador de Audio IA")
ventana.geometry("1300x900")

boton = tk.Button(
    ventana,
    text="Cargar Audio",
    font=("Arial", 15),
    command=cargar_audio
)
boton.pack(pady=10)

label_archivo = tk.Label(ventana, text="Ningún archivo cargado")
label_archivo.pack()

texto = tk.Text(ventana, width=90, height=18, font=("Consolas", 11))
texto.pack(pady=10)

figura = plt.Figure(figsize=(10, 8), dpi=100)
canvas = FigureCanvasTkAgg(figura, master=ventana)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

ventana.mainloop()