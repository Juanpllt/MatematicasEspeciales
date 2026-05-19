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

    N = len(audio)


    Yk = np.fft.rfft(audio)
    frecuencias = np.fft.rfftfreq(N, 1/sr)

    print("\nPrimeros coeficientes de la FFT Y[k]:")
    for i, val in enumerate(Yk[:10]):
        print(f"  Y[{i}] = {val:.6f}")


    delta_f = sr / N
    k_min = int(np.ceil(80   / delta_f))
    k_max = int(np.floor(8000 / delta_f))

    print(f"\n{'='*55}")
    print(f"  FILTRADO ESPECTRAL")
    print(f"  Resolución: {delta_f:.4f} Hz/cajón")
    print(f"  k_min = {k_min}  →  f = {k_min * delta_f:.2f} Hz")
    print(f"  k_max = {k_max}  →  f = {k_max * delta_f:.2f} Hz")
    print(f"{'='*55}")


    Yk_filtrado = np.zeros_like(Yk)
    Yk_filtrado[k_min:k_max+1] = Yk[k_min:k_max+1]


    audio_limpio = np.fft.irfft(Yk_filtrado, n=N)


    UMBRAL = 1e-5
    indices_activos = np.where(np.abs(audio_limpio) > UMBRAL)[0]

    print(f"\n{'='*55}")
    print(f"  MUESTRAS x_limpia[n]")
    print(f"  Total de muestras: N = {N}")
    print(f"{'='*55}")

    if len(indices_activos) >= 3:
        print(f"\n  {'n':>8}  |  {'x_limpia[n]':>14}  |  {'Tiempo (ms)':>12}")
        print(f"  {'-'*8}  |  {'-'*14}  |  {'-'*12}")

        print("\n  -- Primeras 3 muestras activas --")
        for n in indices_activos[:3]:
            t_ms = (n / sr) * 1000
            print(f"  {n:>8}  |  {audio_limpio[n]:>14.8f}  |  {t_ms:>10.4f} ms")

        print("\n  -- Últimas 3 muestras activas --")
        for n in indices_activos[-3:]:
            t_ms = (n / sr) * 1000
            print(f"  {n:>8}  |  {audio_limpio[n]:>14.8f}  |  {t_ms:>10.4f} ms")

        n_ini = indices_activos[0]
        n_fin = indices_activos[-1]
        duracion_ms = (n_fin - n_ini) / sr * 1000
        print(f"\n  Silencio inicial: x[0] a x[{n_ini-1}]  ({n_ini} muestras = {n_ini/sr*1000:.2f} ms)")
        print(f"  Silencio final:   x[{n_fin+1}] a x[{N-1}]  ({N-1-n_fin} muestras = {(N-1-n_fin)/sr*1000:.2f} ms)")
        print(f"  Duración activa:  {duracion_ms:.2f} ms")
    else:
        duracion_ms = N / sr * 1000

    print(f"{'='*55}\n")



    fft_limpia    = np.abs(Yk_filtrado)
    frec_limpia   = frecuencias


    duracion_seg  = N / sr


    centroide     = librosa.feature.spectral_centroid(y=audio_limpio, sr=sr)[0]
    prom_centroide = np.mean(centroide)


    banda_baja  = np.logical_and(frec_limpia >= 80,   frec_limpia <= 500)
    banda_media = np.logical_and(frec_limpia > 500,   frec_limpia <= 4000)
    banda_alta  = np.logical_and(frec_limpia > 4000,  frec_limpia <= 8000)

    energia_baja  = np.sum(fft_limpia[banda_baja]**2)
    energia_media = np.sum(fft_limpia[banda_media]**2)
    energia_alta  = np.sum(fft_limpia[banda_alta]**2)


    f0, _, _      = librosa.pyin(audio_limpio, fmin=50, fmax=500)
    f0_limpia_arr = f0[~np.isnan(f0)]

    if len(f0_limpia_arr) > 0:
        promedio_f0   = np.mean(f0_limpia_arr)
        desviacion_f0 = np.std(f0_limpia_arr)
    else:
        promedio_f0   = 0
        desviacion_f0 = 0


    mfcc          = librosa.feature.mfcc(y=audio_limpio, sr=sr, n_mfcc=13)
    mfcc_promedio = np.mean(mfcc, axis=1)


    energia_total = np.sum(audio_limpio**2)
    ruido_hnr     = audio_limpio - librosa.effects.preemphasis(audio_limpio)
    energia_ruido = np.sum(ruido_hnr**2)
    hnr = 10 * np.log10(energia_total / energia_ruido) if energia_ruido != 0 else 0


    print("="*80)
    print("FILA DE DATOS (copia esto a tu tabla):")
    print("="*80)
    encabezado = (
        "Duracion\tFrecuenciaMuestreo\tCentroideEspectral\t"
        "EnergiaBandaBaja\tEnergiaBandaMedia\tEnergiaBandaAlta\t"
        "F_0Promedio\tVariacionF_0\tHNR\t"
        "MFCC1\tMFCC2\tMFCC3\tMFCC4\tMFCC5\tMFCC6\tMFCC7\t"
        "MFCC8\tMFCC9\tMFCC10\tMFCC11\tMFCC12\tMFCC13"
    )
    fila = (
            f"{duracion_seg:.4f}\t{sr}\t{prom_centroide:.4f}\t"
            f"{energia_baja:.4f}\t{energia_media:.4f}\t{energia_alta:.4f}\t"
            f"{promedio_f0:.4f}\t{desviacion_f0:.4f}\t{hnr:.4f}\t"
            + "\t".join(f"{v:.4f}" for v in mfcc_promedio)
    )
    print(encabezado)
    print(fila)
    print("="*80 + "\n")


    texto.delete("1.0", tk.END)
    texto.insert(tk.END, "===== SEÑAL LIMPIA (tras filtrado + IFFT) =====\n\n")
    texto.insert(tk.END, f"Frecuencia de muestreo: {sr} Hz\n")
    texto.insert(tk.END, f"Duración:               {duracion_seg:.4f} s\n\n")
    texto.insert(tk.END, f"Centroide espectral:    {prom_centroide:.2f} Hz\n\n")
    texto.insert(tk.END, f"Energia baja:           {energia_baja:.2f}\n")
    texto.insert(tk.END, f"Energia media:          {energia_media:.2f}\n")
    texto.insert(tk.END, f"Energia alta:           {energia_alta:.2f}\n\n")
    texto.insert(tk.END, f"F0 promedio:            {promedio_f0:.2f} Hz\n")
    texto.insert(tk.END, f"Variacion F0:           {desviacion_f0:.2f}\n\n")
    texto.insert(tk.END, f"HNR:                    {hnr:.2f} dB\n\n")

    texto.insert(tk.END, "========== MFCC (señal limpia) ==========\n\n")
    for i, valor in enumerate(mfcc_promedio):
        texto.insert(tk.END, f"MFCC {i+1:>2}: {valor:.2f}\n")

    texto.insert(tk.END, "\n========== FILTRADO ==========\n\n")
    texto.insert(tk.END, f"Resolución espectral:   {delta_f:.4f} Hz/cajón\n")
    texto.insert(tk.END, f"k_min = {k_min}  →  {k_min*delta_f:.2f} Hz\n")
    texto.insert(tk.END, f"k_max = {k_max}  →  {k_max*delta_f:.2f} Hz\n\n")

    if len(indices_activos) >= 3:
        texto.insert(tk.END, "Primeras 3 muestras activas x_limpia[n]:\n\n")
        texto.insert(tk.END, f"  {'n':>8}  |  {'Tiempo (ms)':>11}  |  {'x_limpia[n]':>14}\n")
        texto.insert(tk.END, f"  {'-'*8}  |  {'-'*11}  |  {'-'*14}\n")
        for n in indices_activos[:3]:
            t_ms = (n / sr) * 1000
            texto.insert(tk.END,
                         f"  {n:>8}  |  {t_ms:>9.4f} ms  |  {audio_limpio[n]:>14.8f}\n")

        texto.insert(tk.END, "\nUltimas 3 muestras activas x_limpia[n]:\n\n")
        texto.insert(tk.END, f"  {'n':>8}  |  {'Tiempo (ms)':>11}  |  {'x_limpia[n]':>14}\n")
        texto.insert(tk.END, f"  {'-'*8}  |  {'-'*11}  |  {'-'*14}\n")
        for n in indices_activos[-3:]:
            t_ms = (n / sr) * 1000
            texto.insert(tk.END,
                         f"  {n:>8}  |  {t_ms:>9.4f} ms  |  {audio_limpio[n]:>14.8f}\n")


    figura.clear()

    ax1 = figura.add_subplot(411)
    librosa.display.waveshow(audio, sr=sr, ax=ax1, color="steelblue")
    ax1.set_title("Señal original  x[n]")
    ax1.set_xlabel("")

    ax2 = figura.add_subplot(412)
    ax2.plot(frecuencias, np.abs(Yk), color="darkorange", linewidth=0.8, label="Original")
    ax2.plot(frecuencias, fft_limpia,  color="red",        linewidth=0.8, label="Filtrada", alpha=0.7)
    ax2.set_title("FFT  |Y[k]|  —  Original vs Filtrada")
    ax2.set_xlabel("Frecuencia (Hz)")
    ax2.set_ylabel("Amplitud")
    ax2.legend(fontsize=8)

    ax3 = figura.add_subplot(413)
    librosa.display.waveshow(audio_limpio, sr=sr, ax=ax3, color="mediumseagreen")
    ax3.set_title("Señal limpia  x_limpia[n]  =  IFFT{ F_filtrada[k] }")
    ax3.set_xlabel("")

    ax4 = figura.add_subplot(414)
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax4)
    ax4.set_title("MFCC (señal limpia)")
    figura.colorbar(img, ax=ax4, format="%+2.f")

    figura.tight_layout()
    canvas.draw()


ventana = tk.Tk()
ventana.title("Analizador de Audio IA  —  Filtrado + IFFT")
ventana.geometry("1300x950")

boton = tk.Button(
    ventana,
    text="Cargar Audio",
    font=("Arial", 15),
    command=cargar_audio
)
boton.pack(pady=10)

label_archivo = tk.Label(ventana, text="Ningun archivo cargado")
label_archivo.pack()

texto = tk.Text(ventana, width=90, height=20, font=("Consolas", 11))
texto.pack(pady=10)

figura = plt.Figure(figsize=(10, 9), dpi=100)
canvas = FigureCanvasTkAgg(figura, master=ventana)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

ventana.mainloop()