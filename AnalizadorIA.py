import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import subprocess
import tempfile
import os



UMBRAL_F0          = (148.3 + 175.6) / 2     
UMBRAL_F0_STD      = (27.5  + 21.7)  / 2      
UMBRAL_JITTER      = (3.844 + 3.047) / 2      
UMBRAL_PENDIENTE   = (-0.0017 + 0.0051) / 2   
UMBRAL_HNR_REAL    = 0.10                      
UMBRAL_RATIO_SIL   = 0.35                      


PESO_F0         = 0.35
PESO_STD_F0     = 0.20
PESO_JITTER     = 0.20
PESO_PENDIENTE  = 0.15
PESO_HNR_REAL   = 0.10


audios = {1: None, 2: None}
srs    = {1: None, 2: None}
rutas  = {1: "", 2: ""}


def convertir_a_wav(ruta_audio):
    archivo_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    ruta_wav = archivo_temp.name
    archivo_temp.close()
    comando = [
        "ffmpeg", "-i", ruta_audio,
        "-ar", "16000", "-ac", "1",
        ruta_wav, "-y"
    ]
    resultado = subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if resultado.returncode != 0:
        raise RuntimeError(f"ffmpeg falló:\n{resultado.stderr.decode()}")
    if not os.path.exists(ruta_wav):
        raise FileNotFoundError(f"No se generó el WAV en: {ruta_wav}")
    return ruta_wav


def calcular_hnr_autocorr(audio_seg, sr_local, fmin=50, fmax=500):
    ventana  = int(sr_local * 0.04)  
    salto    = int(sr_local * 0.01)   
    hnr_vals = []

    lag_min = int(sr_local / fmax)
    lag_max = int(sr_local / fmin)

    for inicio in range(0, len(audio_seg) - ventana, salto):
        seg = audio_seg[inicio: inicio + ventana]
        seg = seg - np.mean(seg)
        if np.max(np.abs(seg)) < 1e-6:
            continue

        
        acorr = np.correlate(seg, seg, mode='full')
        acorr = acorr[len(acorr)//2:]
        acorr = acorr / (acorr[0] + 1e-10)

        
        if lag_max >= len(acorr):
            continue
        zona   = acorr[lag_min:lag_max]
        pico   = np.max(zona)
        pico   = np.clip(pico, 0, 0.9999)

        
        hnr = 10 * np.log10(pico / (1 - pico + 1e-10))
        hnr_vals.append(hnr)

    return np.median(hnr_vals) if hnr_vals else 0.0


def calcular_jitter(f0_limpia):
    if len(f0_limpia) < 2:
        return 0.0
    diffs = np.abs(np.diff(f0_limpia))
    media = np.mean(f0_limpia)
    return (np.mean(diffs) / media) * 100 if media > 0 else 0.0


def calcular_pendiente_f0(f0_limpia):
    if len(f0_limpia) < 4:
        return 0.0
    x = np.arange(len(f0_limpia))
    m, _ = np.polyfit(x, f0_limpia, 1)
    return m


def calcular_ratio_silencio(audio_seg, sr_local):
    hop   = int(sr_local * 0.010)
    frame = int(sr_local * 0.025)
    rms   = librosa.feature.rms(y=audio_seg, frame_length=frame, hop_length=hop)[0]
    umbral = np.max(rms) * 0.05
    frames_silencio = np.sum(rms < umbral)
    return frames_silencio / len(rms) if len(rms) > 0 else 0.0


def calcular_cv_ritmo(audio_seg, sr_local):
    hop   = int(sr_local * 0.010)
    frame = int(sr_local * 0.025)
    rms   = librosa.feature.rms(y=audio_seg, frame_length=frame, hop_length=hop)[0]
    umbral_rms = np.mean(rms) * 0.5
    activo     = (rms > umbral_rms).astype(int)
    cambios    = np.where(np.diff(activo) == 1)[0]
    if len(cambios) < 3:
        return 0.5
    intervalos = np.diff(cambios) * (hop / sr_local)
    if np.mean(intervalos) == 0:
        return 0.5
    return np.std(intervalos) / np.mean(intervalos)


def clasificar(f0_media, f0_std, jitter, pendiente, hnr_real, cv_ritmo):
    score_ia = 0.0
    razones  = []

  
    if f0_media > 0 and f0_media < UMBRAL_F0:
        score_ia += PESO_F0
        razones.append(
            f"F0={f0_media:.1f} Hz  < {UMBRAL_F0:.1f} Hz -> tono bajo -> IA"
        )
    elif f0_media >= UMBRAL_F0:
        razones.append(
            f"F0={f0_media:.1f} Hz  ≥ {UMBRAL_F0:.1f} Hz -> tono alto -> -> Humano"
        )
    else:
        score_ia += PESO_F0 * 0.5
        razones.append("F0=no detectado -> neutro")

   
    if f0_std > UMBRAL_F0_STD:
        score_ia += PESO_STD_F0
        razones.append(
            f"F0 STD={f0_std:.2f} Hz  > {UMBRAL_F0_STD:.1f} Hz -> variación alta -> IA"
        )
    else:
        razones.append(
            f"F0 STD={f0_std:.2f} Hz  ≤ {UMBRAL_F0_STD:.1f} Hz -> variación natural -> Humano"
        )

    
    if jitter > UMBRAL_JITTER:
        score_ia += PESO_JITTER
        razones.append(
            f"Jitter={jitter:.3f}%  > {UMBRAL_JITTER:.2f}% -> ciclos irregulares -> IA"
        )
    else:
        razones.append(
            f"Jitter={jitter:.3f}%  ≤ {UMBRAL_JITTER:.2f}% -> ciclos regulares -> Humano"
        )

   
    if pendiente < UMBRAL_PENDIENTE:
        score_ia += PESO_PENDIENTE
        razones.append(
            f"Pendiente F0={pendiente:.4f} < {UMBRAL_PENDIENTE:.4f} -> prosodia plana/descendente -> IA"
        )
    else:
        razones.append(
            f"Pendiente F0={pendiente:.4f} ≥ {UMBRAL_PENDIENTE:.4f} -> prosodia ascendente -> Humano"
        )

    
    if hnr_real < UMBRAL_HNR_REAL:
        score_ia += PESO_HNR_REAL
        razones.append(
            f"HNR_real={hnr_real:.2f} dB  < {UMBRAL_HNR_REAL:.2f} -> baja armonía -> IA"
        )
    else:
        razones.append(
            f"HNR_real={hnr_real:.2f} dB  ≥ {UMBRAL_HNR_REAL:.2f} -> alta armonía -> Humano"
        )

    
    es_ia = score_ia >= 0.50
    categoria = determinar_categoria(f0_media, es_ia)

    return score_ia, categoria, razones


def clasificar_ventana_score(v):
    s = 0.0
    if v["f0_media"] > 0 and v["f0_media"] < UMBRAL_F0:   s += PESO_F0
    elif v["f0_media"] == 0:                                s += PESO_F0 * 0.5
    if v["f0_std"]   > UMBRAL_F0_STD:                      s += PESO_STD_F0
    if v["jitter"]   > UMBRAL_JITTER:                      s += PESO_JITTER
    if v["pendiente"]< UMBRAL_PENDIENTE:                   s += PESO_PENDIENTE
    if v["hnr_real"] < UMBRAL_HNR_REAL:                    s += PESO_HNR_REAL
    return s



def determinar_categoria(f0_media, es_ia):
    if f0_media == 0:
        return "IA (género indefinido)" if es_ia else "Humano (género indefinido)"
    if es_ia:
        return "IA Hombre" if f0_media <= 165 else "IA Mujer"
    else:
        return "Humano Hombre" if f0_media <= 185 else "Humano Mujer"



def analizar_por_ventanas(audio_seg, sr_local, tam_s=2.0):
    tam = int(sr_local * tam_s)
    n   = len(audio_seg) // tam
    resultados = []

    for i in range(n):
        seg = audio_seg[i * tam: (i + 1) * tam]

        f0, _, _ = librosa.pyin(seg, fmin=50, fmax=500)
        f0_ok    = f0[~np.isnan(f0)]
        f0_media = float(np.mean(f0_ok))   if len(f0_ok) > 0 else 0.0
        f0_std   = float(np.std(f0_ok))    if len(f0_ok) > 0 else 0.0
        jitter   = calcular_jitter(f0_ok)
        pendiente= calcular_pendiente_f0(f0_ok)
        hnr_real = calcular_hnr_autocorr(seg, sr_local)
        cv_rit   = calcular_cv_ritmo(seg, sr_local)

        resultados.append({
            "ventana":   i + 1,
            "t_inicio":  i * tam_s,
            "t_fin":     (i + 1) * tam_s,
            "f0_media":  f0_media,
            "f0_std":    f0_std,
            "jitter":    jitter,
            "pendiente": pendiente,
            "hnr_real":  hnr_real,
            "cv_ritmo":  cv_rit,
        })

    return resultados



def pipeline(audio_raw, sr_local):
    
    frecuencias  = np.fft.rfftfreq(len(audio_raw), 1 / sr_local)
    mascara_voz  = (frecuencias >= 80) & (frecuencias <= 8000)
    espectro     = np.fft.rfft(audio_raw)
    audio_limpio = np.fft.irfft(espectro * mascara_voz.astype(float))[:len(audio_raw)]

    
    f0, _, _   = librosa.pyin(audio_limpio, fmin=50, fmax=500)
    f0_ok      = f0[~np.isnan(f0)]
    f0_media   = float(np.mean(f0_ok))  if len(f0_ok) > 0 else 0.0
    f0_std     = float(np.std(f0_ok))   if len(f0_ok) > 0 else 0.0
    jitter     = calcular_jitter(f0_ok)
    pendiente  = calcular_pendiente_f0(f0_ok)
    hnr_real   = calcular_hnr_autocorr(audio_limpio, sr_local)
    cv_ritmo   = calcular_cv_ritmo(audio_limpio, sr_local)
    ratio_sil  = calcular_ratio_silencio(audio_limpio, sr_local)

    score_ia, categoria, razones = clasificar(
        f0_media, f0_std, jitter, pendiente, hnr_real, cv_ritmo
    )

    ventanas = analizar_por_ventanas(audio_limpio, sr_local, tam_s=2.0)

    return {
        "audio_limpio": audio_limpio,
        "f0":           f0,
        "f0_ok":        f0_ok,
        "f0_media":     f0_media,
        "f0_std":       f0_std,
        "jitter":       jitter,
        "pendiente":    pendiente,
        "hnr_real":     hnr_real,
        "cv_ritmo":     cv_ritmo,
        "ratio_sil":    ratio_sil,
        "score_ia":     score_ia,
        "categoria":    categoria,
        "razones":      razones,
        "ventanas":     ventanas,
        "sr":           sr_local,
    }


def cargar_audio(n):
    ruta = filedialog.askopenfilename(
        title=f"Seleccionar Audio {n}",
        filetypes=[("Audios", "*.m4a *.mp4 *.wav *.mp3")]
    )
    if ruta == "":
        return
    ruta_wav = None
    try:
        ruta_wav = convertir_a_wav(ruta)
        audio_raw, sr_local = librosa.load(ruta_wav, sr=16000, mono=True)
        audios[n] = audio_raw
        srs[n]    = sr_local
        rutas[n]  = os.path.basename(ruta)
        labels[n].config(text=f"Audio {n}: {rutas[n]}")
        messagebox.showinfo("Cargado", f"Audio {n} cargado: {rutas[n]}")
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo cargar el audio {n}:\n{e}")
    finally:
        if ruta_wav and os.path.exists(ruta_wav):
            try:
                os.remove(ruta_wav)
            except:
                pass



def comparar():
    if audios[1] is None or audios[2] is None:
        messagebox.showwarning("Faltan audios", "Debes cargar los 2 audios antes de comparar.")
        return

    r1 = pipeline(audios[1], srs[1])
    r2 = pipeline(audios[2], srs[2])

    mostrar_texto(r1, r2)
    mostrar_graficas(r1, r2)



def mostrar_texto(r1, r2):
    texto.delete("1.0", tk.END)

    for idx, (r, nom) in enumerate([(r1, rutas[1]), (r2, rutas[2])], start=1):
        pct_ia  = r["score_ia"] * 100
        pct_hum = (1 - r["score_ia"]) * 100

        if r["score_ia"] >= 0.55:
            veredicto = f"VOZ DE IA     (score IA: {pct_ia:.0f}%)"
            confianza = "ALTA" if r["score_ia"] >= 0.75 else "MEDIA"
        elif r["score_ia"] <= 0.45:
            veredicto = f"VOZ HUMANA    (score Humano: {pct_hum:.0f}%)"
            confianza = "ALTA" if r["score_ia"] <= 0.25 else "MEDIA"
        else:
            veredicto = f"INDEFINIDO    (IA {pct_ia:.0f}% / Humano {pct_hum:.0f}%)"
            confianza = "BAJA"

        texto.insert(tk.END, "═" * 65 + "\n")
        texto.insert(tk.END, f"  AUDIO {idx}: {nom}\n")
        texto.insert(tk.END, f"  {veredicto}\n")
        texto.insert(tk.END, f"  Categoría : {r['categoria']}\n")
        texto.insert(tk.END, f"  Confianza : {confianza}\n")
        texto.insert(tk.END, "═" * 65 + "\n\n")

        texto.insert(tk.END, "  CRITERIOS CALIBRADOS CON TUS DATOS REALES:\n")
        for razon in r["razones"]:
            texto.insert(tk.END, f"    • {razon}\n")

        texto.insert(tk.END, "\n  MÉTRICAS GLOBALES:\n")
        texto.insert(tk.END, f"    F0 media     : {r['f0_media']:.2f} Hz    (umbral: {UMBRAL_F0:.1f} Hz)\n")
        texto.insert(tk.END, f"    F0 STD       : {r['f0_std']:.2f} Hz    (umbral: {UMBRAL_F0_STD:.1f} Hz)\n")
        texto.insert(tk.END, f"    Jitter       : {r['jitter']:.3f} %     (umbral: {UMBRAL_JITTER:.2f} %)\n")
        texto.insert(tk.END, f"    Pendiente F0 : {r['pendiente']:.5f} Hz/frame  (umbral: {UMBRAL_PENDIENTE:.5f})\n")
        texto.insert(tk.END, f"    HNR real     : {r['hnr_real']:.3f} dB   (umbral: {UMBRAL_HNR_REAL:.2f} dB)\n")
        texto.insert(tk.END, f"    CV ritmo     : {r['cv_ritmo']:.4f}\n")
        texto.insert(tk.END, f"    Ratio silencio: {r['ratio_sil']:.3f}\n")

        texto.insert(tk.END, f"\n  ANÁLISIS POR VENTANAS (2 s c/u):\n")
        texto.insert(tk.END,
            f"    {'V':>3}  {'t(s)':>9}  {'F0':>6}  {'STD':>6}  "
            f"{'Jit%':>6}  {'Pend':>7}  {'HNRr':>7}  {'Score':>5}  {'Categ.'}\n")
        texto.insert(tk.END, "    " + "-" * 80 + "\n")

        for v in r["ventanas"]:
            sv    = clasificar_ventana_score(v)
            cat_v = determinar_categoria(v["f0_media"], sv >= 0.50)
            texto.insert(tk.END,
                f"    {v['ventana']:>3}  "
                f"{v['t_inicio']:>4.1f}–{v['t_fin']:>4.1f}  "
                f"{v['f0_media']:>6.1f}  "
                f"{v['f0_std']:>6.1f}  "
                f"{v['jitter']:>6.3f}  "
                f"{v['pendiente']:>+7.4f}  "
                f"{v['hnr_real']:>7.2f}  "
                f"{sv:>5.2f}  "
                f"{cat_v}\n"
            )

        texto.insert(tk.END, "\n\n")

    # ---- Comparación directa ----
    texto.insert(tk.END, "═" * 65 + "\n")
    texto.insert(tk.END, "  COMPARACIÓN DIRECTA\n")
    texto.insert(tk.END, "═" * 65 + "\n")

    cat1_ia = r1["score_ia"] >= 0.50
    cat2_ia = r2["score_ia"] >= 0.50

    if cat1_ia != cat2_ia:
        ia_nom  = rutas[1] if cat1_ia else rutas[2]
        hum_nom = rutas[2] if cat1_ia else rutas[1]
        texto.insert(tk.END, f"  VOZ DE IA    -> {ia_nom}\n")
        texto.insert(tk.END, f"  VOZ HUMANA   -> {hum_nom}\n")
    elif cat1_ia and cat2_ia:
        texto.insert(tk.END, "  Ambos clasificados como VOZ DE IA\n")
        mas = rutas[1] if r1["score_ia"] >= r2["score_ia"] else rutas[2]
        texto.insert(tk.END, f"  El más IA: {mas}\n")
    else:
        texto.insert(tk.END, "  Ambos clasificados como VOZ HUMANA\n")
        mas = rutas[1] if r1["score_ia"] <= r2["score_ia"] else rutas[2]
        texto.insert(tk.END, f"  El más humano: {mas}\n")

    texto.insert(tk.END, "\n")
    texto.insert(tk.END, f"  F0 media     : {abs(r1['f0_media']-r2['f0_media']):.1f} Hz\n")
    texto.insert(tk.END, f"  F0 STD       : {abs(r1['f0_std']-r2['f0_std']):.2f} Hz\n")
    texto.insert(tk.END, f"  Jitter       : {abs(r1['jitter']-r2['jitter']):.3f} %\n")
    texto.insert(tk.END, f"  Pendiente F0 : {abs(r1['pendiente']-r2['pendiente']):.5f} Hz/frame\n")
    texto.insert(tk.END, f"  HNR real     : {abs(r1['hnr_real']-r2['hnr_real']):.3f} dB\n")
    texto.insert(tk.END, "\n  REFERENCIA DE UMBRALES (calibrados con tus audios):\n")
    texto.insert(tk.END, f"    F0      : {UMBRAL_F0:.1f} Hz   (IA 148↔Humano 176)\n")
    texto.insert(tk.END, f"    STD F0  : {UMBRAL_F0_STD:.1f} Hz  (IA 27.5↔Humano 21.7)\n")
    texto.insert(tk.END, f"    Jitter  : {UMBRAL_JITTER:.2f}%   (IA 3.84↔Humano 3.05)\n")
    texto.insert(tk.END, f"    Pendiente: {UMBRAL_PENDIENTE:.5f}  (IA -0.0017↔Humano +0.0051)\n")



def mostrar_graficas(r1, r2):
    figura.clear()
    figura.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(3, 2, figure=figura, hspace=0.50, wspace=0.35)

    col1 = "#3a86ff"
    col2 = "#ff006e"

    estilo = {"facecolor": "#161b22", "labelcolor": "#c9d1d9"}

    def estilizar(ax):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#aaa", labelsize=7)
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
        ax.title.set_color("#e0e0e0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    
    ax1 = figura.add_subplot(gs[0, 0])
    librosa.display.waveshow(r1["audio_limpio"], sr=r1["sr"], ax=ax1, color=col1, alpha=0.85)
    ax1.set_title(f"Audio 1: {rutas[1][:35]}\n-> {r1['categoria']}", fontsize=8)
    estilizar(ax1)

    ax2 = figura.add_subplot(gs[0, 1])
    librosa.display.waveshow(r2["audio_limpio"], sr=r2["sr"], ax=ax2, color=col2, alpha=0.85)
    ax2.set_title(f"Audio 2: {rutas[2][:35]}\n-> {r2['categoria']}", fontsize=8)
    estilizar(ax2)


    for ax, r, col, nom in [
        (figura.add_subplot(gs[1, 0]), r1, col1, "Audio 1"),
        (figura.add_subplot(gs[1, 1]), r2, col2, "Audio 2"),
    ]:
        t = librosa.times_like(r["f0"], sr=r["sr"])
        ax.plot(t, r["f0"], color=col, linewidth=1, label=f"F0  μ={r['f0_media']:.1f} Hz")
        ax.axhline(r["f0_media"], color=col, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(UMBRAL_F0, color="#ffd700", linestyle=":", linewidth=1.2,
                   label=f"Umbral {UMBRAL_F0:.0f} Hz")
        if len(t) > 1:
            ax.fill_between([t[0], t[-1]], 0, UMBRAL_F0, alpha=0.07, color="#f44336")
        ax.set_ylim(0, 400)
        ax.set_ylabel("Hz", fontsize=7)
        ax.set_title(f"F0 — {nom}  (STD={r['f0_std']:.1f} Hz)", fontsize=8)
        ax.legend(fontsize=6)
        estilizar(ax)


    for ax, r, col, nom in [
        (figura.add_subplot(gs[2, 0]), r1, col1, "Audio 1"),
        (figura.add_subplot(gs[2, 1]), r2, col2, "Audio 2"),
    ]:
        _graficar_ventanas(ax, r, col, f"Score IA por ventana — {nom}")
        estilizar(ax)

    canvas.draw()


def _graficar_ventanas(ax, r, color_linea, titulo):
    ventanas = r["ventanas"]
    if not ventanas:
        ax.text(0.5, 0.5, "Muy corto\npara ventanas", ha="center", va="center",
                color="#aaa")
        ax.set_title(titulo, fontsize=8)
        return

    xs     = [v["ventana"] for v in ventanas]
    scores = [clasificar_ventana_score(v) for v in ventanas]
    colores= ["#f44336" if s >= 0.50 else "#4caf50" for s in scores]

    ax.bar(xs, scores, color=colores, alpha=0.85, width=0.6)
    ax.axhline(0.50, color="#ffd700", linestyle="--", linewidth=1.2,
               label="Umbral 0.50")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"V{x}" for x in xs], fontsize=6, rotation=45)
    ax.set_ylabel("Score IA", fontsize=7)
    ax.set_title(titulo, fontsize=8)
    ax.legend(fontsize=6)

    for x, s in zip(xs, scores):
        etq = "IA" if s >= 0.50 else "Hum"
        ax.text(x, s + 0.02, etq, ha="center", va="bottom", fontsize=5,
                color="#f44336" if s >= 0.50 else "#4caf50", fontweight="bold")



ventana = tk.Tk()
ventana.title("Comparador de Voz — Humano vs IA  (recalibrado con datos reales)")
ventana.geometry("1500x980")
ventana.configure(bg="#1a1a2e")

frame_top = tk.Frame(ventana, bg="#1a1a2e")
frame_top.pack(pady=10, fill=tk.X, padx=20)

tk.Label(
    frame_top,
    text="🎙 COMPARADOR VOZ HUMANA vs IA  —  recalibrado",
    font=("Courier New", 15, "bold"),
    fg="#e0e0e0", bg="#1a1a2e"
).pack()

tk.Label(
    frame_top,
    text=f"Criterios: F0 (umbral {UMBRAL_F0:.0f} Hz) | STD F0 (umbral {UMBRAL_F0_STD:.0f} Hz) | "
         f"Jitter (umbral {UMBRAL_JITTER:.2f}%) | Pendiente F0 | HNR autocorr",
    font=("Courier New", 8),
    fg="#888", bg="#1a1a2e"
).pack()

frame_btns = tk.Frame(ventana, bg="#1a1a2e")
frame_btns.pack(pady=8)

labels = {}

for n, color_btn in [(1, "#3a86ff"), (2, "#ff006e")]:
    sub = tk.Frame(frame_btns, bg="#1a1a2e")
    sub.pack(side=tk.LEFT, padx=25)
    tk.Button(
        sub,
        text=f"   Cargar Audio {n}  ",
        font=("Courier New", 12, "bold"),
        bg=color_btn, fg="white",
        relief=tk.FLAT, padx=10, pady=6,
        command=lambda x=n: cargar_audio(x)
    ).pack()
    lbl = tk.Label(sub, text=f"Audio {n}: sin cargar",
                   font=("Courier New", 8), fg="#aaa", bg="#1a1a2e")
    lbl.pack(pady=2)
    labels[n] = lbl

tk.Button(
    frame_btns,
    text="  COMPARAR  ",
    font=("Courier New", 13, "bold"),
    bg="#06d6a0", fg="#1a1a2e",
    relief=tk.FLAT, padx=15, pady=8,
    command=comparar
).pack(side=tk.LEFT, padx=30)

frame_txt = tk.Frame(ventana, bg="#1a1a2e")
frame_txt.pack(fill=tk.BOTH, padx=20, pady=(0, 5))

texto = tk.Text(
    frame_txt, width=130, height=18,
    font=("Courier New", 9),
    bg="#0d1117", fg="#c9d1d9",
    insertbackground="white",
    relief=tk.FLAT, padx=8, pady=6
)
scroll = tk.Scrollbar(frame_txt, command=texto.yview)
texto.configure(yscrollcommand=scroll.set)
scroll.pack(side=tk.RIGHT, fill=tk.Y)
texto.pack(fill=tk.BOTH, expand=True)

figura = plt.Figure(figsize=(14, 7), dpi=95)
figura.patch.set_facecolor("#0d1117")
canvas = FigureCanvasTkAgg(figura, master=ventana)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

ventana.mainloop()