

import numpy as np
import librosa
import librosa.feature

from domain.ports import AudioAnalyzerPort


class LibrosaAnalyzer(AudioAnalyzerPort):
    """Extracción de métricas acústicas mediante librosa."""


    def extract_f0(self, audio: np.ndarray, sr: int) -> np.ndarray:
        f0, _, _ = librosa.pyin(audio, fmin=50, fmax=500)
        return f0                                  


    def calculate_jitter(self, f0_ok: np.ndarray) -> float:
        if len(f0_ok) < 2:
            return 0.0
        diffs = np.abs(np.diff(f0_ok))
        media = np.mean(f0_ok)
        return float((np.mean(diffs) / media) * 100) if media > 0 else 0.0


    def calculate_hnr(self, audio: np.ndarray, sr: int) -> float:
        ventana = int(sr * 0.04)
        salto   = int(sr * 0.01)
        lag_min = int(sr / 500)
        lag_max = int(sr / 50)
        vals    = []

        for i in range(0, len(audio) - ventana, salto):
            s = audio[i:i + ventana]
            s = s - np.mean(s)
            if np.max(np.abs(s)) < 1e-6:
                continue
            ac = np.correlate(s, s, mode='full')
            ac = ac[len(ac) // 2:]
            ac = ac / (ac[0] + 1e-10)
            if lag_max >= len(ac):
                continue
            pico = np.clip(np.max(ac[lag_min:lag_max]), 0, 0.9999)
            vals.append(10 * np.log10(pico / (1 - pico + 1e-10)))

        return float(np.median(vals)) if vals else 0.0

  
    def calculate_cv_ritmo(self, audio: np.ndarray, sr: int) -> float:
        hop   = int(sr * 0.010)
        frame = int(sr * 0.025)
        rms   = librosa.feature.rms(y=audio, frame_length=frame, hop_length=hop)[0]
        umbral_rms = np.mean(rms) * 0.5
        activo     = (rms > umbral_rms).astype(int)
        cambios    = np.where(np.diff(activo) == 1)[0]

        if len(cambios) < 3:
            return 0.5
        intervalos = np.diff(cambios) * (hop / sr)
        return float(np.std(intervalos) / np.mean(intervalos)) if np.mean(intervalos) > 0 else 0.5

   
    def calculate_ratio_silencio(self, audio: np.ndarray, sr: int) -> float:
        hop   = int(sr * 0.010)
        frame = int(sr * 0.025)
        rms   = librosa.feature.rms(y=audio, frame_length=frame, hop_length=hop)[0]
        return float(np.sum(rms < np.max(rms) * 0.05) / len(rms)) if len(rms) > 0 else 0.0

    
    def bandpass_filter(self, audio: np.ndarray, sr: int,
                        fmin: float = 80.0, fmax: float = 8000.0) -> np.ndarray:
        frec    = np.fft.rfftfreq(len(audio), 1 / sr)
        mascara = (frec >= fmin) & (frec <= fmax)
        filtrado = np.fft.irfft(
            np.fft.rfft(audio) * mascara.astype(float)
        )[:len(audio)]
        return filtrado
