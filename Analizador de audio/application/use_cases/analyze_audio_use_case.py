# ================================================================
# APPLICATION / USE CASES / ANALYZE AUDIO
# Caso de uso: procesar un array de audio y obtener AnalysisResult.
# Orquesta el puerto de análisis + el clasificador del dominio.
# ================================================================

import numpy as np

from domain.Entidades    import AnalysisResult, VoiceMetrics
from domain.Clasificador  import (calcular_pendiente_f0, clasificar,
                                 construir_window_result)
from domain.ports       import AudioAnalyzerPort


class AnalyzeAudioUseCase:
    """
    Ejecuta el pipeline completo de análisis acústico sobre
    un audio ya cargado (numpy array).

    Dependencia inyectada: AudioAnalyzerPort
    """

    def __init__(self, analyzer: AudioAnalyzerPort):
        self._analyzer = analyzer

    def execute(self, audio_raw: np.ndarray, sr: int,
                nombre_archivo: str = "") -> AnalysisResult:
        """
        Parameters
        ----------
        audio_raw       : señal de audio cruda (float32, mono)
        sr              : frecuencia de muestreo
        nombre_archivo  : nombre visible en la UI

        Returns
        -------
        AnalysisResult completo
        """
        az = self._analyzer

        # 1 — Filtrado de banda vocal
        audio_limpio = az.bandpass_filter(audio_raw, sr, fmin=80.0, fmax=8000.0)

        # 2 — Extracción de F0
        f0    = az.extract_f0(audio_limpio, sr)
        f0_ok = f0[~np.isnan(f0)]

        # 3 — Métricas globales
        f0_media  = float(np.mean(f0_ok))  if len(f0_ok) > 0 else 0.0
        f0_std    = float(np.std(f0_ok))   if len(f0_ok) > 0 else 0.0
        jitter    = az.calculate_jitter(f0_ok)
        pendiente = calcular_pendiente_f0(f0_ok)
        hnr_real  = az.calculate_hnr(audio_limpio, sr)
        cv_ritmo  = az.calculate_cv_ritmo(audio_limpio, sr)
        ratio_sil = az.calculate_ratio_silencio(audio_limpio, sr)

        metrics = VoiceMetrics(
            f0_media=f0_media, f0_std=f0_std, jitter=jitter,
            pendiente=pendiente, hnr_real=hnr_real,
            cv_ritmo=cv_ritmo, ratio_sil=ratio_sil,
        )

        # 4 — Clasificación global
        score_hum, categoria, razones, es_humano = clasificar(metrics)

        # 5 — Análisis por ventanas de 2 s
        ventanas = self._analizar_ventanas(audio_limpio, sr, tam_s=2.0)

        return AnalysisResult(
            audio_limpio=audio_limpio,
            f0=f0, f0_ok=f0_ok, sr=sr,
            metrics=metrics,
            score_humano=score_hum,
            es_humano=es_humano,
            categoria=categoria,
            razones=razones,
            ventanas=ventanas,
            nombre_archivo=nombre_archivo,
        )

    # ── privados ─────────────────────────────────────────────────

    def _analizar_ventanas(self, audio: np.ndarray, sr: int,
                           tam_s: float = 2.0) -> list:
        az      = self._analyzer
        tam     = int(sr * tam_s)
        results = []

        for i in range(len(audio) // tam):
            seg  = audio[i * tam:(i + 1) * tam]
            f0   = az.extract_f0(seg, sr)
            f0_ok = f0[~np.isnan(f0)]

            m = VoiceMetrics(
                f0_media  = float(np.mean(f0_ok))  if len(f0_ok) > 0 else 0.0,
                f0_std    = float(np.std(f0_ok))   if len(f0_ok) > 0 else 0.0,
                jitter    = az.calculate_jitter(f0_ok),
                pendiente = calcular_pendiente_f0(f0_ok),
                hnr_real  = az.calculate_hnr(seg, sr),
                cv_ritmo  = az.calculate_cv_ritmo(seg, sr),
            )
            results.append(
                construir_window_result(i + 1, i * tam_s, (i + 1) * tam_s, m)
            )

        return results
