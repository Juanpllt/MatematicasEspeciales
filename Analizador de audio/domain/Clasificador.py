
import numpy as np
from domain.Valor_objetos import THRESHOLDS as T, WEIGHTS as W
from domain.Entidades import VoiceMetrics, WindowResult



def calcular_pendiente_f0(f0_ok: np.ndarray) -> float:
    if len(f0_ok) < 4:
        return 0.0
    x = np.arange(len(f0_ok))
    m, _ = np.polyfit(x, f0_ok, 1)
    return float(m)


def determinar_categoria(f0_media: float, es_ia: bool) -> str:

    if f0_media == 0:
        return "IA (género indef.)" if es_ia else "Humano (género indef.)"
    if es_ia:
        return "IA Hombre"  if f0_media <= 185 else "IA Mujer"
    else:
        return "Humano Hombre" if f0_media <= 185 else "Humano Mujer"



def score_ventana(metrics: VoiceMetrics) -> float:

    s = 0.0
    if metrics.jitter    < T.jitter:    s += W.jitter
    if metrics.cv_ritmo  < T.cv_ritmo:  s += W.cv_ritmo
    if metrics.pendiente > T.pendiente: s += W.pendiente
    if metrics.f0_media  > T.f0_humano: s += W.f0
    if metrics.f0_std    > T.f0_std_ia: s -= W.penaliz_f0_std
    return max(0.0, min(1.0, s))



def clasificar(metrics: VoiceMetrics) -> tuple[float, str, list[str], bool]:

    score_hum = 0.0
    razones   = []

    if metrics.jitter < T.jitter:
        score_hum += W.jitter
        razones.append(
            f"Jitter={metrics.jitter:.3f}%  <  {T.jitter:.3f}%"
            f"  → ciclos regulares  → HUMANO  (+{W.jitter:.0%})")
    else:
        razones.append(
            f"Jitter={metrics.jitter:.3f}%  ≥  {T.jitter:.3f}%"
            f"  → ciclos irregulares  → IA  (+0%)")

    
    if metrics.cv_ritmo < T.cv_ritmo:
        score_hum += W.cv_ritmo
        razones.append(
            f"CV ritmo={metrics.cv_ritmo:.3f}  <  {T.cv_ritmo:.3f}"
            f"  → ritmo regular  → HUMANO  (+{W.cv_ritmo:.0%})")
    else:
        razones.append(
            f"CV ritmo={metrics.cv_ritmo:.3f}  ≥  {T.cv_ritmo:.3f}"
            f"  → ritmo variable  → IA  (+0%)")

    
    if metrics.pendiente > T.pendiente:
        score_hum += W.pendiente
        razones.append(
            f"Pendiente={metrics.pendiente:.5f}  >  0"
            f"  → prosodia ascendente  → HUMANO  (+{W.pendiente:.0%})")
    else:
        razones.append(
            f"Pendiente={metrics.pendiente:.5f}  ≤  0"
            f"  → prosodia plana/baja  → neutro  (+0%)")

    
    if metrics.f0_media > T.f0_humano:
        score_hum += W.f0
        razones.append(
            f"F0={metrics.f0_media:.1f} Hz  >  {T.f0_humano:.0f} Hz"
            f"  → tono elevado  → HUMANO  (+{W.f0:.0%})")
    else:
        razones.append(
            f"F0={metrics.f0_media:.1f} Hz  ≤  {T.f0_humano:.0f} Hz"
            f"  → tono normal/bajo  → neutro  (+0%)")

    
    if metrics.f0_std > T.f0_std_ia:
        score_hum -= W.penaliz_f0_std
        razones.append(
            f"F0 STD={metrics.f0_std:.1f} Hz  >  {T.f0_std_ia:.0f} Hz"
            f"  → variación tonal extrema  → IA  (-{W.penaliz_f0_std:.0%})")
    else:
        razones.append(
            f"F0 STD={metrics.f0_std:.1f} Hz  ≤  {T.f0_std_ia:.0f} Hz"
            f"  → variación tonal normal  → neutro  (+0%)")


    razones.append(
        f"HNR={metrics.hnr_real:.3f} dB  → informativo (no clasifica en v6)")

    score_hum = max(0.0, min(1.0, score_hum))
    es_humano = score_hum >= 0.50
    categoria = determinar_categoria(metrics.f0_media, not es_humano)

    return score_hum, categoria, razones, es_humano



def construir_window_result(
        ventana: int, t_inicio: float, t_fin: float,
        metrics: VoiceMetrics) -> WindowResult:
    """Construye un WindowResult calculando su score y categoría."""
    sv  = score_ventana(metrics)
    cat = determinar_categoria(metrics.f0_media, sv < 0.50)
    return WindowResult(
        ventana=ventana, t_inicio=t_inicio, t_fin=t_fin,
        metrics=metrics, score_hum=sv, score_ia=1.0 - sv,
        categoria=cat,
    )
