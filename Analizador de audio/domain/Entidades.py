

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class VoiceMetrics:
    """
    Métricas acústicas extraídas de un segmento de audio.
    Objeto de valor inmutable tras su construcción.
    """
    f0_media:   float          
    f0_std:     float           
    jitter:     float          
    pendiente:  float          
    hnr_real:   float          
    cv_ritmo:   float          
    ratio_sil:  float = 0.0   


@dataclass
class WindowResult:
    """Resultado del análisis de una ventana temporal de 2 s."""
    ventana:   int
    t_inicio:  float
    t_fin:     float
    metrics:   VoiceMetrics
    score_hum: float           
    score_ia:  float          
    categoria: str             


@dataclass
class AnalysisResult:
    """
    Resultado completo del análisis de un audio.
    Incluye métricas globales, clasificación y resultados por ventana.
    """
    
    audio_limpio: np.ndarray
    f0:           np.ndarray        
    f0_ok:        np.ndarray        
    sr:           int

    
    metrics: VoiceMetrics

    
    score_humano: float             
    es_humano:    bool
    categoria:    str
    razones:      List[str] = field(default_factory=list)

    
    ventanas: List[WindowResult] = field(default_factory=list)

    
    nombre_archivo: str = ""
