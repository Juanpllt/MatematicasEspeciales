

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassifierThresholds:
    """Umbrales de decisión calibrados con 7 voces reales."""
    jitter:    float = (2.819 + 3.335) / 2   
    cv_ritmo:  float = (0.790 + 0.839) / 2   
    f0_humano: float = 165.0                  
    pendiente: float = 0.0                    
    f0_std_ia: float = 40.0                   


@dataclass(frozen=True)
class ClassifierWeights:
    """Pesos del score HUMANO — v6."""
    jitter:    float = 0.45   
    cv_ritmo:  float = 0.20   
    pendiente: float = 0.20
    f0:        float = 0.10
    penaliz_f0_std: float = 0.25



THRESHOLDS = ClassifierThresholds()
WEIGHTS    = ClassifierWeights()
