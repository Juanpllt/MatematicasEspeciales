# ================================================================
# DOMAIN / PORTS
# Interfaces (puertos) que el dominio necesita del exterior.
# Las implementaciones concretas viven en infrastructure/.
# ================================================================

from abc import ABC, abstractmethod
import numpy as np

from domain.Entidades import VoiceMetrics


class AudioLoaderPort(ABC):
    """Puerto para carga y conversión de archivos de audio."""

    @abstractmethod
    def load(self, ruta: str, sr_target: int = 16000) -> tuple[np.ndarray, int]:
        """
        Carga un archivo de audio (cualquier formato) y lo devuelve
        como array numpy mono a la frecuencia de muestreo solicitada.

        Returns
        -------
        audio : np.ndarray  — muestras normalizadas float32
        sr    : int         — frecuencia de muestreo real
        """


class AudioAnalyzerPort(ABC):
    """Puerto para extracción de métricas acústicas."""

    @abstractmethod
    def extract_f0(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extrae la serie de frecuencia fundamental (Hz), NaN en silencios."""

    @abstractmethod
    def calculate_jitter(self, f0_ok: np.ndarray) -> float:
        """Jitter relativo entre ciclos consecutivos (%)."""

    @abstractmethod
    def calculate_hnr(self, audio: np.ndarray, sr: int) -> float:
        """HNR estimado por autocorrelación (dB)."""

    @abstractmethod
    def calculate_cv_ritmo(self, audio: np.ndarray, sr: int) -> float:
        """Coeficiente de variación del ritmo de actividad vocal."""

    @abstractmethod
    def calculate_ratio_silencio(self, audio: np.ndarray, sr: int) -> float:
        """Fracción de frames por debajo del umbral de silencio."""

    @abstractmethod
    def bandpass_filter(self, audio: np.ndarray, sr: int,
                        fmin: float = 80.0, fmax: float = 8000.0) -> np.ndarray:
        """Filtra el audio en la banda vocal útil."""
