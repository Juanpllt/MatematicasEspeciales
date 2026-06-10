# ================================================================
# APPLICATION / USE CASES / LOAD AUDIO
# Caso de uso: cargar un archivo de audio desde disco.
# Delega la conversión/carga al puerto AudioLoaderPort.
# ================================================================

import numpy as np

from domain.ports import AudioLoaderPort


class LoadAudioUseCase:
    """
    Carga un archivo de audio (cualquier formato soportado por ffmpeg)
    y lo devuelve como array numpy listo para análisis.

    Dependencia inyectada: AudioLoaderPort
    """

    def __init__(self, loader: AudioLoaderPort):
        self._loader = loader

    def execute(self, ruta: str,
                sr_target: int = 16000) -> tuple[np.ndarray, int]:
        """
        Parameters
        ----------
        ruta      : ruta absoluta al archivo de audio
        sr_target : frecuencia de muestreo objetivo

        Returns
        -------
        (audio, sr) — array float32 mono + sr efectivo

        Raises
        ------
        RuntimeError si la conversión falla
        """
        return self._loader.load(ruta, sr_target)
