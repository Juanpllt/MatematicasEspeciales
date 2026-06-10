
import os
import subprocess
import tempfile

import librosa
import numpy as np

from domain.ports import AudioLoaderPort


class FfmpegAudioLoader(AudioLoaderPort):
    """Carga de audio en cualquier formato mediante ffmpeg."""

    def load(self, ruta: str, sr_target: int = 16000) -> tuple[np.ndarray, int]:
        """
        Convierte la ruta a WAV mono 16 kHz (o sr_target) con ffmpeg,
        carga con librosa y devuelve (audio_array, sr).

        Raises
        ------
        RuntimeError si ffmpeg falla.
        """
        ruta_wav = self._convertir_a_wav(ruta, sr_target)
        try:
            audio, sr = librosa.load(ruta_wav, sr=sr_target, mono=True)
        finally:
            self._limpiar(ruta_wav)
        return audio, sr



    @staticmethod
    def _convertir_a_wav(ruta_audio: str, sr_target: int) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        ruta_wav = tmp.name
        tmp.close()

        cmd = [
            "ffmpeg", "-i", ruta_audio,
            "-ar", str(sr_target),
            "-ac", "1",
            ruta_wav, "-y",
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg falló:\n{res.stderr.decode()}")
        return ruta_wav

    @staticmethod
    def _limpiar(ruta_wav: str) -> None:
        try:
            if os.path.exists(ruta_wav):
                os.remove(ruta_wav)
        except OSError:
            pass
