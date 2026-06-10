
import warnings
warnings.filterwarnings("ignore")


from infrastructure.audio.carga_audio_ffmpeg    import FfmpegAudioLoader
from infrastructure.audio.analizador_librosa import LibrosaAnalyzer


from application.use_cases.load_audio_use_case    import LoadAudioUseCase
from application.use_cases.analyze_audio_use_case import AnalyzeAudioUseCase
from application.services.comparacion_service      import ComparisonService


from presentation.main_ventana import MainWindow


def main() -> None:
    
    loader   = FfmpegAudioLoader()
    analyzer = LibrosaAnalyzer()

   
    load_uc    = LoadAudioUseCase(loader)
    analyze_uc = AnalyzeAudioUseCase(analyzer)

    
    compare_svc = ComparisonService()


    app = MainWindow(
        load_uc=load_uc,
        analyze_uc=analyze_uc,
        compare_svc=compare_svc,
    )

    
    app.run()


if __name__ == "__main__":
    main()
