
from dataclasses import dataclass

from domain.Entidades     import AnalysisResult
from domain.Valor_objetos import THRESHOLDS as T, WEIGHTS as W


@dataclass
class ComparisonReport:

    r1: AnalysisResult
    r2: AnalysisResult

   
    delta_f0_media:   float = 0.0
    delta_f0_std:     float = 0.0
    delta_jitter:     float = 0.0
    delta_cv_ritmo:   float = 0.0
    delta_pendiente:  float = 0.0
    delta_hnr:        float = 0.0

    
    mismo_tipo:       bool  = False   
    nom_humano:       str   = ""      
    nom_ia:           str   = ""      
    mas_humano:       str   = ""      
    mas_ia:           str   = ""      


class ComparisonService:


    def compare(self, r1: AnalysisResult, r2: AnalysisResult) -> ComparisonReport:
        report = ComparisonReport(
            r1=r1, r2=r2,
            delta_f0_media  = abs(r1.metrics.f0_media  - r2.metrics.f0_media),
            delta_f0_std    = abs(r1.metrics.f0_std    - r2.metrics.f0_std),
            delta_jitter    = abs(r1.metrics.jitter    - r2.metrics.jitter),
            delta_cv_ritmo  = abs(r1.metrics.cv_ritmo  - r2.metrics.cv_ritmo),
            delta_pendiente = abs(r1.metrics.pendiente - r2.metrics.pendiente),
            delta_hnr       = abs(r1.metrics.hnr_real  - r2.metrics.hnr_real),
        )

        h1, h2 = r1.es_humano, r2.es_humano

        if h1 != h2:
            report.mismo_tipo = False
            report.nom_humano = r1.nombre_archivo if h1 else r2.nombre_archivo
            report.nom_ia     = r2.nombre_archivo if h1 else r1.nombre_archivo
        else:
            report.mismo_tipo = True
            if h1 and h2:          
                report.mas_humano = (r1.nombre_archivo
                                     if r1.score_humano >= r2.score_humano
                                     else r2.nombre_archivo)
            else:                 
                report.mas_ia = (r2.nombre_archivo
                                 if r1.score_humano >= r2.score_humano
                                 else r1.nombre_archivo)

        return report
