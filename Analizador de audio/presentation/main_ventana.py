
import tkinter as tk
from tkinter import messagebox

from domain.Valor_objetos import THRESHOLDS as T, WEIGHTS as W

from application.use_cases         import LoadAudioUseCase, AnalyzeAudioUseCase
from application.services.comparacion_service import ComparisonService

from presentation.widgets.carga_audio_widget import AudioLoaderWidget
from presentation.widgets.texto_resultados_widget import ResultsTextWidget
from presentation.charts.analisis_graficos      import AnalysisCharts



COLORS = {
    "bg":           "#0f0f14",       
    "surface":      "#1a1a2e",       
    "surface2":     "#16213e",       
    "accent1":      "#7c3aed",       
    "accent1_light":"#a855f7",       
    "accent2":      "#06b6d4",       
    "accent3":      "#10b981",       
    "accent3_light":"#34d399",       
    "text_primary": "#f1f5f9",       
    "text_secondary":"#94a3b8",      
    "text_muted":   "#475569",       
    "border":       "#2d2d4a",       
}


class MainWindow:
   

    def __init__(
        self,
        load_uc:     LoadAudioUseCase,
        analyze_uc:  AnalyzeAudioUseCase,
        compare_svc: ComparisonService,
    ):
        self._analyze_uc  = analyze_uc
        self._compare_svc = compare_svc

       
        self._root = tk.Tk()
        self._root.title(
            "Comparador Voz — Humano vs IA  (v6 — 7 voces calibradas)")
        self._root.geometry("1500x980")
        self._root.configure(bg=COLORS["bg"])
        self._root.resizable(True, True)

        
        self._build_header()

        
        self._build_separator()

        
        self._loader_widget = AudioLoaderWidget(self._root, load_uc)

        
        self._build_compare_button()

        
        self._text_widget = ResultsTextWidget(self._root)

        self._charts = AnalysisCharts(self._root)

    

    def run(self) -> None:
        self._root.mainloop()


    def _build_header(self) -> None:
        fh = tk.Frame(self._root, bg=COLORS["bg"])
        fh.pack(pady=(18, 6), fill=tk.X, padx=30)


        header_card = tk.Frame(fh, bg=COLORS["surface"], padx=20, pady=14)
        header_card.pack(fill=tk.X)

        
        accent_bar = tk.Frame(header_card, bg=COLORS["accent1"], width=4)
        accent_bar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 16))

        text_frame = tk.Frame(header_card, bg=COLORS["surface"])
        text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(
            text_frame,
            text="  COMPARADOR DE VOZ  —  HUMANO  vs  IA",
            font=("Segoe UI", 18, "bold"),
            fg=COLORS["text_primary"], bg=COLORS["surface"],
        ).pack(anchor="w")

        tk.Label(
            text_frame,
            text=(
                f"v6  ·  7 voces calibradas  ·  "
                f"Jitter<{T.jitter:.3f}%  ·  "
                f"CV<{T.cv_ritmo:.3f}  ·  "
                f"F0σ>{T.f0_std_ia:.0f}Hz"
            ),
            font=("Segoe UI", 9),
            fg=COLORS["text_muted"], bg=COLORS["surface"],
        ).pack(anchor="w", pady=(2, 0))

       
        ver_frame = tk.Frame(header_card, bg=COLORS["accent1"],
                             padx=8, pady=4)
        ver_frame.pack(side=tk.RIGHT, anchor="e")
        tk.Label(ver_frame, text="v6", font=("Segoe UI", 10, "bold"),
                 fg="white", bg=COLORS["accent1"]).pack()

    def _build_separator(self) -> None:
        sep = tk.Frame(self._root, bg=COLORS["border"], height=1)
        sep.pack(fill=tk.X, padx=30, pady=(0, 4))

    def _build_compare_button(self) -> None:
        fb = tk.Frame(self._root, bg=COLORS["bg"])
        fb.pack(pady=(4, 8))

        btn = tk.Button(
            fb,
            text="    COMPARAR AUDIOS  ",
            font=("Segoe UI", 13, "bold"),
            bg=COLORS["accent3"], fg=COLORS["bg"],
            relief=tk.FLAT, padx=28, pady=11,
            cursor="hand2",
            activebackground=COLORS["accent3_light"],
            activeforeground=COLORS["bg"],
            command=self._comparar,
        )
        btn.pack()
        
        btn.bind("<Enter>", lambda e: btn.configure(bg=COLORS["accent3_light"]))
        btn.bind("<Leave>", lambda e: btn.configure(bg=COLORS["accent3"]))


    def _comparar(self) -> None:
        lw = self._loader_widget
        if not lw.ambos_listos():
            messagebox.showwarning("Faltan audios", "Carga los 2 audios primero.")
            return

        r1 = self._analyze_uc.execute(
            lw.get_audio(1), lw.get_sr(1), lw.get_nombre(1))
        r2 = self._analyze_uc.execute(
            lw.get_audio(2), lw.get_sr(2), lw.get_nombre(2))

        report = self._compare_svc.compare(r1, r2)

        self._text_widget.render(r1, r2, report)
        self._charts.render(r1, r2)
