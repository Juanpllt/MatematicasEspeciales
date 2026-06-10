
import tkinter as tk
import numpy as np

import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from domain.Entidades      import AnalysisResult
from domain.Valor_objetos import THRESHOLDS as T



_BG      = "#0f0f14"
_SURFACE = "#1a1a2e"
_C1      = "#3b82f6"   
_C2      = "#ec4899"   
_CU      = "#f59e0b"   
_CG      = "#10b981"   
_GRID    = "#1e293b"   
_TEXT    = "#94a3b8"   
_TITLE   = "#e2e8f0"   


class AnalysisCharts:
 

    def __init__(self, parent: tk.Widget):
        
        outer = tk.Frame(parent, bg=_SURFACE, padx=2, pady=2)
        outer.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 12))

        header = tk.Frame(outer, bg=_SURFACE, padx=14, pady=8)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="  ANÁLISIS VISUAL",
            font=("Segoe UI", 10, "bold"),
            fg="#a855f7", bg=_SURFACE,
        ).pack(side=tk.LEFT)

        self._figura = plt.Figure(figsize=(14, 6.5), dpi=95)
        self._figura.patch.set_facecolor(_BG)

        self._canvas = FigureCanvasTkAgg(self._figura, master=outer)
        self._canvas.get_tk_widget().pack(
            fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

    

    def render(self, r1: AnalysisResult, r2: AnalysisResult) -> None:
        self._figura.clear()
        gs = gridspec.GridSpec(3, 2, figure=self._figura,
                               hspace=0.62, wspace=0.36,
                               left=0.06, right=0.97,
                               top=0.95, bottom=0.08)

        for ci, (r, col) in enumerate([(r1, _C1), (r2, _C2)]):
            self._plot_waveform( self._figura.add_subplot(gs[0, ci]), r, col, ci+1)
            self._plot_f0(       self._figura.add_subplot(gs[1, ci]), r, col)
            self._plot_ventanas( self._figura.add_subplot(gs[2, ci]), r, ci+1)

        self._canvas.draw()

  

    @staticmethod
    def _estilo(ax, titulo: str, fs: int = 8) -> None:
        ax.set_facecolor(_SURFACE)
        ax.tick_params(colors=_TEXT, labelsize=7)
        ax.xaxis.label.set_color(_TEXT)
        ax.yaxis.label.set_color(_TEXT)
        ax.set_title(titulo, color=_TITLE, fontsize=fs, pad=6,
                     fontfamily="DejaVu Sans")
        for sp in ax.spines.values():
            sp.set_edgecolor(_GRID)
            sp.set_linewidth(0.8)
        ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)

    

    def _plot_waveform(self, ax, r: AnalysisResult, col: str, n: int) -> None:
        librosa.display.waveshow(r.audio_limpio, sr=r.sr, ax=ax,
                                 color=col, alpha=0.9)
        
        ax.fill_between(
            np.linspace(0, len(r.audio_limpio)/r.sr, len(r.audio_limpio)),
            r.audio_limpio, 0,
            color=col, alpha=0.12,
        )
        sh  = r.score_humano
        nom = r.nombre_archivo[:26]
        self._estilo(
            ax,
            f"Audio {n} — {nom}\n"
            f"→ {r.categoria}   H:{sh*100:.0f}% / IA:{(1-sh)*100:.0f}%"
        )

    def _plot_f0(self, ax, r: AnalysisResult, col: str) -> None:
        import librosa as _lib
        t = _lib.times_like(r.f0, sr=r.sr)
        m = r.metrics

        ax.plot(t, r.f0, color=col, lw=1.2,
                label=f"F0 μ={m.f0_media:.0f}Hz  σ={m.f0_std:.0f}Hz")
        ax.axhline(m.f0_media,  color=col, ls="--", lw=0.9, alpha=0.55)
        ax.axhline(T.f0_humano, color=_CU,  ls=":",  lw=1.3,
                   label=f"Umbral {T.f0_humano:.0f}Hz")

        if len(t) > 1:
            ax.fill_between([t[0], t[-1]], T.f0_humano, 500,
                            alpha=0.05, color=_CG)
        ax.set_ylim(0, 500)
        ax.set_ylabel("Hz", fontsize=7)
        ax.legend(fontsize=6, framealpha=0.2, facecolor=_SURFACE,
                  edgecolor=_GRID, labelcolor=_TEXT)

        f0std_txt = f"σ={'IA' if m.f0_std > T.f0_std_ia else 'OK'}"
        jit_txt   = f"Jit={'OK' if m.jitter   < T.jitter   else 'MAL'}"
        cv_txt    = f"CV={'OK'  if m.cv_ritmo < T.cv_ritmo else 'MAL'}"
        nom = r.nombre_archivo[:12]
        self._estilo(ax, f"F0 — {nom}   {f0std_txt}   {jit_txt}   {cv_txt}")

    def _plot_ventanas(self, ax, r: AnalysisResult, n: int) -> None:
        vv = r.ventanas
        if not vv:
            ax.text(0.5, 0.5, "Audio muy corto",
                    ha="center", va="center", color=_TEXT,
                    fontsize=9)
            self._estilo(ax, f"Score HUMANO por ventana — Audio {n}")
            return

        xs  = [v.ventana   for v in vv]
        sc  = [v.score_hum for v in vv]
        col = ["#10b981" if s >= 0.50 else "#ef4444" for s in sc]

        bars = ax.bar(xs, sc, color=col, alpha=0.85, width=0.65,
                      edgecolor="none")

        
        for bar, c in zip(bars, col):
            bar.set_linewidth(0)

        ax.axhline(0.50, color=_CU, ls="--", lw=1.3,
                   label="Umbral 0.50", alpha=0.9)
        ax.set_ylim(0, 1.15)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"V{x}" for x in xs], fontsize=5, rotation=45)
        ax.set_ylabel("Score Humano", fontsize=7)
        ax.legend(fontsize=6, framealpha=0.2, facecolor=_SURFACE,
                  edgecolor=_GRID, labelcolor=_TEXT)

        for x, s in zip(xs, sc):
            ax.text(x, s + 0.04,
                    "H" if s >= 0.50 else "IA",
                    ha="center", va="bottom", fontsize=5,
                    color="#10b981" if s >= 0.50 else "#ef4444",
                    fontweight="bold")

        self._estilo(ax, f"Score HUMANO por ventana — Audio {n}")
