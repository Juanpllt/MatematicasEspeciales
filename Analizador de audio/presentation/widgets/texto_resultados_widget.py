
import tkinter as tk

from domain.Entidades      import AnalysisResult
from domain.Valor_objetos import THRESHOLDS as T, WEIGHTS as W
from application.services.comparacion_service import ComparisonReport



COLORS = {
    "bg":            "#0f0f14",
    "surface":       "#1a1a2e",
    "text_primary":  "#f1f5f9",
    "text_secondary":"#94a3b8",
    "text_muted":    "#475569",
    "audio1":        "#3b82f6",
    "audio2":        "#ec4899",
    "success":       "#10b981",
    "warning":       "#f59e0b",
    "danger":        "#ef4444",
    "accent":        "#a855f7",
    "border":        "#2d2d4a",
    "scrollbar":     "#2d2d4a",
}


class ResultsTextWidget:
   

    def __init__(self, parent: tk.Widget):
        outer = tk.Frame(parent, bg=COLORS["surface"],
                         padx=2, pady=2)
        outer.pack(fill=tk.BOTH, padx=24, pady=(0, 6))

        
        header = tk.Frame(outer, bg=COLORS["surface"], padx=14, pady=8)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text="  RESULTADOS DEL ANÁLISIS",
            font=("Segoe UI", 10, "bold"),
            fg=COLORS["accent"], bg=COLORS["surface"],
        ).pack(side=tk.LEFT)

        
        frame = tk.Frame(outer, bg=COLORS["bg"])
        frame.pack(fill=tk.BOTH, padx=4, pady=(0, 4))

        self._text = tk.Text(
            frame, width=130, height=16,
            font=("Consolas", 9),
            bg=COLORS["bg"], fg=COLORS["text_primary"],
            insertbackground="white",
            selectbackground=COLORS["accent"],
            relief=tk.FLAT, padx=14, pady=10,
            spacing1=1, spacing2=1,
        )

        scroll = tk.Scrollbar(
            frame,
            command=self._text.yview,
            bg=COLORS["surface"],
            troughcolor=COLORS["bg"],
            activebackground=COLORS["accent"],
            relief=tk.FLAT, width=8,
        )
        self._text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._text.pack(fill=tk.BOTH, expand=True)

        
        self._text.tag_configure("header",
            foreground=COLORS["accent"], font=("Consolas", 9, "bold"))
        self._text.tag_configure("audio1",
            foreground=COLORS["audio1"], font=("Consolas", 9, "bold"))
        self._text.tag_configure("audio2",
            foreground=COLORS["audio2"], font=("Consolas", 9, "bold"))
        self._text.tag_configure("success",
            foreground=COLORS["success"])
        self._text.tag_configure("warning",
            foreground=COLORS["warning"])
        self._text.tag_configure("danger",
            foreground=COLORS["danger"])
        self._text.tag_configure("muted",
            foreground=COLORS["text_muted"])
        self._text.tag_configure("secondary",
            foreground=COLORS["text_secondary"])

   

    def render(self, r1: AnalysisResult, r2: AnalysisResult,
               report: ComparisonReport) -> None:
        self._clear()
        self._render_audio(r1, 1)
        self._render_audio(r2, 2)
        self._render_comparacion(report)

    

    def _ins(self, text, tag=None):
        if tag:
            self._text.insert(tk.END, text, tag)
        else:
            self._text.insert(tk.END, text)

    def _render_audio(self, r: AnalysisResult, idx: int) -> None:
        sh  = r.score_humano
        sia = 1.0 - sh
        audio_tag = "audio1" if idx == 1 else "audio2"

        if r.es_humano:
            confianza = "ALTA" if sh >= 0.70 else "MEDIA"
            veredicto = f"VOZ HUMANA      (Humano: {sh*100:.0f}% / IA: {sia*100:.0f}%)"
        else:
            confianza = "ALTA" if sia >= 0.70 else "MEDIA"
            veredicto = f"VOZ DE IA       (IA: {sia*100:.0f}% / Humano: {sh*100:.0f}%)"
            if sia < 0.50:
                veredicto = f"INDEFINIDO      (Humano {sh*100:.0f}% / IA {sia*100:.0f}%)"
                confianza = "BAJA"

        self._ins("  ┌" + "─"*66 + "┐\n", "header")
        self._ins(f"  │  AUDIO {idx}: ", "header")
        self._ins(f"{r.nombre_archivo}", audio_tag)
        self._ins("\n")
        self._ins(f"  │   {veredicto}\n")
        self._ins(f"  │  Categoría : {r.categoria}   Confianza: ", "secondary")
        conf_tag = "success" if confianza == "ALTA" else ("warning" if confianza == "MEDIA" else "danger")
        self._ins(f"{confianza}\n", conf_tag)
        self._ins("  └" + "─"*66 + "┘\n", "header")

        self._ins("\n  CRITERIOS:\n", "secondary")
        for rz in r.razones:
            self._ins(f"    • {rz}\n")

        m = r.metrics
        self._ins("\n  MÉTRICAS GLOBALES:\n", "secondary")
        self._ins(f"    F0 media      : {m.f0_media:.2f} Hz\n")
        self._ins(f"    F0 STD        : {m.f0_std:.2f} Hz")
        umbral_tag = "danger" if m.f0_std > T.f0_std_ia else "success"
        self._ins(f"   (umbral IA > {T.f0_std_ia:.0f} Hz)\n", umbral_tag)
        self._ins(f"    Jitter        : {m.jitter:.3f} %")
        j_tag = "danger" if m.jitter >= T.jitter else "success"
        self._ins(f"   (umbral {T.jitter:.3f} %)\n", j_tag)
        self._ins(f"    Pendiente F0  : {m.pendiente:.6f} Hz/frame\n")
        self._ins(f"    HNR autocorr  : {m.hnr_real:.3f} dB  (informativo)\n", "muted")
        self._ins(f"    CV ritmo      : {m.cv_ritmo:.4f}")
        cv_tag = "danger" if m.cv_ritmo >= T.cv_ritmo else "success"
        self._ins(f"   (umbral {T.cv_ritmo:.3f})\n", cv_tag)
        self._ins(f"    Ratio silencio: {m.ratio_sil:.3f}\n")
        self._ins(f"    Score humano  : {r.score_humano:.2f}  |  Score IA: {1-r.score_humano:.2f}\n")

        self._ins(f"\n  VENTANAS (2 s):\n", "secondary")
        self._ins(
            f"    {'V':>3}  {'t(s)':>9}  {'F0':>6}  {'F0std':>6}  "
            f"{'Jit%':>5}  {'CV':>6}  {'Pend':>8}  {'S.Hum':>6}  {'S.IA':>5}  Categ.\n",
            "muted",
        )
        self._ins("    " + "─"*88 + "\n", "muted")

        for v in r.ventanas:
            linea = (
                f"    {v.ventana:>3}  "
                f"{v.t_inicio:>4.1f}–{v.t_fin:>4.1f}  "
                f"{v.metrics.f0_media:>6.1f}  "
                f"{v.metrics.f0_std:>6.1f}  "
                f"{v.metrics.jitter:>5.2f}  "
                f"{v.metrics.cv_ritmo:>6.3f}  "
                f"{v.metrics.pendiente:>+8.4f}  "
                f"{v.score_hum:>6.2f}  "
                f"{v.score_ia:>5.2f}  {v.categoria}\n"
            )
            tag = "success" if v.score_hum >= 0.50 else "danger"
            self._ins(linea, tag)
        self._ins("\n\n")



    def _render_comparacion(self, rp: ComparisonReport) -> None:
        self._ins("  ┌" + "─"*66 + "┐\n", "header")
        self._ins("  │  COMPARACIÓN DIRECTA\n", "header")
        self._ins("  └" + "─"*66 + "┘\n", "header")

        if not rp.mismo_tipo:
            self._ins(f"    VOZ HUMANA  → ", "success")
            self._ins(f"{rp.nom_humano}\n")
            self._ins(f"    VOZ DE IA   → ", "danger")
            self._ins(f"{rp.nom_ia}\n")
        elif rp.r1.es_humano:
            self._ins("    Ambos clasificados como VOZ HUMANA\n", "warning")
            self._ins(f"      El más humano: {rp.mas_humano}\n")
        else:
            self._ins("    Ambos clasificados como VOZ DE IA\n", "warning")
            self._ins(f"      El más IA: {rp.mas_ia}\n")

        self._ins("\n")
        self._ins(f"   F0 media    : {rp.delta_f0_media:.1f} Hz\n")
        self._ins(f"   F0 STD      : {rp.delta_f0_std:.1f} Hz\n")
        self._ins(f"   Jitter      : {rp.delta_jitter:.3f} %\n")
        self._ins(f"   CV ritmo    : {rp.delta_cv_ritmo:.3f}\n")
        self._ins(f"   Pendiente   : {rp.delta_pendiente:.6f} Hz/frame\n")
        self._ins(f"   HNR autocorr: {rp.delta_hnr:.3f} dB\n")

        self._ins("\n  UMBRALES v6:\n", "secondary")
        self._ins(f"    Jitter   : < {T.jitter:.3f}%  (+{W.jitter:.0%})\n", "muted")
        self._ins(f"    CV ritmo : < {T.cv_ritmo:.3f}    (+{W.cv_ritmo:.0%})\n", "muted")
        self._ins(f"    Pendiente: > 0.0         (+{W.pendiente:.0%})\n", "muted")
        self._ins(f"    F0       : > {T.f0_humano:.0f} Hz   (+{W.f0:.0%})\n", "muted")
        self._ins(f"    F0 STD   : > {T.f0_std_ia:.0f} Hz   (-{W.penaliz_f0_std:.0%})\n", "muted")
        self._ins(f"    HNR      : informativo únicamente\n", "muted")

    

    def _clear(self) -> None:
        self._text.delete("1.0", tk.END)
