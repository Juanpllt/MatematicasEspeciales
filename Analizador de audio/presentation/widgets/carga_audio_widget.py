

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

from application.use_cases import LoadAudioUseCase



COLORS = {
    "bg":            "#0f0f14",
    "surface":       "#1a1a2e",
    "surface2":      "#16213e",
    "audio1":        "#3b82f6",      
    "audio1_light":  "#60a5fa",
    "audio2":        "#ec4899",     
    "audio2_light":  "#f472b6",
    "text_primary":  "#f1f5f9",
    "text_secondary":"#94a3b8",
    "text_muted":    "#475569",
    "success":       "#10b981",
    "border":        "#2d2d4a",
}


class AudioLoaderWidget:


    _FORMATOS = [("Audios", "*.m4a *.mp4 *.wav *.mp3 *.mpeg *.ogg")]

    def __init__(self, parent: tk.Widget, load_use_case: LoadAudioUseCase):
        self._loader = load_use_case

        self._audios: dict[int, np.ndarray | None] = {1: None, 2: None}
        self._srs:    dict[int, int | None]        = {1: None, 2: None}
        self._rutas:  dict[int, str]               = {1: "",   2: ""}
        self._labels: dict[int, tk.Label]          = {}
        self._btns:   dict[int, tk.Button]         = {}

        
        outer = tk.Frame(parent, bg=COLORS["bg"])
        outer.pack(pady=(6, 4))

        frame = tk.Frame(outer, bg=COLORS["bg"])
        frame.pack()

        cfg = {
            1: (COLORS["audio1"],  COLORS["audio1_light"]),
            2: (COLORS["audio2"],  COLORS["audio2_light"]),
        }

        for n, (col, col_h) in cfg.items():
            
            card = tk.Frame(frame, bg=COLORS["surface"],
                            padx=16, pady=12)
            card.pack(side=tk.LEFT, padx=16)

            
            bar = tk.Frame(card, bg=col, width=3)
            bar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

            inner = tk.Frame(card, bg=COLORS["surface"])
            inner.pack(side=tk.LEFT)

            
            tk.Label(
                inner,
                text=f"AUDIO {n}",
                font=("Segoe UI", 8, "bold"),
                fg=col, bg=COLORS["surface"],
            ).pack(anchor="w")

            
            btn = tk.Button(
                inner,
                text=f"    Cargar Audio {n}  ",
                font=("Segoe UI", 11, "bold"),
                bg=col, fg="white",
                relief=tk.FLAT, padx=14, pady=8,
                cursor="hand2",
                activebackground=col_h,
                activeforeground="white",
                command=lambda x=n: self._cargar(x),
            )
            btn.pack(pady=(4, 2))
            btn.bind("<Enter>", lambda e, c=col_h, b=btn: b.configure(bg=c))
            btn.bind("<Leave>", lambda e, c=col,   b=btn: b.configure(bg=c))
            self._btns[n] = btn

            
            lbl = tk.Label(
                inner,
                text=f"Sin cargar",
                font=("Segoe UI", 8),
                fg=COLORS["text_muted"], bg=COLORS["surface"],
            )
            lbl.pack(anchor="w", pady=(2, 0))
            self._labels[n] = lbl

    

    def ambos_listos(self) -> bool:
        return self._audios[1] is not None and self._audios[2] is not None

    def get_audio(self, n: int) -> np.ndarray:
        return self._audios[n]

    def get_sr(self, n: int) -> int:
        return self._srs[n]

    def get_nombre(self, n: int) -> str:
        return self._rutas[n]



    def _cargar(self, n: int) -> None:
        ruta = filedialog.askopenfilename(
            title=f"Seleccionar Audio {n}",
            filetypes=self._FORMATOS,
        )
        if not ruta:
            return
        try:
            audio, sr = self._loader.execute(ruta)
            self._audios[n] = audio
            self._srs[n]    = sr
            self._rutas[n]  = os.path.basename(ruta)
            self._labels[n].config(
                text=f"  {self._rutas[n]}",
                fg=COLORS["success"],
            )
            messagebox.showinfo("Cargado", f"Audio {n} listo: {self._rutas[n]}")
        except Exception as exc:
            messagebox.showerror("Error",
                                 f"No se pudo cargar Audio {n}:\n{exc}")
