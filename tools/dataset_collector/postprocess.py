"""
=============================================================
 postprocess.py — Calcula restimulus a partir del CSV crudo
=============================================================
 El restimulus es la etiqueta de gesto refinada para compensar
 el tiempo de reaccion humano (~250ms para estimulos visuales).

 Para cada segmento activo (stimulus != 0):
   - Inicio: el gesto real comienza ~250ms despues del cue visual
   - Fin: el musculo comienza a relajarse ~250ms antes del cue
   de descanso

 Ejemplo (250ms = 250 muestras @ 1kHz):
   stimulus:    0000[1111111111111]0000
   restimulus:  0000000[1111111]0000000
                    ↑ +250ms    ↑ -250ms
=============================================================
"""

import numpy as np
import pandas as pd


def add_restimulus(csv_path: str, reaction_time_ms: int = 250, fs_hz: int = 1000) -> str:
    """Agrega columna 'restimulus' al CSV y lo sobrescribe.

    Args:
        csv_path:         Ruta al CSV con columna 'stimulus'
        reaction_time_ms: Offset de reaccion en milisegundos
        fs_hz:            Frecuencia de muestreo en Hz

    Returns:
        Ruta al CSV modificado (misma que la entrada)
    """
    df = pd.read_csv(csv_path)

    if "stimulus" not in df.columns:
        raise ValueError(f"El CSV no contiene columna 'stimulus': {csv_path}")

    shift = int(reaction_time_ms * fs_hz / 1000)
    stim = df["stimulus"].values
    restim = stim.copy()

    # Encontrar transiciones en stimulus
    changes = np.where(np.diff(stim) != 0)[0] + 1
    boundaries = np.concatenate(([0], changes, [len(stim)]))

    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        val = stim[seg_start]

        if val != 0:  # Segmento activo (gesto)
            # Recortar inicio: reaccion al cue visual
            trim_start = min(seg_start + shift, seg_end)
            restim[seg_start:trim_start] = 0

            # Recortar final: relajacion anticipada
            trim_end = max(seg_end - shift, trim_start)
            restim[trim_end:seg_end] = 0

    df["restimulus"] = restim
    df.to_csv(csv_path, index=False)

    # Estadisticas
    total_active_stim = np.sum(stim != 0)
    total_active_restim = np.sum(restim != 0)
    trimmed = total_active_stim - total_active_restim
    print(f"[POST] restimulus calculado: {total_active_restim} muestras activas "
          f"({trimmed} recortadas por reaccion de {reaction_time_ms}ms)")

    return csv_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python postprocess.py <archivo.csv> [reaction_time_ms]")
        sys.exit(1)

    path = sys.argv[1]
    rt = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    add_restimulus(path, rt)
    print(f"[OK] {path} actualizado con restimulus")
