"""
=============================================================
 protocol_engine.py — Motor de Bloques Aleatorizados
=============================================================
 Secuencia completa de la sesion:

   IDLE
   → CALIBRATION_NOISE (5s)             reposo total — ruido basal
   → CALIBRATION_MVC   (3s)             contraccion maxima
   → [Set 1..N_SETS]
       shuffle_gestures() al inicio de cada Set
       → REST           (3s)  — muestra el siguiente gesto
       → GESTURE_ACTIVE (5s)
       → REST           (3s)  — muestra el siguiente gesto
       → GESTURE_ACTIVE (5s)
       → ...
       → GESTURE_ACTIVE (5s)  — ultimo gesto del Set
   → DONE

 Ciclo REST-first:
   El reposo SIEMPRE precede al gesto, permitiendo que la GUI
   muestre al usuario que gesto viene a continuacion. Despues del
   ultimo gesto de cada set, se hace shuffle del siguiente set y
   se entra directamente a REST con el proximo gesto visible.
   Despues del ultimo gesto del ultimo set, se emite DONE.

 Protocolo de Bloques Aleatorizados:
   Cada Set es una permutacion aleatoria completa de los gestos.
   Ningun gesto se repite dentro del mismo Set.

 Propiedades thread-safe (GIL safe — leidas desde hilo UDP):
   recording     — True solo durante GESTURE_ACTIVE y REST
   stimulus      — id del gesto (0 fuera de GESTURE_ACTIVE)
   set_id        — bloque actual (1..n_sets)
   repetition_id — i-esima repeticion del gesto activo (1..n_sets)
   cal_phase     — "noise" | "mvc" | None  (propiedad calculada)

 Señales Qt:
   state_changed(str, dict)  — estado actual y metadata de UI
   progress(float, float)    — elapsed_s, total_s
   finished()                — todos los Sets completados
   calibration_done()        — emitida al salir de CALIBRATION_MVC
=============================================================
"""

import random
import time
from enum import Enum

from PyQt5.QtCore import QObject, QTimer, pyqtSignal


class State(Enum):
    IDLE              = "idle"
    CALIBRATION_NOISE = "calibration_noise"
    CALIBRATION_MVC   = "calibration_mvc"
    TRANSITION        = "transition"    # pausa post-MVC antes del primer set
    GESTURE_ACTIVE    = "gesture_active"
    REST              = "rest"
    DONE              = "done"


class ProtocolEngine(QObject):
    """Motor de estados del Protocolo de Bloques Aleatorizados."""

    state_changed    = pyqtSignal(str, object)   # (state_name, info_dict)
    progress         = pyqtSignal(float, float)  # (elapsed_s, total_s)
    finished         = pyqtSignal()              # todos los Sets completados
    calibration_done = pyqtSignal()              # fin de CALIBRATION_MVC

    def __init__(self, config: dict):
        super().__init__()
        proto   = config["protocol"]
        cal_cfg = config.get("calibration", {})

        self._base_gestures    = list(config["gestures"])
        self._n_sets           = int(proto.get("n_sets", 10))
        self._gesture_dur      = float(proto["gesture_duration_s"])
        self._rest_dur         = float(proto["rest_duration_s"])
        self._transition_dur   = float(proto.get("post_mvc_duration_s", 2.0))
        self._noise_dur        = float(cal_cfg.get("noise_duration_s", 5.0))
        self._mvc_dur          = float(cal_cfg.get("mvc_duration_s", 3.0))

        # ── Estado publico thread-safe ─────────────────────────
        self.recording     = False
        self.stimulus      = 0
        self.set_id        = 0    # 1..n_sets
        self.repetition_id = 0    # i-esima rep del gesto activo

        # ── Estado interno ─────────────────────────────────────
        self._state                = State.IDLE
        self._phase_start          = 0.0
        self._current_set          = 0     # 0-indexed
        self._gesture_index_in_set = 0
        self._gestures_in_set:list = []
        self._rep_counters:  dict  = {}    # {gesture_id: total_reps}
        self._cal_done_emitted     = False

        self._timer = QTimer()
        self._timer.setInterval(50)        # 20 Hz
        self._timer.timeout.connect(self._on_tick)

    # ── Propiedades calculadas (thread-safe via GIL) ───────────

    @property
    def cal_phase(self) -> str | None:
        """Devuelve la fase de calibracion activa o None."""
        if self._state == State.CALIBRATION_NOISE: return "noise"
        if self._state == State.CALIBRATION_MVC:   return "mvc"
        return None

    @property
    def state(self) -> State:
        return self._state

    # ── API publica ────────────────────────────────────────────

    def start(self):
        """Iniciar la sesion completa (calibracion + bloques)."""
        self._current_set          = 0
        self._gesture_index_in_set = 0
        self._gestures_in_set      = []
        self._rep_counters         = {g["id"]: 0 for g in self._base_gestures}
        self._cal_done_emitted     = False
        self.stimulus              = 0
        self.recording             = False
        self.set_id                = 0
        self.repetition_id         = 0
        self._enter(State.CALIBRATION_NOISE)
        self._timer.start()

    def shuffle_gestures(self) -> list:
        """Genera una permutacion aleatoria completa de los gestos.

        Garantiza que todos los gestos aparezcan exactamente una vez
        por Set (Protocolo de Bloques Aleatorizados).
        Se llama automaticamente al inicio de cada Set.

        Returns:
            Lista de gestos en orden aleatorio sin repeticion.
        """
        shuffled = list(self._base_gestures)
        random.shuffle(shuffled)
        return shuffled

    # ── Maquina de estados ─────────────────────────────────────

    def _enter(self, new_state: State):
        self._state       = new_state
        self._phase_start = time.time()

        if new_state == State.CALIBRATION_NOISE:
            self.recording = False
            self.stimulus  = 0

        elif new_state == State.CALIBRATION_MVC:
            self.recording = False
            self.stimulus  = 0

        elif new_state == State.TRANSITION:
            self.recording = False
            self.stimulus  = 0
            # Emitir calibration_done al entrar a la transicion
            if not self._cal_done_emitted:
                self._cal_done_emitted = True
                self.calibration_done.emit()

        elif new_state == State.REST:
            self.stimulus  = 0
            self.recording = True

            # Fallback: emitir calibration_done si no se emitio en TRANSITION
            if not self._cal_done_emitted:
                self._cal_done_emitted = True
                self.calibration_done.emit()

        elif new_state == State.GESTURE_ACTIVE:
            g = self._gestures_in_set[self._gesture_index_in_set]
            self.stimulus  = g["id"]
            self.recording = True
            self.set_id        = self._current_set + 1
            self.repetition_id = self._rep_counters[g["id"]]

        elif new_state == State.DONE:
            self.stimulus  = 0
            self.recording = False
            self._timer.stop()

        self._emit_state()

    def _on_tick(self):
        """Tick del timer a 20 Hz — maneja transiciones temporales."""
        elapsed = time.time() - self._phase_start

        if self._state == State.CALIBRATION_NOISE:
            self.progress.emit(elapsed, self._noise_dur)
            if elapsed >= self._noise_dur:
                self._enter(State.CALIBRATION_MVC)

        elif self._state == State.CALIBRATION_MVC:
            self.progress.emit(elapsed, self._mvc_dur)
            if elapsed >= self._mvc_dur:
                # Calibracion completa → pausa de transicion antes del primer set
                self._prepare_set(0)
                self._enter(State.TRANSITION)

        elif self._state == State.TRANSITION:
            self.progress.emit(elapsed, self._transition_dur)
            if elapsed >= self._transition_dur:
                self._enter(State.REST)

        elif self._state == State.REST:
            self.progress.emit(elapsed, self._rest_dur)
            if elapsed >= self._rest_dur:
                # REST terminó → ejecutar el gesto que se estaba anunciando
                self._enter(State.GESTURE_ACTIVE)

        elif self._state == State.GESTURE_ACTIVE:
            self.progress.emit(elapsed, self._gesture_dur)
            if elapsed >= self._gesture_dur:
                self._after_gesture()

    def _prepare_set(self, set_index: int):
        """Preparar un nuevo Set: barajar gestos y apuntar al primero.

        No entra a ningun estado — el llamador decide si entrar a REST.
        """
        self._current_set          = set_index
        self._gesture_index_in_set = 0
        self._gestures_in_set      = self.shuffle_gestures()
        # Incrementar contador de repeticion para el primer gesto
        g = self._gestures_in_set[0]
        self._rep_counters[g["id"]] += 1
        self.set_id        = self._current_set + 1
        self.repetition_id = self._rep_counters[g["id"]]
        print(
            f"[ENGINE] Set {set_index + 1}/{self._n_sets}: "
            f"{[g['name'] for g in self._gestures_in_set]}"
        )

    def _after_gesture(self):
        """Despues de GESTURE_ACTIVE: avanzar al siguiente gesto o terminar."""
        self._gesture_index_in_set += 1

        if self._gesture_index_in_set < len(self._base_gestures):
            # Mas gestos en el mismo Set → REST mostrando el siguiente
            g = self._gestures_in_set[self._gesture_index_in_set]
            self._rep_counters[g["id"]] += 1
            self.repetition_id = self._rep_counters[g["id"]]
            self._enter(State.REST)

        elif self._current_set < self._n_sets - 1:
            # Ultimo gesto del set, pero hay mas sets → preparar y REST
            self._prepare_set(self._current_set + 1)
            self._enter(State.REST)

        else:
            # Ultimo gesto del ultimo set → sesion completa
            self._enter(State.DONE)
            self.finished.emit()

    # ── Emision de estado para la GUI ──────────────────────────

    def _emit_state(self):
        # Durante REST: _gesture_index_in_set apunta al gesto QUE VIENE
        # Durante GESTURE_ACTIVE: apunta al gesto EN EJECUCION
        g = (
            self._gestures_in_set[self._gesture_index_in_set]
            if self._gestures_in_set
            and self._gesture_index_in_set < len(self._gestures_in_set)
            else None
        )

        info = {
            "state":          self._state.value,
            "gesture":        g,
            "next_gesture":   g,     # durante REST, g ES el siguiente gesto
            "set_id":         self._current_set + 1,
            "n_sets":         self._n_sets,
            "gesture_index":  self._gesture_index_in_set,
            "n_gestures":     len(self._base_gestures),
            "repetition_id":  self.repetition_id,
            "rep_counters":   dict(self._rep_counters),
        }
        self.state_changed.emit(self._state.value, info)

    def __repr__(self):
        return (
            f"ProtocolEngine(state={self._state.value}, "
            f"set={self.set_id}/{self._n_sets}, "
            f"rep={self.repetition_id}, stim={self.stimulus})"
        )
