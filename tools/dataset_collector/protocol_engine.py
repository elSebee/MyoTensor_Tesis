"""
=============================================================
 protocol_engine.py — Maquina de estados del protocolo
=============================================================
 Controla la secuencia: COUNTDOWN → [GESTURE → REST] × reps × gestos → DONE

 Estados:
   IDLE      — esperando que el usuario presione start
   COUNTDOWN — cuenta regresiva antes de empezar (3, 2, 1...)
   GESTURE   — muestra la instruccion, graba con stimulus = gesture.id
   REST      — muestra "DESCANSA", graba con stimulus = 0
   DONE      — protocolo completado

 Propiedades thread-safe (leidas por DataWriter desde hilo UDP):
   recording   — True durante GESTURE y REST
   stimulus    — id del gesto activo (0 durante reposo)
   repetition  — numero de repeticion actual (1-indexed)

 Emite señales Qt para actualizar la GUI:
   state_changed(str, dict) — estado y metadata
   progress(float, float)   — elapsed_s, total_s
   finished()               — protocolo completado
=============================================================
"""

import time
from enum import Enum

from PyQt5.QtCore import QObject, QTimer, pyqtSignal


class State(Enum):
    IDLE = "idle"
    COUNTDOWN = "countdown"
    GESTURE = "gesture"
    REST = "rest"
    DONE = "done"


class ProtocolEngine(QObject):
    """Maquina de estados que controla el protocolo de adquisicion."""

    # Señales Qt
    state_changed = pyqtSignal(str, object)    # (state_name, info_dict)
    progress = pyqtSignal(float, float)        # (elapsed_s, total_s)
    finished = pyqtSignal()                    # protocolo completado

    def __init__(self, config: dict):
        super().__init__()
        self.gestures = config["gestures"]
        self.reps = config["protocol"]["repetitions"]
        self.gesture_dur = config["protocol"]["gesture_duration_s"]
        self.rest_dur = config["protocol"]["rest_duration_s"]
        self.countdown_s = config["protocol"]["countdown_s"]

        # ── Estado publico (leido desde hilo UDP — GIL safe) ───
        self.recording = False
        self.stimulus = 0
        self.repetition = 0
        self.gesture_index = 0

        # ── Estado interno ─────────────────────────────────────
        self._state = State.IDLE
        self._phase_start = 0.0
        self._countdown_digit = -1   # digito visible actual (3, 2, 1)

        self._timer = QTimer()
        self._timer.setInterval(50)   # 20 Hz — suficiente para UI + timing
        self._timer.timeout.connect(self._on_tick)

    @property
    def state(self) -> State:
        return self._state

    def start(self):
        """Iniciar el protocolo desde la cuenta regresiva."""
        self.gesture_index = 0
        self.repetition = 1
        self.stimulus = 0
        self.recording = False
        self._countdown_digit = -1
        self._enter_state(State.COUNTDOWN)
        self._timer.start()

    # ── Transiciones de estado ─────────────────────────────────

    def _enter_state(self, new_state: State):
        self._state = new_state
        self._phase_start = time.time()

        if new_state == State.GESTURE:
            g = self.gestures[self.gesture_index]
            self.stimulus = g["id"]
            self.recording = True

        elif new_state == State.REST:
            self.stimulus = 0
            self.recording = True

        elif new_state == State.COUNTDOWN:
            self.stimulus = 0
            self.recording = False

        elif new_state == State.DONE:
            self.stimulus = 0
            self.recording = False
            self._timer.stop()

        self._emit_state()

    def _enter_gesture(self):
        self._enter_state(State.GESTURE)

    def _enter_rest(self):
        self._enter_state(State.REST)

    def _advance(self):
        """Avanzar al siguiente paso del protocolo."""
        if self.repetition < self.reps:
            # Siguiente repeticion del mismo gesto
            self.repetition += 1
            self._enter_gesture()
        elif self.gesture_index < len(self.gestures) - 1:
            # Siguiente gesto, repeticion 1
            self.gesture_index += 1
            self.repetition = 1
            self._enter_gesture()
        else:
            # Protocolo completo
            self._enter_state(State.DONE)
            self.finished.emit()

    # ── Tick del timer (20 Hz) ─────────────────────────────────

    def _on_tick(self):
        elapsed = time.time() - self._phase_start

        if self._state == State.COUNTDOWN:
            if elapsed >= self.countdown_s:
                self._enter_gesture()
            else:
                self.progress.emit(elapsed, self.countdown_s)
                # Solo emitir state_changed cuando el digito visible cambia (3→2→1)
                # Evita que la GUI resetee la barra de progreso en cada tick
                new_digit = max(1, int(self.countdown_s - elapsed + 0.99))
                if new_digit != self._countdown_digit:
                    self._countdown_digit = new_digit
                    self._emit_state()

        elif self._state == State.GESTURE:
            if elapsed >= self.gesture_dur:
                self._enter_rest()
            else:
                self.progress.emit(elapsed, self.gesture_dur)

        elif self._state == State.REST:
            if elapsed >= self.rest_dur:
                self._advance()
            else:
                self.progress.emit(elapsed, self.rest_dur)

    # ── Emision de estado para la GUI ──────────────────────────

    def _emit_state(self):
        gesture = (self.gestures[self.gesture_index]
                   if self.gesture_index < len(self.gestures) else None)

        elapsed = time.time() - self._phase_start

        info = {
            "gesture": gesture,
            "repetition": self.repetition,
            "total_reps": self.reps,
            "gesture_index": self.gesture_index,
            "total_gestures": len(self.gestures),
            "state": self._state.value,
        }

        if self._state == State.COUNTDOWN:
            remaining = max(0, int(self.countdown_s - elapsed + 0.99))
            info["countdown"] = remaining

        self.state_changed.emit(self._state.value, info)

    # ── Info para debug ────────────────────────────────────────

    def __repr__(self):
        return (f"ProtocolEngine(state={self._state.value}, "
                f"gesture={self.gesture_index}, rep={self.repetition}, "
                f"stim={self.stimulus}, rec={self.recording})")
