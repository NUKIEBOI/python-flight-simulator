"""Pilot control state.

Conventions (matching flight_model.py):
  elevator  -1..+1   (+1 = pull back / nose up)
  aileron   -1..+1   (+1 = roll right)
  rudder    -1..+1   (+1 = right rudder / nose right)
  throttle   0..+1
  trim      -1..+1   (+ = nose-up trim)

Trim logic: real trim tabs re-zero the elevator so the airplane holds an
attitude hands-off. Here trim is modelled as an offset added to the elevator
command, with authority limited to a fraction of full elevator travel
(see Aircraft.trim_authority). Positive trim adds nose-up elevator.
"""

from dataclasses import dataclass, field


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class ControlState:
    elevator: float = 0.0
    aileron: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.0
    flap_index: int = 0
    trim: float = 0.0
    brakes: bool = False

    FLAP_DETENTS = (0, 10, 20, 30)  # degrees; index selects detent

    # --- Flaps ----------------------------------------------------------
    def flap_degrees(self) -> int:
        return self.FLAP_DETENTS[self.flap_index]

    def cycle_flaps(self) -> int:
        """Advance one detent, wrapping 30 -> 0. Returns new flap angle."""
        self.flap_index = (self.flap_index + 1) % len(self.FLAP_DETENTS)
        return self.flap_degrees()

    def set_flaps(self, index: int):
        self.flap_index = int(clamp(index, 0, len(self.FLAP_DETENTS) - 1))

    # --- Trim -----------------------------------------------------------
    def trim_up(self, amount: float = 0.05):
        """Nose-up trim step (default 5% of trim range)."""
        self.trim = clamp(self.trim + amount, -1.0, 1.0)

    def trim_down(self, amount: float = 0.05):
        """Nose-down trim step."""
        self.trim = clamp(self.trim - amount, -1.0, 1.0)

    def reset_trim(self):
        self.trim = 0.0

    # --- Combined surface positions -------------------------------------
    def elevator_total(self, trim_authority: float = 0.6) -> float:
        """Elevator command including trim offset, clamped to [-1, 1]."""
        return clamp(self.elevator + self.trim * trim_authority, -1.0, 1.0)

    # --- Housekeeping ---------------------------------------------------
    def reset(self):
        self.elevator = 0.0
        self.aileron = 0.0
        self.rudder = 0.0
        self.throttle = 0.0
        self.flap_index = 0
        self.trim = 0.0
        self.brakes = False

    def normalized(self) -> "ControlState":
        """Return a copy with every channel clamped to its legal range."""
        c = ControlState(
            elevator=clamp(self.elevator, -1, 1),
            aileron=clamp(self.aileron, -1, 1),
            rudder=clamp(self.rudder, -1, 1),
            throttle=clamp(self.throttle, 0, 1),
            flap_index=int(clamp(self.flap_index, 0, len(self.FLAP_DETENTS) - 1)),
            trim=clamp(self.trim, -1, 1),
            brakes=bool(self.brakes),
        )
        return c
