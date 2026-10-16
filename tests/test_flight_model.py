"""Flight-model verification tests. Plain asserts only -- no pytest needed.

Run from the project root:
    python tests/test_flight_model.py
"""

import math
import os
import sys

# Allow running this file directly: put the project root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from aircraft import Cessna172
from controls import ControlState
from flight_model import FlightModel, lift_coefficient, quat_from_pitch


def _finite(fm, label):
    state = list(fm.pos) + list(fm.vel_b) + list(fm.quat) + list(fm.omega)
    assert all(math.isfinite(x) for x in state), f"non-finite state: {label}"


def test_trimmed_level_flight_holds_altitude():
    """At trim elevator, level flight at 50 m/s should hold altitude."""
    fm = FlightModel()
    ac = fm.ac
    V, alt = 50.0, 1000.0
    de = fm.trim_de_for_level(V, alt)
    assert -1.0 <= de <= 1.0, f"trim out of range: {de}"

    # Steady level flight: flight path horizontal, fuselage pitched up by the
    # angle of attack. Body velocity = R^T(pitch) @ (V,0,0)_world.
    rho = fm.atmo.density(alt)
    q = 0.5 * rho * V * V
    CL_need = ac.mass * 9.80665 / (q * ac.wing_area)
    alpha = (CL_need - ac.CL0) / ac.CL_alpha
    fm.reset(pos=[0, 0, alt],
             vel_b=[V * math.cos(alpha), 0.0, -V * math.sin(alpha)],
             quat=quat_from_pitch(alpha), on_ground=False)

    # Throttle matched to drag so speed (and hence lift) stays constant.
    CD = ac.CD0 + ac.induced_k * CL_need * CL_need
    D = q * ac.wing_area * CD
    T_avail = min(ac.thrust_static, ac.prop_power / V)
    ctrl = ControlState()
    ctrl.elevator = de
    ctrl.throttle = D / T_avail
    for _ in range(int(15 * 60)):
        fm.step(1.0 / 60.0, ctrl)
        _finite(fm, "trimmed flight")
    assert abs(fm.pos[2] - alt) < 6.0, f"altitude drift: {fm.pos[2]-alt:.1f} m"
    print(f"  trim de={de:+.3f}, alpha={alpha*57.3:.1f} deg, "
          f"altitude drift={fm.pos[2]-alt:+.2f} m over 15 s")


def test_takeoff_roll_rotates_and_climbs():
    """Full throttle + rotation at 55 kts must produce a positive climb."""
    fm = FlightModel()
    ctrl = ControlState()
    ctrl.throttle = 1.0
    ctrl.set_flaps(1)  # 10 deg

    dt = 1.0 / 60.0
    t = 0.0
    liftoff_t = None
    while t < 45.0:
        kts = fm.airspeed() * 1.94384
        if kts >= 55.0 and fm.on_ground:
            ctrl.elevator = 0.75
        if not fm.on_ground:
            if liftoff_t is None:
                liftoff_t = t
            pitch = fm.attitude_euler()[0] * 180.0 / math.pi
            ctrl.elevator = max(-1.0, min(1.0, 0.15 + (8.0 - pitch) * 0.05))
        fm.step(dt, ctrl)
        t += dt
        _finite(fm, "takeoff")

    assert liftoff_t is not None, "never lifted off"
    assert not fm.on_ground, "still on ground at end"
    assert fm.pos[2] > 30.0, f"weak climb: {fm.pos[2]:.1f} m"
    from flight_model import quat_to_matrix
    vs = float((quat_to_matrix(fm.quat) @ fm.vel_b)[2])
    assert vs > 0.5, f"not climbing at end: {vs:.2f} m/s"
    print(f"  liftoff at t={liftoff_t:.1f}s, final alt={fm.pos[2]:.0f} m, "
          f"vs={vs:.1f} m/s")


def test_sixty_seconds_maneuvering_never_nan():
    """Aggressive control inputs for 60 s must never produce NaN/inf."""
    fm = FlightModel()
    ctrl = ControlState()
    ctrl.throttle = 1.0
    ctrl.set_flaps(1)

    dt = 1.0 / 60.0
    t = 0.0
    seq = 0
    # Scripted maneuver blocks, 5 s each: takeoff, rolls, climbs, descents,
    # flap changes, throttle chops, steep turns, slow flight.
    while t < 60.0:
        kts = fm.airspeed() * 1.94384
        block = int(t // 5) % 12
        if block == 0:
            ctrl.elevator = 0.75 if kts >= 55 and fm.on_ground else 0.1
            ctrl.aileron, ctrl.rudder = 0.0, 0.0
        elif block == 1:
            ctrl.elevator, ctrl.aileron = 0.1, 0.8
        elif block == 2:
            ctrl.elevator, ctrl.aileron = 0.1, -0.8
        elif block == 3:
            ctrl.elevator, ctrl.aileron = 0.6, 0.0
        elif block == 4:
            ctrl.elevator, ctrl.aileron = -0.5, 0.0
        elif block == 5:
            ctrl.set_flaps(3)
            ctrl.elevator, ctrl.throttle = 0.3, 0.4
        elif block == 6:
            ctrl.set_flaps(0)
            ctrl.throttle = 1.0
            ctrl.rudder = 0.9
        elif block == 7:
            ctrl.rudder, ctrl.aileron, ctrl.elevator = -0.9, 0.5, 0.2
        elif block == 8:
            ctrl.rudder, ctrl.aileron = 0.0, 0.0
            ctrl.elevator = 1.0  # yank to stall
        elif block == 9:
            ctrl.elevator = -0.6  # recover
        elif block == 10:
            ctrl.throttle = 0.0  # chop power
            ctrl.elevator = 0.4
        else:
            ctrl.throttle = 1.0
            ctrl.elevator, ctrl.aileron = 0.0, 0.6
        fm.step(dt, ctrl)
        t += dt
        _finite(fm, f"t={t:.1f}")
        # Keep it sane: if we somehow dug a hole, that is a test failure too.
        assert fm.pos[2] > -50.0, f"crashed through the world at t={t:.1f}"
        seq += 1
    print(f"  {seq} steps, all finite; final alt={fm.pos[2]:.0f} m")


def test_stall_reduces_lift():
    """Past the critical angle, the wing must make LESS lift than before it."""
    ac = Cessna172()
    cl_pre = lift_coefficient(0.25, 0.0, ac)    # ~14 deg, flying
    cl_post = lift_coefficient(0.45, 0.0, ac)  # ~26 deg, stalled
    assert cl_post < cl_pre, f"stall broken: {cl_post=} >= {cl_pre=}"
    print(f"  CL(14 deg)={cl_pre:.3f} > CL(26 deg, stalled)={cl_post:.3f}")


TESTS = [
    ("trimmed level flight holds altitude", test_trimmed_level_flight_holds_altitude),
    ("takeoff roll rotates and climbs", test_takeoff_roll_rotates_and_climbs),
    ("60 s maneuvering never NaN", test_sixty_seconds_maneuvering_never_nan),
    ("stall reduces lift", test_stall_reduces_lift),
]


def main():
    failures = 0
    for name, fn in TESTS:
        print(f"TEST: {name}")
        try:
            fn()
        except AssertionError as e:
            failures += 1
            print(f"  FAILED: {e}")
        except Exception as e:  # noqa: BLE001 - report anything unexpected
            failures += 1
            print(f"  ERROR: {type(e).__name__}: {e}")
        else:
            print("  ok")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
