"""Entry point for the Python Flight Simulator.

Usage:
    python main.py            Launch the 3D simulator (needs a display).
    python main.py --test     Headless scripted takeoff test: no window,
                              prints telemetry. Useful on servers / CI.
"""

import argparse
import math
import sys


def run_test() -> int:
    """Scripted takeoff with no graphics. Returns 0 on success."""
    # NOTE: simulator.py (Panda3D) is deliberately NOT imported here, so this
    # path never touches the graphics stack.
    from controls import ControlState
    from flight_model import FlightModel

    fm = FlightModel()
    ctrl = ControlState()
    ctrl.throttle = 1.0
    ctrl.set_flaps(1)  # 10 degrees, typical short-field takeoff setting

    dt = 1.0 / 60.0
    t = 0.0
    rotated = False
    liftoff_t = None
    print(f"{'t(s)':>6} {'kts':>6} {'alt(ft)':>8} {'vs(fpm)':>8} "
          f"{'pitch(deg)':>10} {'thr':>4}")
    next_print = 0.0
    ok = True
    while t < 60.0:
        V = fm.airspeed()
        kts = V * 1.94384
        # Rotate at 55 kts: pull back, then ease off once climbing.
        if kts >= 55.0 and not rotated:
            rotated = True
        if rotated and fm.on_ground:
            ctrl.elevator = 0.75
        elif not fm.on_ground:
            # Climb attitude: hold ~8 deg nose-up with elevator + trim feel.
            pitch = fm.attitude_euler()[0] * 180.0 / math.pi
            ctrl.elevator = max(-1.0, min(1.0, 0.15 + (8.0 - pitch) * 0.05))
        if not fm.on_ground and liftoff_t is None:
            liftoff_t = t

        fm.step(dt, ctrl)
        t += dt

        state = fm.pos
        if not all(map(math.isfinite, list(state) + list(fm.vel_b)
                       + list(fm.quat) + list(fm.omega))):
            print("FAIL: non-finite state encountered")
            ok = False
            break
        if t >= next_print:
            next_print += 1.0
            pitch = fm.attitude_euler()[0] * 180.0 / math.pi
            import numpy as np
            from flight_model import quat_to_matrix
            vs = float((quat_to_matrix(fm.quat) @ fm.vel_b)[2]) * 196.85
            print(f"{t:6.1f} {kts:6.0f} {state[2]*3.28084:8.0f} "
                  f"{vs:8.0f} {pitch:10.1f} {ctrl.throttle:4.2f}")

    alt_ft = fm.pos[2] * 3.28084
    print(f"\nLiftoff at t={liftoff_t:.1f}s" if liftoff_t else "\nNo liftoff!")
    print(f"Final: alt={alt_ft:.0f} ft, on_ground={fm.on_ground}")
    if liftoff_t is None or fm.on_ground or alt_ft < 100:
        print("FAIL: takeoff did not establish a climb")
        ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Python Flight Simulator")
    parser.add_argument("--test", action="store_true",
                        help="headless scripted takeoff, no window")
    args = parser.parse_args()
    if args.test:
        return run_test()
    from simulator import main as sim_main
    sim_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
