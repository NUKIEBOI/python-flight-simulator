# Python Flight Simulator

A realistic single-engine airplane flight simulator written in Python, built as
a learning project: clean, modular, well-commented code with a real 6-DOF
flight dynamics model (not a toy arcade approximation).

The aircraft is modeled on a **Cessna 172-class** GA airplane (~1043 kg, 16.2 m²
wing). You take off from a runway, fly with a proper lift/drag/stall model, and
land — all computed from first-principles aerodynamics each physics step.

## Features

- **Real flight dynamics**: angle of attack / sideslip, lift curve with stall
  past the critical angle, induced + parasite drag, propeller thrust vs.
  airspeed, full 6-DOF rigid-body integration at 240 Hz substeps.
- **Stability & control derivatives**: pitch/roll/yaw stiffness, damping, and
  control-surface effectiveness, plus adverse yaw and dihedral effect.
- **Ground handling**: takeoff roll with rolling friction, nosewheel steering,
  wheel brakes, rotation and liftoff when lift exceeds weight.
- **Flaps & trim**: 0/10/20/30° detents with lift/drag increments; elevator
  trim for hands-off flight.
- **3D world (Panda3D)**: procedural terrain with field patchwork and a
  fully marked runway (edge lines, centerline, thresholds, touchdown zones,
  designation numbers), gradient sky dome with sun glow, drifting clouds,
  trees, hangars, windsock, runway edge lights, real-time shadows, and a
  refined airplane with wing struts, gear fairings, prop-blur disc, and
  navigation lights + tail strobe — no downloaded assets. Compact,
  minimal HUD with attitude indicator and throttle bar.
- **Three cameras**: chase, cockpit (with attitude indicator), tower flyby.
- **2D HUD**: airspeed (kts), altitude (ft), heading, vertical speed (fpm),
  throttle %, flap detent, trim, stall warning, on-ground indicator.
- **Autopilot**: tap `F` for a wing-leveler + pitch-hold autopilot that owns
  aileron/elevator while you keep throttle and rudder; tap `F` again to fly.
- **Headless test mode**: scripted takeoff with telemetry, no window needed.

## Install

```bash
pip install -r requirements.txt
```

(Requires Python 3.9+. Only `panda3d` and `numpy` are needed.)

## Run

```bash
python main.py            # launch the 3D simulator (needs a display)
python main.py --test     # headless scripted takeoff, prints telemetry
python tests/test_flight_model.py   # physics verification tests
```

## Controls

| Key | Action |
|---|---|
| `Up` / `W` | Nose down (push) |
| `Down` / `S` | Nose up (pull) |
| `Left` / `A` | Roll left |
| `Right` / `D` | Roll right |
| `Q` / `E` | Rudder left / right (steers nosewheel on ground) |
| `Shift` | Throttle up (hold) |
| `Ctrl` | Throttle down (hold) |
| `F` | Autopilot on/off (wing leveler + pitch hold) |
| `T` / `G` | Elevator trim nose-up / nose-down |
| `B` | Toggle wheel brakes |
| `C` | Camera: chase → cockpit → tower → top-down |
| `P` | Free cam (orbit with `A`/`D`, zoom with `W`/`S`; auto-engages autopilot) |
| `V` | Switch aircraft: C-172 ↔ Viper jet |
| `H` | Hide/show all UI |
| `/` | Toggle help overlay |
| `SPACE` | Pause |
| `R` | Reset flight |
| `Esc` | Quit |

**Takeoff:** hold `Shift` for full throttle, steer with `Q`/`E`,
rotate at 55 kts (pull `S`) — 140 kts in the Viper jet —
then tap `F` for the autopilot wing leveler.

## Aircraft

Two flyables, switched with `V` (position and direction are kept):

- **C-172** — the classic trainer: 55 kt rotation, gentle and forgiving.
- **Viper jet** — F-16-class thrust-to-weight above 1. It holds airspeed in
  any attitude: climb vertically, loop, and maneuver without bleeding speed
  the way the Cessna does. Rotate at 140 kts and mind the thrust.

## Training sortie

Every flight is a tasked mission, DCS-style: the gold objective tracker at the
top of the screen walks you through it, and the tower talks to you on the
radio (blue text).

1. **Take off** from runway 27.
2. **Fly 4 checkpoint gates** — glowing orange rings with beacon poles; the
   next one pulses, cleared ones turn green.
3. **Intercept the bandit drone** patrolling north-east — the tracker shows
   bearing (left/right) and range; get within 220 m for the splash.
4. **Land back on runway 27.** The tower grades your touchdown: *greased it*,
   *firm*, *hard*, or *crash landing*, by sink rate.

Press `R` any time to reset the flight and the mission.

## Architecture

| File | What it does |
|---|---|
| `aircraft.py` | Dataclass of C172-class parameters: mass, inertias, geometry, aero coefficients, engine, control limits, flap tables. |
| `atmosphere.py` | ISA atmosphere: density, pressure, temperature, speed of sound vs. altitude. |
| `flight_model.py` | The dynamics core. **Headless (NumPy only).** Quaternion attitude, airspeed/alpha/beta, lift with stall, drag, prop thrust, control moments with damping, semi-implicit Euler at 240 Hz, flat-runway ground model with friction/steering/brakes. |
| `controls.py` | Pilot control state (elevator/aileron/rudder/throttle/flaps/trim/brakes) with trim logic. |
| `simulator.py` | Panda3D app: procedural terrain + runway texture, sky, lighting, low-poly airplane, 3 cameras, HUD with attitude indicator, keyboard handling. Importing it never starts the main loop. |
| `main.py` | Entry point: `python main.py` runs the sim; `--test` runs a headless scripted takeoff. |
| `tests/test_flight_model.py` | Plain-assert physics tests (no pytest required). |

## How the flight model works

Each physics substep (240 Hz):

1. **Air data**: body velocity → airspeed `V`, angle of attack `α = atan2(w, u)`,
   sideslip `β = asin(v / V)`; air density from the ISA model at current altitude.
2. **Forces**: lift `L = q̄·S·CL(α)` with a lift curve that decays past the
   ~17° stall angle; drag `D = q̄·S·(CD0 + ΔCD_flaps + k·CL²)` plus extra
   separated-flow drag when stalled; sideforce from sideslip. Propeller thrust
   is `min(static thrust, prop power / airspeed) × throttle`.
3. **Moments**: pitch/roll/yaw from stiffness (`Cma`, `Clb`, `Cnb`), control
   deflections, and rate damping (`Cmq`, `Clp`, `Cnr`) — the damping terms are
   what keep the airplane flyable instead of twitchy.
4. **Integration**: forces → body acceleration (including the `ω × v` transport
   term), moments → angular acceleration via the inertia tensor; quaternion
   attitude updated and re-normalized every substep.
5. **Ground**: while on the runway, vertical motion is constrained, rolling
   friction and nosewheel steering apply, and liftoff happens naturally when
   lift exceeds weight.

Everything is deterministic — no randomness — so tests and replays are
repeatable.

## Roadmap ideas

1. **Wind & turbulence**: add a wind vector + Dryden gust model to the air-data
   step; expose it in the HUD.
2. **Autopilot**: a simple PID wing-leveler / altitude-hold using the existing
   `trim_de_for_level()` helper as a starting point.
3. **More airplanes**: parameter sets for a Cub-class taildragger or a twin;
   the whole model already reads everything from `aircraft.py`.
