"""Aircraft parameters for a Cessna 172-class single-engine GA airplane.

All numbers are plausible approximations for a C172S at ~1043 kg (2300 lb),
documented below. They are good enough for a believable desktop simulator and
for learning how each parameter shapes handling -- they are NOT certified data.

Sources / basis (from memory of public C172S specs):
  - MTOW 1157 kg (2550 lb); typical training weight ~1043 kg used here.
  - Wing area 16.2 m^2 (174 ft^2), span 11.0 m (36 ft) -> aspect ratio ~7.5.
  - Engine: Lycoming IO-360-L2A, 180 hp (134 kW) driving a fixed-pitch prop.
  - Stall speed ~48 KIAS flaps-down implies CLmax ~2.0 with full flaps.
  - Moments of inertia are textbook C172-class values (Ixx~1285, Iyy~1825,
    Izz~2666 kg m^2).
  - Stability/control derivatives are typical light-aircraft values chosen so
    the model is positively stable in pitch, spirally neutral-ish in roll/yaw,
    and stalls near 15-17 degrees angle of attack.

Sign conventions used by flight_model.py (body frame: x forward, y right,
z up -- right-handed):
  - elevator control +1 = pull back (nose up).  Cmde < 0 gives nose-up moment.
  - aileron control +1 = roll right.  Positive p (about +x) is roll LEFT, so
    Clda < 0 gives right roll for positive input.
  - rudder control +1 = right rudder (nose right).  Positive r (about +z)
    takes the nose right, so Cndr > 0.
  - Pitch moment about +y: positive = nose DOWN (right-hand rule about the
    right-pointing y axis).  Hence Cma > 0 is statically stable.
"""

from dataclasses import dataclass, field


@dataclass
class Aircraft:
    """Complete parameter set for the simulated airplane."""

    name: str = "Cessna 172-class (approx)"

    # --- Mass & inertia -------------------------------------------------
    mass: float = 1043.0          # kg
    Ixx: float = 1285.0           # kg m^2, roll inertia
    Iyy: float = 1825.0           # kg m^2, pitch inertia
    Izz: float = 2666.0           # kg m^2, yaw inertia

    # --- Wing geometry --------------------------------------------------
    wing_area: float = 16.2       # m^2
    wing_span: float = 11.0       # m
    wing_chord: float = 1.474     # m, mean aerodynamic chord (area/span)

    # --- Lift / drag ----------------------------------------------------
    # Lift curve: CL(alpha) = CL0 + CLa*alpha (+ flap increment), linear until
    # alpha_crit, then stalling (see flight_model.lift_coefficient).
    # CL0 = 0.25 bakes in the cambered wing's lift at zero alpha.
    # With these numbers the clean stall lands near 50 KIAS and full-flap
    # stall near 40 KIAS at 1043 kg -- close to published C172S figures
    # (53/48 KCAS at max gross).
    CL0: float = 0.25             # lift coeff at alpha = 0
    CL_alpha: float = 4.6         # per rad, lift-curve slope
    alpha_crit: float = 0.28      # rad (~16 deg), stall angle
    CD0: float = 0.032            # parasite drag coefficient
    induced_k: float = 0.0568     # induced drag factor 1/(pi*AR*e)
    CY_beta: float = -0.5         # per rad, sideforce due to sideslip

    # --- Pitch ----------------------------------------------------------
    Cm0: float = 0.02             # pitching moment at alpha = 0
    Cma: float = 0.9              # per rad, pitch stiffness (stable > 0 here)
    Cmq: float = -12.0            # per rad, pitch damping
    Cmde: float = -1.2            # per rad, elevator effectiveness

    # --- Roll -----------------------------------------------------------
    Clb: float = 0.09             # per rad, dihedral effect
    Clp: float = -0.45            # per rad, roll damping
    Clda: float = -0.18           # per rad, aileron effectiveness
    Cldr: float = -0.02           # per rad, roll due to rudder

    # --- Yaw ------------------------------------------------------------
    Cnb: float = 0.12             # per rad, weathervane stability
    Cnr: float = -0.15            # per rad, yaw damping
    Cnda: float = -0.02           # per rad, adverse yaw from ailerons
    Cndr: float = 0.10            # per rad, rudder effectiveness

    # --- Propulsion -----------------------------------------------------
    # Simple propeller model: thrust is the lesser of a static-thrust cap
    # (dominates at low speed) and available power / airspeed.
    thrust_static: float = 3200.0  # N at full throttle, zero airspeed
    prop_power: float = 80500.0    # W, effective propulsive power (~60% of
                                   # 134 kW shaft power after prop/install losses)

    # --- Control limits -------------------------------------------------
    elevator_limit: float = 0.40  # rad of surface deflection
    aileron_limit: float = 0.35   # rad
    rudder_limit: float = 0.35    # rad
    trim_authority: float = 0.6   # trim adds up to +/-60% of elevator travel

    # --- Flaps: detents in degrees, with lift/drag increments -----------
    flap_detents_deg: tuple = (0, 10, 20, 30)
    flap_dCL: tuple = (0.0, 0.30, 0.60, 0.90)
    flap_dCD: tuple = (0.0, 0.020, 0.045, 0.080)

    # --- Game assists ---------------------------------------------------
    # Scales the built-in wing-leveler autopilot gains for this airframe
    # (faster airframes need gentler corrections).
    ap_gain: float = 1.0


def Cessna172() -> Aircraft:
    """Factory returning the default C172-class parameter set."""
    return Aircraft()


@dataclass
class ViperJet(Aircraft):
    """F-16-class flyable jet.

    Thrust/weight > 1, so it holds airspeed in any attitude -- vertical
    climbs, loops and hard maneuvering without bleeding speed the way the
    C172 does. Control derivatives are detuned for high-speed keyboard
    flying (a real jet's raw sensitivity would be unflyable here).
    """

    name: str = "Viper Jet"

    # --- Mass & inertia -------------------------------------------------
    mass: float = 8500.0
    Ixx: float = 9000.0
    Iyy: float = 16000.0
    Izz: float = 19000.0

    # --- Wing geometry --------------------------------------------------
    wing_area: float = 27.9
    wing_span: float = 9.45
    wing_chord: float = 2.95

    # --- Lift / drag ----------------------------------------------------
    CL0: float = 0.20
    CL_alpha: float = 4.0
    alpha_crit: float = 0.30
    CD0: float = 0.028
    induced_k: float = 0.12
    CY_beta: float = -0.5

    # --- Pitch ----------------------------------------------------------
    Cm0: float = 0.02
    Cma: float = 0.9
    Cmq: float = -14.0
    Cmde: float = -0.20          # detuned for high-q keyboard flying

    # --- Roll -----------------------------------------------------------
    Clb: float = 0.09
    Clp: float = -0.50
    Clda: float = -0.05          # detuned
    Cldr: float = -0.02

    # --- Yaw ------------------------------------------------------------
    Cnb: float = 0.12
    Cnr: float = -0.18
    Cnda: float = -0.02
    Cndr: float = 0.03           # detuned

    # --- Propulsion -----------------------------------------------------
    # Near-constant 110 kN military thrust at any airspeed.
    thrust_static: float = 110000.0
    prop_power: float = 60_000_000.0

    # --- Game assists ---------------------------------------------------
    ap_gain: float = 0.4


def make_viper_jet() -> ViperJet:
    """Factory returning the ViperJet parameter set."""
    return ViperJet()


@dataclass
class BoeingJet(Aircraft):
    """Boeing 737-class twin-engine airliner.

    Heavy and gentle: modest thrust/weight, so it climbs like an airliner,
    not a fighter. Control derivatives are detuned for high-speed keyboard
    flying. Game-scaled (not full 37 m) so it sits sensibly on the gear.
    """

    name: str = "Boeing 737-class"

    # --- Mass & inertia -------------------------------------------------
    mass: float = 42000.0
    Ixx: float = 280000.0
    Iyy: float = 520000.0
    Izz: float = 650000.0

    # --- Wing geometry --------------------------------------------------
    wing_area: float = 125.0
    wing_span: float = 35.0
    wing_chord: float = 3.6

    # --- Lift / drag ----------------------------------------------------
    CL0: float = 0.25
    CL_alpha: float = 4.2
    alpha_crit: float = 0.26
    CD0: float = 0.030
    induced_k: float = 0.045
    CY_beta: float = -0.5

    # --- Pitch ----------------------------------------------------------
    Cm0: float = 0.0
    Cma: float = 0.9
    Cmq: float = -16.0
    Cmde: float = -0.50            # detuned for high-q keyboard flying

    # --- Roll -----------------------------------------------------------
    Clb: float = 0.09
    Clp: float = -0.55
    Clda: float = -0.08           # detuned (airliners roll gently)
    Cldr: float = -0.02

    # --- Yaw ------------------------------------------------------------
    Cnb: float = 0.12
    Cnr: float = -0.20
    Cnda: float = -0.02
    Cndr: float = 0.035           # detuned

    # --- Propulsion -----------------------------------------------------
    # Near-constant 240 kN total at any airspeed (twin engines).
    thrust_static: float = 240000.0
    prop_power: float = 150_000_000.0

    # --- Game assists ---------------------------------------------------
    ap_gain: float = 0.35


def make_boeing_jet() -> BoeingJet:
    """Factory returning the BoeingJet parameter set."""
    return BoeingJet()


# Convenience: flap lookup tables as dicts for readable access.
def flap_tables(ac: Aircraft):
    return {
        "detents": ac.flap_detents_deg,
        "dCL": ac.flap_dCL,
        "dCD": ac.flap_dCD,
    }
