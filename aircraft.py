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


def Cessna172() -> Aircraft:
    """Factory returning the default C172-class parameter set."""
    return Aircraft()


# Convenience: flap lookup tables as dicts for readable access.
def flap_tables(ac: Aircraft):
    return {
        "detents": ac.flap_detents_deg,
        "dCL": ac.flap_dCL,
        "dCD": ac.flap_dCD,
    }
