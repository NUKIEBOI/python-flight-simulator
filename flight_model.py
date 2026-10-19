"""6-DOF flight dynamics for a light GA airplane.

HEADLESS: this module imports only NumPy (plus aircraft/atmosphere/controls).
No graphics here, so the physics can be unit-tested and run without a window.

Frames
------
* World: x east-ish, y north-ish, z UP (meters). Flat runway at z = 0.
* Body: x forward, y right, z up (right-handed).
  - Angular rates (p, q, r) about (x, y, z).
  - Positive q (pitch about +y) is nose-DOWN (right-hand rule about the
    right-pointing y axis). Positive r (about +z) is nose-RIGHT.
  - Attitude stored as quaternion [w, x, y, z], body -> world.

Integration: semi-implicit (symplectic) Euler with fixed 240 Hz substeps.
That is stable for these dynamics, exactly deterministic (no randomness
anywhere), and far easier to keep NaN-free around ground contact than RK4.

Ground model (simple but complete enough to take off and land):
  - Flat runway at z = 0; the CG sits at GEAR_HEIGHT when on the wheels.
  - Rolling friction opposes horizontal motion (higher with wheel brakes).
  - Strong lateral tire grip damps sideways velocity; rudder steers the
    nosewheel through a yaw moment proportional to weight-on-wheels.
  - A pitch spring keeps the nosewheel on the ground until the pilot rotates;
    roll is held wings-level by the gear.
  - Liftoff when aerodynamic lift exceeds weight; touchdown when the wheels
    reach the runway while descending.
"""

import math

import numpy as np

from aircraft import Cessna172
from atmosphere import Atmosphere
from controls import ControlState

G = 9.80665
GEAR_HEIGHT = 1.0      # m, CG height when sitting on the runway
SUBSTEP_HZ = 240.0     # physics substeps per second


# ---------------------------------------------------------------------------
# Quaternion helpers (scalar-first [w, x, y, z])
# ---------------------------------------------------------------------------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12 or not math.isfinite(n):
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Body -> world rotation matrix for a right-handed frame."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def quat_from_pitch(theta: float) -> np.ndarray:
    """Quaternion for a pure nose-up pitch of theta radians."""
    # Nose-up is a NEGATIVE rotation about +y (right-hand rule), see module doc.
    return quat_normalize(np.array([math.cos(theta / 2), 0.0,
                                   -math.sin(theta / 2), 0.0]))


# ---------------------------------------------------------------------------
# Aerodynamics helpers
# ---------------------------------------------------------------------------

def lift_coefficient(alpha: float, flap_dcl: float, ac) -> float:
    """Lift coefficient with a stall past the critical angle.

    alpha follows the standard convention: positive when the nose is ABOVE
    the relative wind (alpha = atan2(-w, u) in our z-up body frame).
    Linear (CL0 + CLa*alpha) up to alpha_crit, then decays toward ~62% of
    the peak over the next 0.35 rad. Negative-alpha stall is mirrored.
    """
    alpha = max(-1.2, min(1.2, alpha))
    cl_lin = ac.CL0 + ac.CL_alpha * alpha + flap_dcl
    if alpha <= ac.alpha_crit and alpha >= -0.25:
        return cl_lin
    if alpha > ac.alpha_crit:
        cl_max = ac.CL0 + ac.CL_alpha * ac.alpha_crit + flap_dcl
        excess = min(alpha - ac.alpha_crit, 0.35)
        return cl_max * (1.0 - 0.38 * (excess / 0.35))
    # Deep negative alpha: decay the (negative) linear value toward zero.
    cl_ref = ac.CL0 + ac.CL_alpha * -0.25 + flap_dcl
    excess = min(-0.25 - alpha, 0.35)
    return cl_ref * (1.0 - 0.38 * (excess / 0.35))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Flight model
# ---------------------------------------------------------------------------

class FlightModel:
    def __init__(self, aircraft=None):
        self.ac = aircraft or Cessna172()
        self.atmo = Atmosphere()
        self.time = 0.0
        self.on_ground = True
        self.reset()

    # -- state ------------------------------------------------------------
    def reset(self, pos=None, vel_b=None, quat=None, omega=None,
              on_ground: bool = True):
        self.pos = np.array([0.0, 0.0, GEAR_HEIGHT]) if pos is None else np.array(pos, float)
        self.vel_b = np.zeros(3) if vel_b is None else np.array(vel_b, float)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0]) if quat is None else quat_normalize(np.array(quat, float))
        self.omega = np.zeros(3) if omega is None else np.array(omega, float)
        self.on_ground = on_ground
        self.time = 0.0
        self._last = {}  # info dict from the most recent force evaluation

    # -- public helpers ---------------------------------------------------
    def airspeed(self) -> float:
        return float(np.linalg.norm(self.vel_b))

    def attitude_euler(self):
        """Return (pitch_rad nose-up+, roll_rad left-positive, yaw_rad)."""
        R = quat_to_matrix(self.quat)
        pitch = math.asin(_clamp(R[2, 0], -1.0, 1.0))   # nose-up positive
        roll = math.asin(_clamp(R[2, 1], -1.0, 1.0))    # left-roll positive
        yaw = math.atan2(R[1, 0], R[0, 0])
        return pitch, roll, yaw

    def info(self) -> dict:
        """Last computed flight info (alpha, beta, V, stalled, q, ...)."""
        return dict(self._last)

    def trim_de_for_level(self, V: float, alt_m: float, flap_index: int = 0) -> float:
        """Elevator setting (-1..1) that zeroes pitch moment in level flight.

        Useful for tests and (later) autopilots. Bisection on the elevator
        channel; the moment is linear in de so this converges in ~40 iters.
        """
        ac = self.ac
        rho = self.atmo.density(alt_m)
        q = 0.5 * rho * V * V
        flap_dcl = ac.flap_dCL[flap_index]
        CL_need = ac.mass * G / (q * ac.wing_area)
        alpha = (CL_need - ac.CL0 - flap_dcl) / ac.CL_alpha

        def cm(de):
            return ac.Cm0 + ac.Cma * alpha + ac.Cmde * de * ac.elevator_limit

        lo, hi = -1.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            (lo, hi) = (mid, hi) if cm(mid) > 0 else (lo, mid)
        return 0.5 * (lo + hi)

    # -- core -------------------------------------------------------------
    def step(self, dt: float, ctrl: ControlState):
        """Advance the simulation by dt seconds (clamped, deterministic)."""
        dt = _clamp(float(dt), 0.0, 0.1)
        n = max(1, int(round(dt * SUBSTEP_HZ)))
        h = dt / n
        for _ in range(n):
            self._substep(h, ctrl)
        self.time += dt

    # -- internals --------------------------------------------------------
    def _forces_moments(self, ctrl: ControlState):
        ac = self.ac
        R = quat_to_matrix(self.quat)

        rho = self.atmo.density(self.pos[2])
        u, v, w = self.vel_b
        V = math.sqrt(u * u + v * v + w * w)
        V_safe = max(V, 0.5)

        # Standard alpha: positive = nose ABOVE the relative wind.
        # (z-up body frame, so alpha = atan2(-w, u).)
        alpha = _clamp(math.atan2(-w, u), -1.2, 1.2)
        beta = math.asin(_clamp(v / V_safe, -1.0, 1.0)) if V > 0.5 else 0.0
        qbar = 0.5 * rho * V * V
        S, b, cbar = ac.wing_area, ac.wing_span, ac.wing_chord

        flap_idx = int(_clamp(ctrl.flap_index, 0, len(ac.flap_detents_deg) - 1))
        flap_dcl = ac.flap_dCL[flap_idx]
        flap_dcd = ac.flap_dCD[flap_idx]

        # Control surface deflections [rad]
        de = ctrl.elevator_total(ac.trim_authority) * ac.elevator_limit
        da = _clamp(ctrl.aileron, -1, 1) * ac.aileron_limit
        dr = _clamp(ctrl.rudder, -1, 1) * ac.rudder_limit

        # --- Aerodynamic forces (body frame) ---
        stalled = abs(alpha) > ac.alpha_crit and V > 5.0
        CL = lift_coefficient(alpha, flap_dcl, ac)
        CD = ac.CD0 + flap_dcd + ac.induced_k * CL * CL
        if stalled:  # separated flow adds drag
            CD += 0.15 * min(1.0, (abs(alpha) - ac.alpha_crit) / 0.2)
        L = qbar * S * CL
        D = qbar * S * CD
        Y = qbar * S * ac.CY_beta * beta

        ca, sa = math.cos(alpha), math.sin(alpha)
        cb, sb = math.cos(beta), math.sin(beta)
        # e_v: unit airflow vector in body coords. With alpha = atan2(-w, u):
        #   u = V*cos(a)*cos(b), v = V*sin(b), w = -V*sin(a)*cos(b).
        # e_L: lift direction. World-vertical lift (0,0,L), expressed in a
        #   body pitched nose-up by theta = alpha, is L*(sin a, 0, cos a):
        #   R_y(-a) @ (L sin a, 0, L cos a) = (0, 0, L). So lift tilts
        #   slightly FORWARD in body coords at positive alpha.
        ev = np.array([ca * cb, sb, -sa * cb])
        eL = np.array([sa, 0.0, ca])
        F_aero = -D * ev + L * eL + np.array([0.0, Y, 0.0])

        # --- Propulsion: thrust along body x ---
        thr = _clamp(ctrl.throttle, 0.0, 1.0)
        T = thr * min(ac.thrust_static, ac.prop_power / max(V, 6.0))
        F_aero[0] += T

        # --- Weight ---
        F_grav_b = R.T @ np.array([0.0, 0.0, -ac.mass * G])

        # --- Moments ---
        p, qrate, r = self.omega
        v_damp = max(V, 2.0)  # keep damping derivatives finite at low speed
        # Near/fully stalled wing -> the tail flies in separated, turbulent
        # flow, so the elevator progressively loses authority. Without this,
        # full back-elevator could pin the nose in a deep stall forever;
        # now the nose drops and the wing recovers, like the real thing.
        _a_onset = ac.alpha_crit * 0.85
        if abs(alpha) > _a_onset and V > 5.0:
            _depth = min(1.0, (abs(alpha) - _a_onset) / 0.15)
            elev_auth = 1.0 - 0.65 * _depth
        else:
            elev_auth = 1.0
        Cm = (ac.Cm0 + ac.Cma * alpha
              + ac.Cmq * (qrate * cbar / (2 * v_damp))
              + ac.Cmde * de * elev_auth)
        Cl = (ac.Clb * beta
              + ac.Clp * (p * b / (2 * v_damp))
              + ac.Clda * da + ac.Cldr * dr)
        Cn = (ac.Cnb * beta
              + ac.Cnr * (r * b / (2 * v_damp))
              + ac.Cnda * da + ac.Cndr * dr)
        M_aero = qbar * S * np.array([b * Cl, cbar * Cm, b * Cn])

        F = F_aero + F_grav_b
        M = M_aero

        # --- Ground model ---
        weight_on_wheels = 0.0
        if self.on_ground:
            F_w = R @ F_aero                      # aero force in world frame
            lift_up = F_w[2]
            weight_on_wheels = max(0.0, ac.mass * G - lift_up)

            v_w = R @ self.vel_b
            vh = math.sqrt(v_w[0] ** 2 + v_w[1] ** 2)

            # Rolling friction (world frame -> body)
            mu = 0.55 if ctrl.brakes else 0.035
            if vh > 0.05 and weight_on_wheels > 0:
                fdir = -v_w[:2] / vh
                F_fric_w = np.array([fdir[0] * mu * weight_on_wheels,
                                     fdir[1] * mu * weight_on_wheels, 0.0])
            else:
                F_fric_w = np.zeros(3)
            F += R.T @ F_fric_w

            # Lateral tire grip: kill sideways sliding (world frame).
            # Heading direction (horizontal projection of body x-axis):
            hx, hy = R[0, 0], R[1, 0]
            hnorm = math.sqrt(hx * hx + hy * hy) + 1e-9
            hx, hy = hx / hnorm, hy / hnorm
            v_along = v_w[0] * hx + v_w[1] * hy
            v_lat = np.array([v_w[0] - v_along * hx,
                              v_w[1] - v_along * hy, 0.0])
            F_lat_w = -ac.mass * 3.0 * v_lat * min(1.0, weight_on_wheels / (ac.mass * G))
            F += R.T @ F_lat_w

            # Nosewheel steering from rudder + yaw damping on the ground.
            pitch, roll, _yaw = self._euler_from_R(R)
            M[2] += dr * weight_on_wheels * 1.5 - r * 3000.0
            # Pitch attitude on the ground is held by the gear geometry: a
            # stiff spring toward a slight nose-up taxi attitude. The pilot
            # overcomes it with elevator to rotate - but only once there is
            # enough airflow (around 55 kts), so the nose can't be hauled
            # skyward mid-roll into a premature stall/mush. (Positive M_y =
            # nose-down, so the restoring moment carries a + sign.)
            M[1] += (pitch - 0.02) * ac.mass * G * 2.5 - qrate * 3000.0
            # Gear holds the wings level.
            M[0] += -roll * ac.mass * G * 1.0 - p * 4000.0

        self._last = {
            "alpha": alpha, "beta": beta, "V": V, "rho": rho, "qbar": qbar,
            "CL": CL, "CD": CD, "thrust": T, "stalled": stalled,
            "de_rad": de, "da_rad": da, "dr_rad": dr,
            "weight_on_wheels": weight_on_wheels,
        }
        return F, M

    @staticmethod
    def _euler_from_R(R):
        pitch = math.asin(_clamp(R[2, 0], -1.0, 1.0))
        roll = math.asin(_clamp(R[2, 1], -1.0, 1.0))
        yaw = math.atan2(R[1, 0], R[0, 0])
        return pitch, roll, yaw

    def _substep(self, h: float, ctrl: ControlState):
        ac = self.ac
        F, M = self._forces_moments(ctrl)

        # Angular dynamics: I*w_dot = M - w x (I w)
        I = np.array([ac.Ixx, ac.Iyy, ac.Izz])
        Iw = I * self.omega
        w_dot = (M - np.cross(self.omega, Iw)) / I
        self.omega = self.omega + w_dot * h

        # Translational dynamics in the rotating body frame:
        # v_dot_body = F/m - w x v
        v_dot = F / ac.mass - np.cross(self.omega, self.vel_b)
        self.vel_b = self.vel_b + v_dot * h

        # Attitude: q_dot = 0.5 * q otimes [0, w]; renormalize every substep.
        qd = 0.5 * quat_multiply(self.quat,
                                 np.array([0.0, *self.omega]))
        self.quat = quat_normalize(self.quat + qd * h)

        # Position in the world frame.
        R = quat_to_matrix(self.quat)
        v_w = R @ self.vel_b
        self.pos = self.pos + v_w * h

        # Ground contact projection / liftoff / touchdown.
        if self.on_ground:
            Fw_up = (R @ (F - R.T @ np.array([0.0, 0.0, -ac.mass * G])))[2]
            # (aero + thrust vertical component; weight excluded)
            if Fw_up > ac.mass * G and v_w[2] >= -0.01:
                self.on_ground = False  # liftoff
            else:
                self.pos[2] = GEAR_HEIGHT
                # Remove any downward world velocity (no bouncing model).
                v_w = R @ self.vel_b
                if v_w[2] < 0:
                    self.vel_b = self.vel_b - R.T @ np.array([0.0, 0.0, v_w[2]])
        else:
            if self.pos[2] <= GEAR_HEIGHT:
                v_w = R @ self.vel_b
                if v_w[2] < 0:
                    self.pos[2] = GEAR_HEIGHT
                    self.on_ground = True

        # Hard safety net: nothing leaves this function non-finite.
        for arr in (self.pos, self.vel_b, self.quat, self.omega):
            if not np.all(np.isfinite(arr)):
                raise FloatingPointError("non-finite state in _substep")