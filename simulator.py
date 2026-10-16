"""Panda3D visualization for the flight simulator.

Importing this module does NOT start the application or open a window: it only
defines the FlightSim class and helpers. The app starts when main() is called
(see main.py), which keeps `python main.py --test` headless.

Coordinate conversion: the physics uses x-forward / y-right / z-up while
Panda3D uses x-right / y-forward / z-up. sim_to_panda() maps positions, and
the plane's orientation is set with setHpr() from heading/pitch/roll angles
extracted in Panda's convention (derived in comments below).
"""

import math
import random
import sys

import numpy as np

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (
    AmbientLight,
    CardMaker,
    ClockObject,
    DirectionalLight,
    Fog,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    PNMImage,
    TextNode,
    Texture,
    Vec4,
    loadPrcFileData,
)

from aircraft import Cessna172
from atmosphere import Atmosphere
from controls import ControlState
from flight_model import FlightModel, quat_to_matrix

DEG = 180.0 / math.pi
MS_TO_KTS = 1.94384
M_TO_FT = 3.28084
MS_TO_FPM = 196.85


# ---------------------------------------------------------------------------
# Procedural geometry helpers (no external assets)
# ---------------------------------------------------------------------------

def make_box_node(name, sx, sy, sz, color):
    """Build a colored box GeomNode centered on the origin.

    sx/sy/sz are full extents along x/y/z in the CURRENT frame (the caller is
    responsible for frame conventions). color is an (r, g, b) tuple.
    """
    fmt = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vwriter = GeomVertexWriter(vdata, "vertex")
    cwriter = GeomVertexWriter(vdata, "color")
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    verts = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
             (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    for v in verts:
        vwriter.addData3(*v)
        cwriter.addData4(color[0], color[1], color[2], 1.0)
    tris = GeomTriangles(Geom.UHStatic)
    for a, b, c, d in [(0, 1, 2, 3), (4, 6, 5, 7), (0, 4, 5, 1),
                       (2, 6, 7, 3), (0, 3, 7, 4), (1, 5, 6, 2)]:
        tris.addVertices(a, b, c)
        tris.addVertices(a, c, d)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


def sim_to_panda(pos_sim):
    """Map a sim-frame (x fwd, y right, z up) position to Panda3D coords."""
    x, y, z = pos_sim
    return (y, x, z)  # panda: x right, y fwd, z up


def sim_quat_to_hpr(quat):
    """Extract (heading, pitch, roll) degrees in Panda3D's convention.

    Panda composes setHpr(h, p, r) as M = Rz(h) Rx(p) Ry(r) (row vectors).
    With R_sim the body->world matrix, R_panda = P R_sim P where
    P maps sim (x fwd, y right, z up) to panda (x right, y fwd, z up).
    Decomposing M = R_panda^T gives:
        pitch = asin(M[2,1])            (nose-up positive)
        heading = atan2(-M[0,1], M[1,1])
        roll = atan2(-M[2,0], M[2,2])   (left-roll positive, matches display)
    Verified: pure sim pitch/yaw/roll each extract back correctly.
    """
    R = quat_to_matrix(quat)
    # R_panda = P @ R @ P with P = [[0,1,0],[1,0,0],[0,0,1]] (symmetric)
    Rp = np.array([
        [R[1, 1], R[1, 0], R[1, 2]],
        [R[0, 1], R[0, 0], R[0, 2]],
        [R[2, 1], R[2, 0], R[2, 2]],
    ])
    M = Rp.T
    pitch = math.asin(max(-1.0, min(1.0, M[2, 1])))
    heading = math.atan2(-M[0, 1], M[1, 1])
    roll = math.atan2(-M[2, 0], M[2, 2])
    return heading * DEG, pitch * DEG, roll * DEG


# ---------------------------------------------------------------------------
# The application
# ---------------------------------------------------------------------------

HELP_TEXT = """\
CONTROLS
  Up / W ............ nose down          Down / S .......... nose up
  Left / A .......... roll left         Right / D ......... roll right
  Q / E ............. rudder left/right
  Shift ............. throttle up       Ctrl .............. throttle down
  F ................. flaps (cycles 0/10/20/30 deg)
  T / G ............. elevator trim nose-up / nose-down
  B ................. toggle wheel brakes
  C ................. camera (chase / cockpit / tower)
  P ................. pause             R ................. reset flight
  H ................. toggle this help  Esc ............... quit

Takeoff: full throttle (hold Shift), flaps 10 (F), rotate at 55 kts (pull S).
"""


class FlightSim(ShowBase):
    """Panda3D flight simulator application."""

    def __init__(self):
        super().__init__()
        self.disableMouse()

        self.ac = Cessna172()
        self.fm = FlightModel(self.ac)
        self.ctrl = ControlState()
        self.paused = False
        self.cam_mode = 0  # 0 chase, 1 cockpit, 2 tower
        self.prop_angle = 0.0
        self.keys = {}

        self._build_world()
        self._build_airplane()
        self._build_hud()
        self._bind_keys()

        self.taskMgr.add(self._update, "sim-update")

    # -- world ------------------------------------------------------------
    def _make_ground_texture(self, size=1024, world=4000.0):
        """Procedural terrain: grass with variation + a runway along x."""
        rnd = random.Random(42)  # deterministic
        img = PNMImage(size, size)
        for j in range(size):
            for i in range(size):
                # Large soft patches via sines + fine per-pixel noise.
                patch = (math.sin(i * 0.021) * math.sin(j * 0.017)
                         + 0.5 * math.sin(i * 0.053 + 1.7) * math.sin(j * 0.047 + 0.6))
                n = rnd.uniform(-0.035, 0.035)
                g = 0.42 + 0.10 * patch + n
                img.setXel(i, j, 0.30 + 0.6 * n, g, 0.24 + 0.5 * n)
        # Runway: world x in [-600, 600], y in [-10, 10], drawn in asphalt.
        def to_px(x):
            return int((x + world / 2) / world * (size - 1))
        rx0, rx1 = to_px(-600), to_px(600)
        ry0, ry1 = to_px(-10), to_px(10)
        for j in range(ry0, ry1 + 1):
            for i in range(rx0, rx1 + 1):
                img.setXel(i, j, 0.26, 0.26, 0.28)
        # Edge lines.
        for j in (ry0 + 1, ry1 - 1):
            for i in range(rx0, rx1 + 1):
                img.setXel(i, j, 0.9, 0.9, 0.9)
        # Centerline dashes: 30 m dashes every 60 m.
        yc = to_px(0)
        x = -585.0
        while x < 585.0:
            for k in range(int(30 / world * size)):
                i = to_px(x) + k
                if rx0 <= i <= rx1:
                    for dj in (-1, 0, 1):
                        img.setXel(i, yc + dj, 0.92, 0.92, 0.92)
            x += 60.0
        # Threshold bars at both ends (8 bars each).
        for end_x in (-585.0, 555.0):
            for bar in range(8):
                y = -7.5 + bar * 2.15
                for k in range(int(25 / world * size)):
                    i = to_px(end_x) + k
                    for dj in range(int(1.2 / world * size) + 1):
                        j = to_px(y) + dj
                        if rx0 <= i <= rx1 and ry0 <= j <= ry1:
                            img.setXel(i, j, 0.92, 0.92, 0.92)
        tex = Texture("ground")
        tex.load(img)
        return tex

    def _build_world(self):
        # Ground.
        cm = CardMaker("ground")
        cm.setFrame(-2000, 2000, -2000, 2000)
        ground = self.render.attachNewNode(cm.generate())
        ground.setTexture(self._make_ground_texture())
        ground.setPos(0, 0, -0.05)

        # Gradient sky: big box with a vertical gradient texture, fog off.
        sky_img = PNMImage(4, 256)
        for j in range(256):
            t = j / 255.0  # 0 bottom -> 1 top
            r = 0.78 + (0.20 - 0.78) * t
            g = 0.84 + (0.45 - 0.84) * t
            b = 0.90 + (0.95 - 0.90) * t
            for i in range(4):
                sky_img.setXel(i, j, r, g, b)
        sky_tex = Texture("sky")
        sky_tex.load(sky_img)
        sky = self.render.attachNewNode(make_box_node("sky", 9000, 9000, 9000,
                                                       (1, 1, 1)))
        sky.setTexture(sky_tex)
        sky.setTwoSided(True)
        sky.setFogOff()
        sky.setDepthWrite(False)
        sky.setBin("background", -10)
        sky.setPos(0, 0, 0)

        # Lighting: warm sun + sky ambient.
        sun = DirectionalLight("sun")
        sun.setColor(Vec4(1.0, 0.96, 0.88, 1.0))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(35, -50, 0)
        self.render.setLight(sun_np)
        amb = AmbientLight("amb")
        amb.setColor(Vec4(0.55, 0.60, 0.68, 1.0))
        self.render.setLight(self.render.attachNewNode(amb))

        # Distance haze.
        fog = Fog("haze")
        fog.setColor(0.78, 0.83, 0.89)
        fog.setExpDensity(0.00045)
        self.render.setFog(fog)

    # -- airplane ---------------------------------------------------------
    def _build_airplane(self):
        """Low-poly C172-ish model from boxes (sim frame: x fwd, y right)."""
        white = (0.92, 0.92, 0.94)
        red = (0.75, 0.12, 0.12)
        dark = (0.12, 0.13, 0.16)
        glass = (0.15, 0.22, 0.30)

        self.plane = self.render.attachNewNode("plane")

        # Parts below are dimensioned directly in PANDA axes
        # (x right = span, y forward = length, z up) to avoid a frame flip.
        parts = [
            ("fuselage", 1.15, 8.4, 1.35, white, (0, 0.4, 0.15)),
            ("stripe", 1.17, 8.0, 0.22, red, (0, 0.35, -0.25)),
            ("cabin", 1.0, 2.0, 0.75, glass, (0, 1.1, 0.95)),
            ("wing", 11.0, 1.55, 0.14, white, (0, 0.6, 1.05)),
            ("wingtip_l", 0.5, 0.4, 0.16, red, (-5.4, 0.6, 1.05)),
            ("wingtip_r", 0.5, 0.4, 0.16, red, (5.4, 0.6, 1.05)),
            ("hstab", 3.4, 1.05, 0.10, white, (0, -3.55, 0.35)),
            ("fin", 0.12, 1.05, 1.7, red, (0, -3.55, 1.05)),
            ("gear_l", 0.35, 0.55, 0.55, dark, (-0.85, 1.6, -0.75)),
            ("gear_r", 0.35, 0.55, 0.55, dark, (0.85, 1.6, -0.75)),
            ("gear_n", 0.30, 0.45, 0.50, dark, (0, 3.1, -0.72)),
            ("strut_l", 0.12, 0.12, 0.8, dark, (-0.85, 1.6, -0.25)),
            ("strut_r", 0.12, 0.12, 0.8, dark, (0.85, 1.6, -0.25)),
            ("strut_n", 0.10, 0.10, 0.7, dark, (0, 3.1, -0.25)),
        ]
        # NOTE: extents are (x right, y forward, z up) in Panda's frame:
        # fuselage 8.4 m long, wing 11 m span, matching the C172 class.
        for name, ex, ey, ez, color, pos in parts:
            node = NodePath(make_box_node(name, ex, ey, ez, color))
            node.setPos(*pos)
            node.setTwoSided(True)
            node.reparentTo(self.plane)

        # Propeller: blades in the x-z plane at the nose, spinning about y.
        self.prop = self.plane.attachNewNode("prop")
        blade = NodePath(make_box_node("blade", 0.10, 0.06, 2.0, dark))
        blade.setTwoSided(True)
        blade.reparentTo(self.prop)
        blade2 = NodePath(make_box_node("blade2", 2.0, 0.06, 0.10, dark))
        blade2.setTwoSided(True)
        blade2.reparentTo(self.prop)
        spinner = NodePath(make_box_node("spinner", 0.30, 0.45, 0.30, red))
        spinner.reparentTo(self.prop)
        self.prop.setPos(0, 4.65, 0.15)

    # -- HUD --------------------------------------------------------------
    def _text(self, pos, scale=0.05, align=TextNode.ALeft):
        return OnscreenText(text="", pos=pos, scale=scale, align=align,
                            fg=(1, 1, 1, 1), bg=(0, 0, 0, 0.45),
                            mayChange=True)

    def _build_hud(self):
        self.hud_left = self._text((-1.72, 0.88), 0.055)
        self.hud_right = self._text((1.72, 0.88), 0.055, TextNode.ARight)
        self.hud_warn = OnscreenText(text="", pos=(0, 0.80), scale=0.075,
                                    align=TextNode.ACenter, fg=(1, 0.15, 0.15, 1),
                                    mayChange=True)
        self.hud_ground = OnscreenText(text="", pos=(0, 0.68), scale=0.05,
                                      align=TextNode.ACenter, fg=(0.4, 1, 0.4, 1),
                                      mayChange=True)
        self._build_attitude_indicator()
        self.help_text = OnscreenText(text=HELP_TEXT, pos=(-1.70, 0.55),
                                     scale=0.042, align=TextNode.ALeft,
                                     fg=(1, 1, 1, 1), bg=(0, 0, 0, 0.55),
                                     mayChange=False)

    def _build_attitude_indicator(self):
        """Simple AI: blue/brown ball that pitches and rolls under a fixed symbol."""
        root = NodePath("ai")
        root.reparentTo(self.aspect2d)
        root.setPos(0, 0, -0.70)

        frame = CardMaker("ai_frame")
        frame.setFrame(-0.19, 0.19, -0.19, 0.19)
        fn = root.attachNewNode(frame.generate())
        fn.setColor(0.05, 0.05, 0.08, 0.85)
        fn.setBin("fixed", 10)
        fn.setDepthTest(False)

        self.ai_ball = root.attachNewNode("ai_ball")
        sky_cm = CardMaker("ai_sky")
        sky_cm.setFrame(-0.17, 0.17, 0.0, 0.34)
        sky_n = self.ai_ball.attachNewNode(sky_cm.generate())
        sky_n.setColor(0.20, 0.45, 0.85, 1.0)
        sky_n.setBin("fixed", 11)
        sky_n.setDepthTest(False)
        gnd_cm = CardMaker("ai_gnd")
        gnd_cm.setFrame(-0.17, 0.17, -0.34, 0.0)
        gnd_n = self.ai_ball.attachNewNode(gnd_cm.generate())
        gnd_n.setColor(0.55, 0.38, 0.22, 1.0)
        gnd_n.setBin("fixed", 11)
        gnd_n.setDepthTest(False)

        # Fixed aircraft symbol (stays put while the ball moves).
        sym = root.attachNewNode("ai_sym")
        for x0, x1 in [(-0.13, -0.03), (0.03, 0.13)]:
            wcm = CardMaker("ai_wing")
            wcm.setFrame(x0, x1, -0.012, 0.012)
            wn = sym.attachNewNode(wcm.generate())
            wn.setColor(1, 1, 1, 1)
            wn.setBin("fixed", 12)
            wn.setDepthTest(False)
        dot_cm = CardMaker("ai_dot")
        dot_cm.setFrame(-0.012, 0.012, -0.012, 0.012)
        dn = sym.attachNewNode(dot_cm.generate())
        dn.setColor(1, 1, 1, 1)
        dn.setBin("fixed", 12)
        dn.setDepthTest(False)

    # -- input ------------------------------------------------------------
    def _bind_keys(self):
        def bind(key, name):
            self.accept(key, self._set_key, [name, True])
            self.accept(key + "-up", self._set_key, [name, False])

        for key, name in [("arrow_up", "up"), ("arrow_down", "down"),
                          ("arrow_left", "left"), ("arrow_right", "right"),
                          ("w", "up"), ("s", "down"), ("a", "left"), ("d", "right"),
                          ("q", "rud_l"), ("e", "rud_r"),
                          ("shift", "thr_up"), ("control", "thr_dn")]:
            bind(key, name)

        self.accept("f", self._cycle_flaps)
        self.accept("t", self._trim_up)
        self.accept("g", self._trim_down)
        self.accept("b", self._toggle_brakes)
        self.accept("c", self._cycle_camera)
        self.accept("p", self._toggle_pause)
        self.accept("r", self._reset)
        self.accept("h", self._toggle_help)
        self.accept("escape", base.userExit)

    def _set_key(self, name, held):
        self.keys[name] = held

    def _held(self, name):
        return self.keys.get(name, False)

    def _cycle_flaps(self):
        self.ctrl.cycle_flaps()

    def _trim_up(self):
        self.ctrl.trim_up()

    def _trim_down(self):
        self.ctrl.trim_down()

    def _toggle_brakes(self):
        self.ctrl.brakes = not self.ctrl.brakes

    def _cycle_camera(self):
        self.cam_mode = (self.cam_mode + 1) % 3
        self.plane.show()  # cockpit mode hides it again below if needed

    def _toggle_pause(self):
        self.paused = not self.paused

    def _toggle_help(self):
        if self.help_text.isHidden():
            self.help_text.show()
        else:
            self.help_text.hide()

    def _reset(self):
        self.fm.reset()
        self.ctrl.reset()

    # -- per-frame update -------------------------------------------------
    def _update_controls_from_keys(self, dt):
        c = self.ctrl
        # Elevator: Up/W = push (nose down), Down/S = pull (nose up).
        elev_target = (1.0 if self._held("down") else 0.0) + \
                      (-1.0 if self._held("up") else 0.0)
        ail_target = (1.0 if self._held("right") else 0.0) + \
                     (-1.0 if self._held("left") else 0.0)
        rud_target = (1.0 if self._held("rud_r") else 0.0) + \
                     (-1.0 if self._held("rud_l") else 0.0)
        rate = 2.2
        for attr, target in (("elevator", elev_target), ("aileron", ail_target),
                             ("rudder", rud_target)):
            cur = getattr(c, attr)
            if target == 0.0:  # spring-loaded back to neutral
                cur -= math.copysign(min(abs(cur), 3.0 * dt), cur) if cur != 0 else 0.0
            else:
                step = math.copysign(min(abs(target - cur), rate * dt), target - cur)
                cur += step
            setattr(c, attr, max(-1.0, min(1.0, cur)))
        if self._held("thr_up"):
            c.throttle = min(1.0, c.throttle + 0.45 * dt)
        if self._held("thr_dn"):
            c.throttle = max(0.0, c.throttle - 0.45 * dt)

    def _update(self, task):
        dt = min(ClockObject.getGlobalClock().getDt(), 0.05)
        if not self.paused:
            self._update_controls_from_keys(dt)
            self.fm.step(dt, self.ctrl)
            self.prop_angle = (self.prop_angle
                               + (4.0 + 38.0 * self.ctrl.throttle) * dt)
            self.prop.setR(math.degrees(self.prop_angle) % 360.0)
        self._update_plane_node()
        self._update_camera(dt)
        self._update_hud()
        return task.cont

    def _update_plane_node(self):
        px, py, pz = sim_to_panda(self.fm.pos)
        self.plane.setPos(px, py, pz)
        h, p, r = sim_quat_to_hpr(self.fm.quat)
        self.plane.setHpr(h, p, r)

    def _update_camera(self, dt):
        if self.camera is None:
            return  # headless mode (window-type none): no camera exists
        px, py, pz = sim_to_panda(self.fm.pos)
        h, p, r = sim_quat_to_hpr(self.fm.quat)
        plane_pos = np.array([px, py, pz])
        if self.cam_mode == 0:  # chase
            self.plane.show()
            # 16 m behind, 5 m above, in the plane's frame.
            hr, pr = math.radians(h), math.radians(p)
            # Forward vector in panda frame from heading/pitch.
            fx, fy, fz = (math.sin(hr) * math.cos(pr),
                          math.cos(hr) * math.cos(pr), math.sin(pr))
            back = np.array([fx, fy, fz])
            target = plane_pos - back * 16.0 + np.array([0, 0, 5.0])
            cur = np.array(self.camera.getPos())
            smooth = cur + (target - cur) * min(1.0, dt * 4.0)
            self.camera.setPos(*smooth)
            self.camera.lookAt(px, py, pz)
        elif self.cam_mode == 1:  # cockpit
            self.plane.hide()
            self.camera.setPos(px, py + 0.9, pz + 1.55)
            self.camera.setHpr(h, p, r)
        else:  # tower flyby
            self.plane.show()
            self.camera.setPos(120, 300, 25)
            self.camera.lookAt(px, py, pz)

    def _update_hud(self):
        fm, c = self.fm, self.ctrl
        V = fm.airspeed()
        kts = V * MS_TO_KTS
        alt_ft = fm.pos[2] * M_TO_FT
        pitch, roll, yaw = fm.attitude_euler()
        hdg = (yaw * DEG) % 360.0
        # Vertical speed from world velocity.
        R = quat_to_matrix(fm.quat)
        vs_fpm = float((R @ fm.vel_b)[2]) * MS_TO_FPM
        info = fm.info()

        self.hud_left.setText(
            f"SPD {kts:5.0f} kt\nALT {alt_ft:6.0f} ft\n"
            f"HDG {hdg:03.0f}\nVS {vs_fpm:+5.0f} fpm")
        self.hud_right.setText(
            f"THR {c.throttle * 100:3.0f}%\nFLAP {c.flap_degrees():2d} deg\n"
            f"TRIM {c.trim:+.2f}\nBRK {'ON' if c.brakes else 'off'}")
        if info.get("stalled"):
            self.hud_warn.setText("*** STALL ***" if (fm.time % 1.0) < 0.6 else "")
        else:
            self.hud_warn.setText("")
        self.hud_ground.setText("ON GROUND" if fm.on_ground else "")

        # Attitude indicator: the ball carries the horizon; when the aircraft
        # rolls right (roll is left-positive, so roll < 0), the horizon appears
        # to rotate counter-clockwise -> ball rotation = -roll. Nose-up pushes
        # the horizon down -> ball shifts down.
        self.ai_ball.setR(-roll * DEG)
        self.ai_ball.setZ(-pitch * 0.55)

        if self.paused:
            self.hud_ground.setText("PAUSED — press P to resume")

def main():
    """Entry point for the graphical simulator."""
    loadPrcFileData("", "window-title Python Flight Simulator")
    loadPrcFileData("", "win-size 1280 720")
    app = FlightSim()
    app.run()


if __name__ == "__main__":
    main()
