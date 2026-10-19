"""Panda3D visualization for the flight simulator.

Importing this module does NOT start the application or open a window: it only
defines the FlightSim class and helpers. The app starts when main() is called
(see main.py), which keeps `python main.py --test` headless.

Coordinate conversion: the physics uses x-forward / y-right / z-up while
Panda3D uses x-right / y-forward / z-up. sim_to_panda() maps positions, and
the plane's orientation is set with setHpr() from heading/pitch/roll angles
extracted in Panda's convention (derived in comments below).

All scenery is procedural (no downloaded assets): textured terrain with field
patchwork, a marked runway with designation numbers, gradient sky dome with
sun glow, drifting clouds, trees, hangars, windsock, runway edge lights, and
a refined Cessna model with spinning prop blur, wing struts, gear fairings,
and navigation lights. Real-time shadows follow the airplane.
"""

import math
import random
import sys

import numpy as np

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
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
    TextureStage,
    TransparencyAttrib,
    Vec4,
    loadPrcFileData,
)

from aircraft import Cessna172
from atmosphere import Atmosphere
from controls import ControlState
from flight_model import FlightModel, quat_to_matrix, GEAR_HEIGHT

DEG = 180.0 / math.pi
MS_TO_KTS = 1.94384
M_TO_FT = 3.28084
MS_TO_FPM = 196.85

# Sun travel direction in Panda coords, derived from the light's Hpr(35,-50,0)
# (see _build_world). SUN_DIR points from the scene toward the sun.
_SUN_TRAVEL = np.array([math.sin(math.radians(35)) * math.cos(math.radians(-50)),
                        math.cos(math.radians(35)) * math.cos(math.radians(-50)),
                        math.sin(math.radians(-50))])
SUN_DIR = -_SUN_TRAVEL / np.linalg.norm(_SUN_TRAVEL)


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


def make_cone_node(name, r_base, height, color, segments=10):
    """Cone with its base ring at z=0 and apex at z=height."""
    fmt = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")
    cw = GeomVertexWriter(vdata, "color")
    vw.addData3(0, 0, 0)
    cw.addData4(*color, 1.0)
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        vw.addData3(r_base * math.cos(a), r_base * math.sin(a), 0)
        cw.addData4(*color, 1.0)
    vw.addData3(0, 0, height)
    cw.addData4(*color, 1.0)
    apex = segments + 1
    tris = GeomTriangles(Geom.UHStatic)
    for i in range(segments):
        a, b = 1 + i, 1 + (i + 1) % segments
        tris.addVertices(0, b, a)      # base fan
        tris.addVertices(a, b, apex)   # side
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


def make_disc_node(name, radius, segments=40):
    """Flat UV-mapped disc in the X-Z plane (normal +/-Y), for prop blur."""
    fmt = GeomVertexFormat.getV3t2()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")
    uw = GeomVertexWriter(vdata, "texcoord")
    vw.addData3(0, 0, 0)
    uw.addData2(0.5, 0.5)
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        x, z = radius * math.cos(a), radius * math.sin(a)
        vw.addData3(x, 0, z)
        uw.addData2(0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.sin(a))
    tris = GeomTriangles(Geom.UHStatic)
    for i in range(segments):
        tris.addVertices(0, 1 + i, 1 + (i + 1) % segments)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


def make_sphere_node(name, radius, sectors=28, rings=14):
    """UV-mapped sphere (equirect-ish: v=1 at the top), viewed from inside."""
    fmt = GeomVertexFormat.getV3t2()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")
    uw = GeomVertexWriter(vdata, "texcoord")
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(sectors + 1):
            theta = 2.0 * math.pi * s / sectors
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            vw.addData3(x, y, z)
            uw.addData2(s / sectors, 1.0 - r / rings)
    tris = GeomTriangles(Geom.UHStatic)
    for r in range(rings):
        for s in range(sectors):
            a = r * (sectors + 1) + s
            b = a + 1
            c = a + sectors + 1
            d = c + 1
            tris.addVertices(a, c, b)
            tris.addVertices(b, c, d)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


def make_cylinder_node(name, radius, height, color, segments=14):
    """Upright cylinder: side wall + top cap, base ring at z=0."""
    fmt = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")
    cw = GeomVertexWriter(vdata, "color")
    for zz in (0.0, height):
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            vw.addData3(radius * math.cos(a), radius * math.sin(a), zz)
            cw.addData4(*color, 1.0)
    top_center = 2 * segments
    vw.addData3(0, 0, height)
    cw.addData4(*color, 1.0)
    tris = GeomTriangles(Geom.UHStatic)
    for i in range(segments):
        a, b = i, (i + 1) % segments
        c, d = segments + i, segments + (i + 1) % segments
        tris.addVertices(a, c, b)   # wall
        tris.addVertices(b, c, d)
        tris.addVertices(top_center, c, d)  # cap
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
  C ................. camera (chase / cockpit / tower / top-down)
  P ................. pause             R ................. reset flight
  H ................. toggle this help  Esc ............... quit

Takeoff: full throttle (hold Shift), flaps 10 (F), rotate at 55 kts (pull S).
"""


class FlightSim(ShowBase):
    """Panda3D flight simulator application."""

    def __init__(self):
        super().__init__()
        self.disableMouse()
        self.render.setAntialias(AntialiasAttrib.MMultisample)
        self.render.setShaderAuto()  # per-pixel lighting + shadows

        self.ac = Cessna172()
        self.fm = FlightModel(self.ac)
        # Start at the south threshold: the full runway and the base
        # complex lie ahead, so the takeoff roll shows them off.
        self.fm.pos = np.array([-480.0, 0.0, GEAR_HEIGHT])
        self.ctrl = ControlState()
        self.paused = False
        self.cam_mode = 0  # 0 chase, 1 cockpit, 2 tower, 3 top-down
        self.prop_angle = 0.0
        self.keys = {}
        self.clouds = []

        self._build_world()
        self._build_airplane()
        self._build_hud()
        self._bind_keys()

        self.taskMgr.add(self._update, "sim-update")

    # -- procedural textures ------------------------------------------------
    @staticmethod
    def _tex_from_image(img, name):
        tex = Texture(name)
        tex.load(img)
        return tex

    def _make_ground_texture(self, size=1024, world=4000.0):
        """Terrain: patchwork fields with hedgerows, noise, and two roads."""
        rnd = random.Random(42)  # deterministic
        img = PNMImage(size, size)
        # Base meadow.
        for j in range(size):
            for i in range(size):
                n = rnd.uniform(-0.03, 0.03)
                img.setXel(i, j, 0.30 + 0.5 * n, 0.44 + 0.6 * n, 0.24 + 0.4 * n)
        # Field patchwork: large soft rectangles in varied greens/tans.
        for _ in range(70):
            w = rnd.uniform(120, 420)
            h = rnd.uniform(120, 420)
            x0 = rnd.uniform(-world / 2, world / 2 - w)
            y0 = rnd.uniform(-world / 2, world / 2 - h)
            tone = rnd.random()
            if tone < 0.55:
                col = (rnd.uniform(0.24, 0.34), rnd.uniform(0.42, 0.55),
                       rnd.uniform(0.20, 0.28))
            elif tone < 0.8:
                col = (rnd.uniform(0.45, 0.58), rnd.uniform(0.42, 0.50),
                       rnd.uniform(0.25, 0.32))
            else:
                col = (rnd.uniform(0.35, 0.45), rnd.uniform(0.38, 0.46),
                       rnd.uniform(0.22, 0.28))
            i0 = int((x0 + world / 2) / world * (size - 1))
            i1 = int((x0 + w + world / 2) / world * (size - 1))
            j0 = int((y0 + world / 2) / world * (size - 1))
            j1 = int((y0 + h + world / 2) / world * (size - 1))
            for j in range(max(0, j0), min(size, j1 + 1)):
                for i in range(max(0, i0), min(size, i1 + 1)):
                    edge = (j in (j0, j1) or i in (i0, i1))
                    if edge:
                        img.setXel(i, j, col[0] * 0.55, col[1] * 0.55,
                                   col[2] * 0.55)
                    else:
                        n = rnd.uniform(-0.02, 0.02)
                        img.setXel(i, j, col[0] + n, col[1] + n, col[2] + n)
        # Roads (image coords; orientation is cosmetic here).
        for j in range(size):
            for i in range(int(0.72 * size), int(0.728 * size)):
                img.setXel(i, j, 0.33, 0.33, 0.34)
        for j in range(int(0.30 * size), int(0.308 * size)):
            for i in range(size):
                img.setXel(i, j, 0.33, 0.33, 0.34)
        return self._tex_from_image(img, "ground")

    def _make_runway_texture(self):
        """Crisp concrete runway: edge lines, centerline, thresholds,
        touchdown zones, and designation numbers. Runway runs along +Y."""
        W, H = 128, 2048
        RW_W, RW_L = 30.0, 1240.0  # meters
        rnd = random.Random(7)
        img = PNMImage(W, H)

        def mx(x):
            return int((x + RW_W / 2) / RW_W * (W - 1))

        def my(y):
            # Image top (j=0) maps to v=1 = far/north end (+Y).
            return int((RW_L / 2 - y) / RW_L * (H - 1))

        white = (0.90, 0.90, 0.90)
        for j in range(H):
            for i in range(W):
                n = rnd.uniform(-0.022, 0.022)
                img.setXel(i, j, 0.33 + n, 0.33 + n, 0.34 + n)

        def hline(y0_m, y1_m, x0_m, x1_m, col=white):
            for j in range(my(y1_m), my(y0_m) + 1):
                for i in range(mx(x0_m), mx(x1_m) + 1):
                    if 0 <= i < W and 0 <= j < H:
                        img.setXel(i, j, *col)

        # Edge lines.
        hline(-600, 600, 13.2, 13.9)
        hline(-600, 600, -13.9, -13.2)
        # Centerline dashes: 30 m dash, 30 m gap.
        y = -585.0
        while y < 585.0:
            hline(y, y + 30.0, -0.45, 0.45)
            y += 60.0
        # Threshold piano keys at both ends (8 bars).
        for end in (-1, 1):
            y_base = end * 606.0
            for bar in range(8):
                x = -12.6 + bar * 3.4
                hline(min(y_base, y_base - end * 18.0),
                      max(y_base, y_base - end * 18.0), x, x + 2.2)
        # Touchdown zone pairs.
        for end in (-1, 1):
            for d in (150.0, 300.0, 450.0):
                yc = end * (606.0 - d)
                hline(yc - 6.0, yc + 6.0, 6.5, 8.3)
                hline(yc - 6.0, yc + 6.0, -8.3, -6.5)

        # Designation numbers, 3x5 blocky font.
        font = {"0": ["111", "101", "101", "101", "111"],
                "2": ["111", "001", "111", "100", "111"],
                "7": ["111", "001", "010", "010", "010"],
                "9": ["111", "101", "111", "001", "111"]}

        def draw_number(text, y_center_m, rot180):
            cell = 8
            total_w = len(text) * 4 * cell - cell
            top = my(y_center_m) - (5 * cell) // 2
            cx = W // 2 - total_w // 2
            for di, ch in enumerate(text):
                for r in range(5):
                    for c in range(3):
                        if font[ch][r][c] == "1":
                            rr, cc = (4 - r, 2 - c) if rot180 else (r, c)
                            for dj in range(cell):
                                for di2 in range(cell):
                                    i = cx + di * 4 * cell + cc * cell + di2
                                    j = top + rr * cell + dj
                                    if 0 <= i < W and 0 <= j < H:
                                        img.setXel(i, j, *white)

        # "27" at the south end reads for a northbound takeoff;
        # "09" at the north end reads for a southbound one.
        draw_number("27", -520.0, rot180=False)
        draw_number("09", 520.0, rot180=True)
        return self._tex_from_image(img, "runway")

    def _make_concrete_texture(self, size=256, slab=64):
        """Poured-concrete slabs with seams, for aprons and taxiways."""
        rnd = random.Random(21)
        img = PNMImage(size, size)
        for j in range(size):
            for i in range(size):
                n = rnd.uniform(-0.02, 0.02)
                seam = (i % slab < 2) or (j % slab < 2)
                base = 0.36 if seam else 0.47
                img.setXel(i, j, base + n, base + n, base + 0.01 + n)
        return self._tex_from_image(img, "concrete")

    def _make_taxiway_texture(self):
        """Asphalt strip with a yellow centerline, tiled along its length."""
        W, H = 64, 512
        rnd = random.Random(33)
        img = PNMImage(W, H)
        for j in range(H):
            for i in range(W):
                n = rnd.uniform(-0.02, 0.02)
                img.setXel(i, j, 0.30 + n, 0.30 + n, 0.31 + n)
        # Yellow centerline (continuous, like a real taxiway).
        for j in range(H):
            for i in range(W // 2 - 2, W // 2 + 2):
                img.setXel(i, j, 0.75, 0.62, 0.15)
        tex = self._tex_from_image(img, "taxiway")
        tex.setWrapU(Texture.WMRepeat)
        tex.setWrapV(Texture.WMRepeat)
        return tex

    def _make_sky_texture(self):
        """Vertical gradient: deep zenith blue -> warm horizon haze."""
        img = PNMImage(8, 256)
        for j in range(256):
            t = j / 255.0  # 0 = image top = zenith
            if t < 0.5:
                k = t / 0.5
                r = 0.10 + (0.83 - 0.10) * k
                g = 0.30 + (0.87 - 0.30) * k
                b = 0.70 + (0.91 - 0.70) * k
            else:
                k = (t - 0.5) / 0.5
                r = 0.83 + (0.32 - 0.83) * k
                g = 0.87 + (0.36 - 0.87) * k
                b = 0.91 + (0.40 - 0.91) * k
            for i in range(8):
                img.setXel(i, j, r, g, b)
        return self._tex_from_image(img, "sky")

    def _make_glow_texture(self, size=96):
        """Soft radial glow (white core -> transparent), for sun and lights."""
        img = PNMImage(size, size, 4)
        for j in range(size):
            for i in range(size):
                dx = (i / (size - 1)) * 2.0 - 1.0
                dy = (j / (size - 1)) * 2.0 - 1.0
                d = math.sqrt(dx * dx + dy * dy)
                a = max(0.0, 1.0 - d) ** 2.2
                img.setXel(i, j, 1.0, 1.0, 1.0)
                img.setAlpha(i, j, a)
        return self._tex_from_image(img, "glow")

    def _make_cloud_texture(self, size=128):
        """Puffy cloud puff from composited radial blobs."""
        rnd = random.Random(1234)
        img = PNMImage(size, size, 4)
        blobs = [(rnd.uniform(0.25, 0.75), rnd.uniform(0.35, 0.65),
                  rnd.uniform(0.15, 0.30)) for _ in range(9)]
        for j in range(size):
            for i in range(size):
                x, y = i / size, j / size
                a = 0.0
                for bx, by, br in blobs:
                    d2 = ((x - bx) ** 2 + (y - by) ** 2) / (br * br)
                    if d2 < 4.0:
                        a += math.exp(-d2 * 1.6) * 0.75
                a = min(1.0, a)
                # Slightly darker base for depth.
                shade = 0.92 + 0.08 * (1.0 - min(1.0, a))
                img.setXel(i, j, shade, shade, 1.0)
                img.setAlpha(i, j, min(1.0, a * 1.15))
        return self._tex_from_image(img, "cloud")

    def _make_prop_texture(self, size=128):
        """Translucent prop-blur disc: faint hub, stronger rim."""
        img = PNMImage(size, size, 4)
        for j in range(size):
            for i in range(size):
                dx = (i / (size - 1)) * 2.0 - 1.0
                dy = (j / (size - 1)) * 2.0 - 1.0
                d = math.sqrt(dx * dx + dy * dy)
                if d > 1.0:
                    a = 0.0
                else:
                    a = 0.10 + 0.30 * (d ** 2)
                img.setXel(i, j, 0.82, 0.82, 0.86)
                img.setAlpha(i, j, a)
        return self._tex_from_image(img, "propblur")

    # -- world ------------------------------------------------------------
    def _build_world(self):
        # Ground with field patchwork.
        cm = CardMaker("ground")
        cm.setFrame(-2000, 2000, -2000, 2000)
        ground = self.render.attachNewNode(cm.generate())
        ground.setTexture(self._make_ground_texture())
        ground.setPos(0, 0, -0.05)
        # Solid fallback: if the texture ever fails to show, grass still does.
        gbase = NodePath(make_box_node("ground_base", 4000, 4000, 0.1,
                                       (0.32, 0.46, 0.26)))
        gbase.setPos(0, 0, -0.12)
        gbase.reparentTo(self.render)

        # Runway as its own card (crisp markings), along the takeoff axis.
        rcm = CardMaker("runway")
        rcm.setFrame(-15, 15, -620, 620)
        runway = self.render.attachNewNode(rcm.generate())
        runway.setTexture(self._make_runway_texture())
        runway.setPos(0, 0, 0.06)
        rwbase = NodePath(make_box_node("runway_base", 30.5, 1240.5, 0.08,
                                        (0.36, 0.36, 0.37)))
        rwbase.setPos(0, 0, 0.0)
        rwbase.reparentTo(self.render)

        # Sky dome: gradient sphere, unlit, drawn behind everything.
        sky = self.render.attachNewNode(make_sphere_node("sky", 8000.0))
        sky.setTexture(self._make_sky_texture())
        sky.setTwoSided(True)
        sky.setLightOff()
        sky.setFogOff()
        sky.setDepthWrite(False)
        sky.setBin("background", -10)

        # Sun glow sprite, repositioned every frame relative to the camera.
        glow_tex = self._make_glow_texture()
        self.glow_tex = glow_tex
        scm = CardMaker("sun")
        scm.setFrame(-650, 650, -650, 650)
        self.sun_card = self.render.attachNewNode(scm.generate())
        self.sun_card.setTexture(glow_tex)
        self.sun_card.setColor(1.0, 0.93, 0.80, 1.0)
        self.sun_card.setBillboardPointEye()
        self.sun_card.setTransparency(TransparencyAttrib.MAlpha)
        self.sun_card.setLightOff()
        self.sun_card.setFogOff()
        self.sun_card.setDepthWrite(False)
        self.sun_card.setBin("background", -5)

        # Drifting clouds.
        cloud_tex = self._make_cloud_texture()
        rnd = random.Random(99)
        for _ in range(14):
            ccm = CardMaker("cloud")
            w = rnd.uniform(380, 750)
            ccm.setFrame(-w / 2, w / 2, -w / 3.2, w / 3.2)
            c = self.render.attachNewNode(ccm.generate())
            c.setTexture(cloud_tex)
            c.setTransparency(TransparencyAttrib.MAlpha)
            c.setLightOff()
            c.setBillboardPointEye()
            ang = rnd.uniform(0, 2 * math.pi)
            rad = rnd.uniform(600, 2700)
            c.setPos(rad * math.cos(ang), rad * math.sin(ang),
                     rnd.uniform(380, 680))
            c.setDepthWrite(False)
            self.clouds.append((c, rnd.uniform(2.0, 5.0)))

        # Lighting: warm sun + cool sky fill.
        # NOTE: real-time shadows are OFF. They caused a driver-level bug on
        # some machines where the shadow pass leaked into the main view as
        # a duplicated top-down copy of the scene across the top of the
        # screen. Re-enable only on machines verified to handle it.
        sun = DirectionalLight("sun")
        sun.setColor(Vec4(1.0, 0.94, 0.82, 1.0))
        self.sun_np = self.render.attachNewNode(sun)
        self.sun_np.setHpr(35, -50, 0)
        self.render.setLight(self.sun_np)
        amb = AmbientLight("amb")
        amb.setColor(Vec4(0.42, 0.47, 0.58, 1.0))
        self.render.setLight(self.render.attachNewNode(amb))

        # Distance haze.
        fog = Fog("haze")
        fog.setColor(0.80, 0.84, 0.89)
        fog.setExpDensity(0.00040)
        self.render.setFog(fog)

        self._build_airbase(rnd)

    def _build_airbase(self, rnd):
        """Military airbase: parallel taxiway, apron, control tower,
        hangars, hardened shelters, rotating radar, parked jets, and
        support vehicles."""
        concrete = self._make_concrete_texture()
        concrete.setWrapU(Texture.WMRepeat)
        concrete.setWrapV(Texture.WMRepeat)

        def pave(name, x0, x1, y0, y1, z=0.04, rep_u=1, rep_v=1, tex=None):
            cm = CardMaker(name)
            cm.setFrame(x0, x1, y0, y1)
            n = self.render.attachNewNode(cm.generate())
            n.setTexture(concrete if tex is None else tex)
            n.setTexScale(TextureStage.getDefault(), rep_u, rep_v)
            n.setPos(0, 0, z)
            # Solid fallback under the textured card.
            base = NodePath(make_box_node(name + "_base", x1 - x0 + 0.5,
                                          y1 - y0 + 0.5, 0.06,
                                          (0.42, 0.42, 0.43)))
            base.setPos((x0 + x1) / 2.0, (y0 + y1) / 2.0, z - 0.05)
            base.reparentTo(self.render)
            return n

        # Parallel taxiway east of the runway + three connectors.
        pave("taxiway", 82, 98, -620, 620, tex=self._make_taxiway_texture(),
             rep_v=12)
        for yc in (-400, 0, 400):
            pave("connector_%d" % yc, 14, 84, yc - 7, yc + 7)
        # Main apron.
        pave("apron", 110, 290, -180, 180, rep_u=3, rep_v=6)

        gray = (0.60, 0.61, 0.64)
        dark = (0.36, 0.37, 0.40)

        # Control tower.
        tower_parts = [
            ("tshaft", 9, 9, 30, gray, (200, -240, 15)),
            ("tcab", 14, 14, 5.5, (0.13, 0.19, 0.27), (200, -240, 32.5)),
            ("troof", 15.5, 15.5, 1.0, dark, (200, -240, 35.8)),
            ("tmast", 0.5, 0.5, 9, dark, (200, -240, 40.5)),
        ]
        for name, ex, ey, ez, color, pos in tower_parts:
            node = NodePath(make_box_node(name, ex, ey, ez, color))
            node.setPos(*pos)
            node.reparentTo(self.render)

        # Hangar row.
        for hy in (-100, -20, 60):
            walls = NodePath(make_box_node("hangar", 46, 32, 12, gray))
            walls.setPos(205, hy, 6)
            walls.reparentTo(self.render)
            roof = NodePath(make_box_node("hangar_roof", 49, 35, 1.4, dark))
            roof.setPos(205, hy, 12.7)
            roof.reparentTo(self.render)
            door = NodePath(make_box_node("hangar_door", 0.5, 30, 9,
                                          (0.15, 0.16, 0.19)))
            door.setPos(205 - 23.2, hy, 4.5)
            door.reparentTo(self.render)

        # Hardened aircraft shelters (earth tone).
        for hx, hy in ((262, -140), (262, 140)):
            sh = NodePath(make_box_node("has", 26, 20, 9, (0.46, 0.43, 0.35)))
            sh.setPos(hx, hy, 4.5)
            sh.reparentTo(self.render)
            shr = NodePath(make_box_node("has_roof", 28, 22, 1.2,
                                         (0.38, 0.35, 0.28)))
            shr.setPos(hx, hy, 9.6)
            shr.reparentTo(self.render)

        # Radar: pedestal + rotating dome.
        ped = NodePath(make_box_node("radar_ped", 7, 7, 10, gray))
        ped.setPos(330, 260, 5)
        ped.reparentTo(self.render)
        self.radar_dome = NodePath(make_sphere_node("radome", 4.6))
        self.radar_dome.setColor(0.90, 0.90, 0.92)
        self.radar_dome.setPos(330, 260, 14.6)
        self.radar_dome.reparentTo(self.render)

        # Parked jets on the apron.
        for jx, jy in ((150, -120), (150, -40), (150, 40)):
            self._build_static_jet(jx, jy, 90.0)

        # Fuel truck.
        truck = self.render.attachNewNode("truck")
        tparts = [
            ("cab", 2.6, 2.8, 2.6, (0.36, 0.37, 0.29), (0, 3.6, 1.7)),
            ("tank", 2.8, 6.8, 2.8, (0.72, 0.72, 0.68), (0, -1.2, 1.9)),
            ("chassis", 2.4, 10.5, 0.7, dark, (0, 0, 0.75)),
        ]
        for name, ex, ey, ez, color, pos in tparts:
            node = NodePath(make_box_node(name, ex, ey, ez, color))
            node.setPos(*pos)
            node.setTwoSided(True)
            node.reparentTo(truck)
        for wx, wy in ((-1.2, 3.6), (1.2, 3.6), (-1.2, -0.5), (1.2, -0.5),
                       (-1.2, -3.2), (1.2, -3.2)):
            wheel = NodePath(make_box_node("wheel", 0.5, 0.9, 0.9,
                                           (0.08, 0.08, 0.09)))
            wheel.setPos(wx, wy, 0.45)
            wheel.reparentTo(truck)
        truck.setPos(235, -160, 0)
        truck.setH(15.0)

        # Apron floodlight poles.
        for fx, fy in ((115, -175), (285, -175), (115, 175), (285, 175)):
            pole = NodePath(make_box_node("flood_pole", 0.4, 0.4, 20, dark))
            pole.setPos(fx, fy, 10)
            pole.reparentTo(self.render)
            head = NodePath(make_box_node("flood_head", 3.2, 1.0, 0.8,
                                          (0.2, 0.2, 0.22)))
            head.setPos(fx, fy, 20.3)
            head.reparentTo(self.render)
            fcm = CardMaker("flood_glow")
            fcm.setFrame(-2.2, 2.2, -2.2, 2.2)
            g = self.render.attachNewNode(fcm.generate())
            g.setTexture(self.glow_tex)
            g.setColor(1.0, 0.92, 0.70, 0.9)
            g.setBillboardPointEye()
            g.setTransparency(TransparencyAttrib.MAlpha)
            g.setLightOff()
            g.setDepthWrite(False)
            g.setPos(fx, fy, 19.7)

        # Operations building.
        ops = NodePath(make_box_node("ops", 34, 16, 9, gray))
        ops.setPos(160, -205, 4.5)
        ops.reparentTo(self.render)
        opsr = NodePath(make_box_node("ops_roof", 36, 18, 1.0, dark))
        opsr.setPos(160, -205, 9.5)
        opsr.reparentTo(self.render)

        # Fuel farm: three cylindrical tanks.
        for i, tx in enumerate((146, 162, 178)):
            tank = NodePath(make_cylinder_node("fueltank", 6.0, 10.0,
                                               (0.78, 0.77, 0.72)))
            tank.setPos(tx, -320, 0)
            tank.reparentTo(self.render)
            band = NodePath(make_cylinder_node("tankband", 6.15, 1.2,
                                               (0.72, 0.10, 0.10)))
            band.setPos(tx, -320, 8.0)
            band.reparentTo(self.render)

        # Approach light masts past both runway thresholds.
        for end in (-1, 1):
            for k in range(5):
                ay = end * (660.0 + k * 30.0)
                mast = NodePath(make_box_node("amast", 0.35, 0.35, 7, dark))
                mast.setPos(0, ay, 3.5)
                mast.reparentTo(self.render)
                bar = NodePath(make_box_node("ambar", 5.0, 0.35, 0.35, dark))
                bar.setPos(0, ay, 7.0)
                bar.reparentTo(self.render)
                acm = CardMaker("aglow")
                acm.setFrame(-1.4, 1.4, -1.4, 1.4)
                ag = self.render.attachNewNode(acm.generate())
                ag.setTexture(self.glow_tex)
                ag.setColor(1.0, 1.0, 1.0, 0.95)
                ag.setBillboardPointEye()
                ag.setTransparency(TransparencyAttrib.MAlpha)
                ag.setLightOff()
                ag.setDepthWrite(False)
                ag.setPos(0, ay, 7.0)

        # Windsock near the south end.
        pole = NodePath(make_box_node("pole", 0.18, 0.18, 6.5,
                                      (0.75, 0.75, 0.78)))
        pole.setPos(28, -560, 3.25)
        pole.reparentTo(self.render)
        self.sock_pivot = self.render.attachNewNode("sock_pivot")
        self.sock_pivot.setPos(28, -560, 6.4)
        fmt = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData("sock", fmt, Geom.UHStatic)
        vw = GeomVertexWriter(vdata, "vertex")
        cw = GeomVertexWriter(vdata, "color")
        segs, sock_len = 10, 2.6
        for ring, (yy, rr) in enumerate(((0.0, 0.50), (sock_len, 0.16))):
            for i in range(segs):
                a = 2.0 * math.pi * i / segs
                vw.addData3(rr * math.cos(a), yy, rr * math.sin(a))
                if ring == 0:
                    cw.addData4(0.95, 0.45, 0.10, 1.0)
                else:
                    cw.addData4(0.94, 0.94, 0.94, 1.0)
        tris = GeomTriangles(Geom.UHStatic)
        for i in range(segs):
            a, b = i, (i + 1) % segs
            c, d = segs + i, segs + (i + 1) % segs
            tris.addVertices(a, c, b)
            tris.addVertices(b, c, d)
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        sock = NodePath(GeomNode("sock"))
        sock.node().addGeom(geom)
        sock.setTwoSided(True)
        sock.reparentTo(self.sock_pivot)

        # Trees: pines scattered clear of the runway and the base complex.
        trunk_c = (0.30, 0.22, 0.15)
        leaf_c = [(0.16, 0.34, 0.16), (0.20, 0.40, 0.18)]
        placed = 0
        while placed < 60:
            x = rnd.uniform(-1900, 1900)
            y = rnd.uniform(-1900, 1900)
            if abs(x) < 55 and abs(y) < 700:
                continue
            if x > 40 and abs(y) < 720:
                continue
            if math.hypot(x - 330, y - 260) < 70:
                continue
            if math.hypot(x, y) < 230:
                continue
            s = rnd.uniform(0.8, 1.7)
            trunk = NodePath(make_box_node("trunk", 0.5 * s, 0.5 * s,
                                           2.2 * s, trunk_c))
            trunk.setPos(x, y, 1.1 * s)
            trunk.reparentTo(self.render)
            lc = leaf_c[placed % 2]
            for rr, hh, zz in ((2.6, 4.2, 2.0), (1.8, 3.0, 4.6)):
                cone = NodePath(make_cone_node("pine", rr * s, hh * s, lc))
                cone.setPos(x, y, zz * s)
                cone.reparentTo(self.render)
            placed += 1

        # Runway edge lights: small posts with a warm glow.
        for y in range(-600, 601, 60):
            for x in (-16.5, 16.5):
                post = NodePath(make_box_node("elpost", 0.14, 0.14, 0.9,
                                              (0.2, 0.2, 0.22)))
                post.setPos(x, y, 0.45)
                post.reparentTo(self.render)
                lcm = CardMaker("elglow")
                lcm.setFrame(-0.8, 0.8, -0.8, 0.8)
                g = self.render.attachNewNode(lcm.generate())
                g.setTexture(self.glow_tex)
                g.setColor(1.0, 0.85, 0.55, 0.85)
                g.setBillboardPointEye()
                g.setTransparency(TransparencyAttrib.MAlpha)
                g.setLightOff()
                g.setDepthWrite(False)
                g.setPos(x, y, 1.1)

    def _build_static_jet(self, x, y, heading):
        """Simple parked military jet silhouette (procedural boxes)."""
        jet = self.render.attachNewNode("static_jet")
        gray = (0.44, 0.45, 0.48)
        dark = (0.14, 0.14, 0.16)
        glass = (0.10, 0.14, 0.20)
        parts = [
            ("jfus", 1.7, 11.0, 1.7, gray, (0, 0, 1.7)),
            ("jcanopy", 0.95, 2.6, 0.75, glass, (0, 2.4, 2.75)),
            ("jtail", 2.6, 1.0, 0.12, gray, (0, -4.7, 1.8)),
            ("jfin", 0.14, 1.3, 2.1, gray, (0, -4.7, 2.9)),
            ("jnozzle", 1.0, 0.9, 1.0, dark, (0, -5.7, 1.7)),
            ("jgear_n", 0.3, 0.3, 1.0, dark, (0, 3.4, 0.5)),
            ("jgear_l", 0.3, 0.3, 1.0, dark, (-1.1, -0.6, 0.5)),
            ("jgear_r", 0.3, 0.3, 1.0, dark, (1.1, -0.6, 0.5)),
        ]
        for name, ex, ey, ez, color, pos in parts:
            node = NodePath(make_box_node(name, ex, ey, ez, color))
            node.setPos(*pos)
            node.setTwoSided(True)
            node.reparentTo(jet)
        # Swept wings and tailplanes (tips angled back).
        for side in (-1, 1):
            wing = NodePath(make_box_node("jwing", 5.4, 1.7, 0.14, gray))
            wing.setPos(side * 3.1, -0.9, 1.55)
            wing.setH(-side * 28.0)
            wing.setTwoSided(True)
            wing.reparentTo(jet)
            stab = NodePath(make_box_node("jstab", 2.4, 0.9, 0.10, gray))
            stab.setPos(side * 1.5, -4.7, 1.8)
            stab.setH(-side * 22.0)
            stab.setTwoSided(True)
            stab.reparentTo(jet)
        # Pointed nose.
        nose = NodePath(make_cone_node("jnose", 0.85, 2.4, gray))
        nose.setP(-90)  # apex forward (+Y)
        nose.setPos(0, 5.5, 1.7)
        nose.reparentTo(jet)
        jet.setPos(x, y, 0)
        jet.setH(heading)

    # -- airplane ---------------------------------------------------------
    def _build_airplane(self):
        """Refined C172-ish model (panda axes: x span, y length, z up)."""
        white = (0.93, 0.93, 0.95)
        red = (0.72, 0.10, 0.10)
        dark = (0.10, 0.11, 0.14)
        glass_c = (0.13, 0.19, 0.28)

        self.plane = self.render.attachNewNode("plane")

        parts = [
            ("fuselage", 1.15, 8.6, 1.35, white, (0, 0.3, 0.15)),
            ("stripe", 1.17, 8.2, 0.20, red, (0, 0.25, -0.28)),
            ("wing", 11.0, 1.6, 0.12, white, (0, 0.7, 1.02)),
            ("wingtip_l", 0.55, 0.45, 0.14, red, (-5.35, 0.7, 1.02)),
            ("wingtip_r", 0.55, 0.45, 0.14, red, (5.35, 0.7, 1.02)),
            ("hstab", 3.4, 1.0, 0.10, white, (0, -3.6, 0.35)),
            ("fin", 0.12, 1.05, 1.75, red, (0, -3.6, 1.05)),
            ("finstripe", 0.13, 0.9, 0.25, white, (0, -3.6, 1.55)),
            ("fair_l", 0.38, 0.95, 0.55, white, (-0.85, 1.6, -0.78)),
            ("fair_r", 0.38, 0.95, 0.55, white, (0.85, 1.6, -0.78)),
            ("fair_n", 0.32, 0.80, 0.50, white, (0, 3.1, -0.75)),
            ("strut_l", 0.14, 0.14, 0.85, dark, (-0.85, 1.6, -0.28)),
            ("strut_r", 0.14, 0.14, 0.85, dark, (0.85, 1.6, -0.28)),
            ("strut_n", 0.12, 0.12, 0.75, dark, (0, 3.1, -0.28)),
        ]
        for name, ex, ey, ez, color, pos in parts:
            node = NodePath(make_box_node(name, ex, ey, ez, color))
            node.setPos(*pos)
            node.setTwoSided(True)
            node.reparentTo(self.plane)

        # Tinted canopy.
        cabin = NodePath(make_box_node("cabin", 1.0, 2.2, 0.72, glass_c))
        cabin.setPos(0, 1.15, 0.98)
        cabin.setTwoSided(True)
        cabin.setTransparency(TransparencyAttrib.MAlpha)
        cabin.setAlphaScale(0.92)
        cabin.reparentTo(self.plane)

        # High-wing struts (diagonal, in the x-z plane).
        for side in (-1, 1):
            strut = NodePath(make_box_node("wstrut", 0.09, 0.55, 2.15, white))
            strut.setPos(side * 1.75, 0.7, 0.32)
            strut.setR(side * -53.0)
            strut.setTwoSided(True)
            strut.reparentTo(self.plane)

        # Propeller: blades in the x-z plane at the nose, spinning about y.
        self.prop = self.plane.attachNewNode("prop")
        for name, ex, ez in (("blade", 0.11, 2.1), ("blade2", 2.1, 0.11)):
            blade = NodePath(make_box_node(name, ex, 0.06, ez, dark))
            blade.setTwoSided(True)
            blade.reparentTo(self.prop)
        spinner = NodePath(make_cone_node("spinner", 0.22, 0.5, red))
        spinner.setP(-90)  # apex forward (+Y)
        spinner.setPos(0, 0.1, 0)
        spinner.reparentTo(self.prop)
        self.prop.setPos(0, 4.75, 0.15)

        # Translucent blur disc that fades in with throttle.
        self.prop_disc = NodePath(make_disc_node("propdisc", 1.12))
        self.prop_disc.setTexture(self._make_prop_texture())
        self.prop_disc.setTransparency(TransparencyAttrib.MAlpha)
        self.prop_disc.setLightOff()
        self.prop_disc.setDepthWrite(False)
        self.prop_disc.setTwoSided(True)
        self.prop_disc.setPos(0, 4.78, 0.15)
        self.prop_disc.reparentTo(self.plane)

        # Navigation lights + tail strobe (billboarded glows).
        def nav_light(x, y, z, color):
            lcm = CardMaker("nav")
            lcm.setFrame(-0.28, 0.28, -0.28, 0.28)
            n = self.plane.attachNewNode(lcm.generate())
            n.setTexture(self.glow_tex)
            n.setColor(*color)
            n.setBillboardPointEye()
            n.setTransparency(TransparencyAttrib.MAlpha)
            n.setLightOff()
            n.setDepthWrite(False)
            n.setPos(x, y, z)
            return n

        nav_light(-5.55, 0.7, 1.02, (1.0, 0.12, 0.12, 0.95))   # left red
        nav_light(5.55, 0.7, 1.02, (0.15, 1.0, 0.25, 0.95))     # right green
        self.strobe = nav_light(0, -4.15, 0.55, (1.0, 1.0, 1.0, 1.0))

    # -- HUD --------------------------------------------------------------
    def _label(self, pos, scale=0.040, align=TextNode.ALeft,
               fg=(1, 1, 1, 1)):
        """Small clean text with a soft shadow (no bulky background box)."""
        return OnscreenText(text="", pos=pos, scale=scale, align=align,
                            fg=fg, shadow=(0, 0, 0, 0.75), mayChange=True)

    def _build_hud(self):
        # All readouts live on the left side of the screen.
        self.hud_left = self._label((-1.72, 0.90))
        self.hud_right = self._label((-1.72, 0.60))
        self.hud_hint = self._label((-1.72, -0.94), scale=0.030,
                                    fg=(0.85, 0.85, 0.88, 0.9))
        self.hud_hint.setText("H help  |  C camera  |  P pause  |  R reset")
        self.hud_warn = self._label((0, 0.78), scale=0.058,
                                    align=TextNode.ACenter, fg=(1, 0.18, 0.18, 1))
        self.hud_status = self._label((0, 0.66), scale=0.038,
                                      align=TextNode.ACenter,
                                      fg=(0.45, 1.0, 0.45, 1))
        self._build_attitude_indicator()
        self.help_text = OnscreenText(text=HELP_TEXT, pos=(-1.70, 0.30),
                                     scale=0.034, align=TextNode.ALeft,
                                     fg=(1, 1, 1, 1), bg=(0, 0, 0, 0.55),
                                     shadow=(0, 0, 0, 0.6), mayChange=False)

    def _build_attitude_indicator(self):
        """Compact AI: blue/brown ball that pitches and rolls under a fixed
        symbol. Smaller than before, tucked at bottom-center."""
        root = NodePath("ai")
        root.reparentTo(self.aspect2d)
        root.setPos(0, 0, -0.74)
        S = 0.15  # half-size

        frame = CardMaker("ai_frame")
        frame.setFrame(-S, S, -S, S)
        fn = root.attachNewNode(frame.generate())
        fn.setColor(0.05, 0.05, 0.08, 0.85)
        fn.setBin("fixed", 10)
        fn.setDepthTest(False)

        self.ai_ball = root.attachNewNode("ai_ball")
        for nm, y0, y1, col in (("ai_sky", 0.0, 2 * S, (0.20, 0.45, 0.85, 1.0)),
                                ("ai_gnd", -2 * S, 0.0, (0.55, 0.38, 0.22, 1.0))):
            cm = CardMaker(nm)
            cm.setFrame(-S * 0.92, S * 0.92, y0, y1)
            n = self.ai_ball.attachNewNode(cm.generate())
            n.setColor(*col)
            n.setBin("fixed", 11)
            n.setDepthTest(False)
        # Horizon line.
        hcm = CardMaker("ai_horizon")
        hcm.setFrame(-S * 0.92, S * 0.92, -0.006, 0.006)
        hn = self.ai_ball.attachNewNode(hcm.generate())
        hn.setColor(1, 1, 1, 0.9)
        hn.setBin("fixed", 11)
        hn.setDepthTest(False)

        # Fixed aircraft symbol (stays put while the ball moves).
        sym = root.attachNewNode("ai_sym")
        for x0, x1 in [(-0.10, -0.025), (0.025, 0.10)]:
            wcm = CardMaker("ai_wing")
            wcm.setFrame(x0, x1, -0.010, 0.010)
            wn = sym.attachNewNode(wcm.generate())
            wn.setColor(1, 0.85, 0.2, 1)
            wn.setBin("fixed", 12)
            wn.setDepthTest(False)
        dot_cm = CardMaker("ai_dot")
        dot_cm.setFrame(-0.010, 0.010, -0.010, 0.010)
        dn = sym.attachNewNode(dot_cm.generate())
        dn.setColor(1, 0.85, 0.2, 1)
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
        self.cam_mode = (self.cam_mode + 1) % 4
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
        self.fm.pos = np.array([-480.0, 0.0, GEAR_HEIGHT])
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
        # Prop blur fades in with throttle; strobe double-blinks.
        self.prop_disc.setAlphaScale(0.08 + 0.55 * self.ctrl.throttle)
        phase = (self.fm.time * 1.1) % 1.0
        if phase < 0.06 or (0.18 < phase < 0.24):
            self.strobe.show()
        else:
            self.strobe.hide()
        # Windsock sway + radar sweep.
        t = self.fm.time
        self.sock_pivot.setH(18.0 * math.sin(t * 0.6) + 8.0 * math.sin(t * 1.7))
        self.radar_dome.setH((t * 25.0) % 360.0)
        # Clouds drift and wrap.
        for node, speed in self.clouds:
            x = node.getX() + speed * dt
            if x > 2900:
                x = -2900
            node.setX(x)
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
        # Sun glow follows the camera at a fixed offset toward the sun.
        cx, cy, cz = self.camera.getPos()
        self.sun_card.setPos(cx + SUN_DIR[0] * 7000.0,
                             cy + SUN_DIR[1] * 7000.0,
                             cz + SUN_DIR[2] * 7000.0)
        px, py, pz = sim_to_panda(self.fm.pos)
        h, p, r = sim_quat_to_hpr(self.fm.quat)
        plane_pos = np.array([px, py, pz])
        # Forward vector in panda frame from heading/pitch.
        hr, pr = math.radians(h), math.radians(p)
        fx, fy, fz = (math.sin(hr) * math.cos(pr),
                      math.cos(hr) * math.cos(pr), math.sin(pr))
        back = np.array([fx, fy, fz])
        if self.cam_mode == 0:  # chase
            self.plane.show()
            # 16 m behind, 5 m above, in the plane's frame.
            target = plane_pos - back * 16.0 + np.array([0, 0, 5.0])
            cur = np.array(self.camera.getPos())
            smooth = cur + (target - cur) * min(1.0, dt * 4.0)
            self.camera.setPos(*smooth)
            self.camera.lookAt(px, py, pz)
        elif self.cam_mode == 1:  # cockpit
            self.plane.hide()
            self.camera.setPos(px, py + 0.9, pz + 1.55)
            self.camera.setHpr(h, p, r)
        elif self.cam_mode == 2:  # tower: from the control-tower cab
            self.plane.show()
            self.camera.setPos(200, -240, 38)
            self.camera.lookAt(px, py, pz)
        else:  # top-down: high above and slightly behind, looking down
            self.plane.show()
            target = plane_pos - back * 18.0 + np.array([0, 0, 55.0])
            cur = np.array(self.camera.getPos())
            smooth = cur + (target - cur) * min(1.0, dt * 3.0)
            self.camera.setPos(*smooth)
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

        thr_n = int(round(c.throttle * 10))
        thr_bar = "[" + "#" * thr_n + "-" * (10 - thr_n) + "]"
        self.hud_left.setText(
            f"SPD {kts:5.0f} kt\nALT {alt_ft:6.0f} ft\n"
            f"HDG {hdg:03.0f}\nVS {vs_fpm:+5.0f} fpm")
        self.hud_right.setText(
            f"THR {thr_bar} {c.throttle * 100:3.0f}%\n"
            f"FLAP {c.flap_degrees():2d} deg\n"
            f"TRIM {c.trim:+.2f}\nBRK {'ON' if c.brakes else 'off'}")
        if info.get("stalled"):
            self.hud_warn.setText("*** STALL ***" if (fm.time % 1.0) < 0.6 else "")
        else:
            self.hud_warn.setText("")
        if self.paused:
            self.hud_status.setText("PAUSED - press P to resume")
        elif fm.on_ground:
            self.hud_status.setText("ON GROUND")
        else:
            self.hud_status.setText("")

        # Attitude indicator: the ball carries the horizon; when the aircraft
        # rolls right (roll is left-positive, so roll < 0), the horizon appears
        # to rotate counter-clockwise -> ball rotation = -roll. Nose-up pushes
        # the horizon down -> ball shifts down.
        self.ai_ball.setR(-roll * DEG)
        self.ai_ball.setZ(-pitch * 0.55)


def main():
    """Entry point for the graphical simulator."""
    loadPrcFileData("", "window-title Python Flight Simulator")
    loadPrcFileData("", "win-size 1280 720")
    loadPrcFileData("", "framebuffer-multisample 1")
    app = FlightSim()
    app.run()


if __name__ == "__main__":
    main()
