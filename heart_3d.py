"""
Interactive 3D heart  (PyVista)

Install:   pip install pyvista numpy
Run:       python heart_3d.py

Controls
  Left-drag      rotate           Scroll         zoom
  Right-click    identify a part  Sliders        exercise + energy drinks
  Keys 1-4       presets: 1 rest, 2 energy drink, 3 heavy run, 4 run + drink

Physiology values are typical textbook numbers for illustration, not medical advice.
"""
import sys
import textwrap
import time

import numpy as np
import pyvista as pv

BLUE = np.array([0.23, 0.50, 0.77])
RED = np.array([0.89, 0.29, 0.29])
LIGHT = np.array([1.0, 0.65, 0.60])

# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
pl = pv.Plotter(off_screen="--smoke" in sys.argv, window_size=(1200, 800),
                title="Interactive heart")
pl.set_background("#fbfaf7", top="#dde5ee")

parts = {}  # actor id -> (name, description)


def add_chamber(name, info, center, radii, tilt_deg, color):
    """Ellipsoid chamber. Scaling the actor about its center makes it 'beat'."""
    m = pv.Sphere(radius=1, theta_resolution=64, phi_resolution=64)
    m.scale(radii, inplace=True)
    m.rotate_z(tilt_deg, inplace=True)
    m.translate(center, inplace=True)
    actor = pl.add_mesh(m, color=color, opacity=0.5, smooth_shading=True,
                        specular=0.6, specular_power=30)
    actor.origin = center
    parts[id(actor)] = (name, info)
    return actor


def add_vessel(name, info, pts, radius, color):
    tube = pv.Spline(np.array(pts, float), 80).tube(radius=radius, n_sides=24)
    actor = pl.add_mesh(tube, color=color, opacity=0.6, smooth_shading=True,
                        specular=0.5)
    parts[id(actor)] = (name, info)
    return actor


RA = add_chamber("Right atrium",
                 "Collects oxygen-poor blood from the body and passes it to the right ventricle.",
                 (-1.15, 0.85, 0.05), (0.80, 0.85, 0.75), 0, BLUE)
LA = add_chamber("Left atrium",
                 "Receives oxygen-rich blood returning from the lungs.",
                 (1.0, 0.85, -0.5), (0.75, 0.75, 0.70), 0, RED)
RV = add_chamber("Right ventricle",
                 "Pumps blood to the lungs at fairly low pressure.",
                 (-0.45, -0.55, 0.3), (0.85, 1.20, 0.80), 11, BLUE)
LV = add_chamber("Left ventricle",
                 "The powerhouse. Its thick wall drives blood around the whole body.",
                 (0.55, -0.65, 0.0), (0.95, 1.45, 0.90), 17, RED)

AORTA_PTS = [(0.4, 0.3, 0.1), (0.4, 1.6, 0.05), (0.9, 2.55, -0.1),
             (1.7, 2.15, -0.3), (1.95, 0.9, -0.4)]
PA_PTS = [(-0.3, 0.2, 0.5), (-0.1, 1.4, 0.6), (0.1, 2.4, 0.45)]
aorta = add_vessel("Aorta", "Main artery carrying oxygen-rich blood out to the body.",
                   AORTA_PTS, 0.22, RED)
pulm_art = add_vessel("Pulmonary artery",
                      "Carries oxygen-poor blood from the right ventricle to the lungs.",
                      PA_PTS, 0.17, BLUE)
add_vessel("Superior vena cava", "Large vein returning blood from the upper body.",
           [(-1.35, 2.9, 0), (-1.25, 1.8, 0.02), (-1.15, 1.2, 0.05)], 0.17, BLUE)
add_vessel("Inferior vena cava", "Large vein returning blood from the lower body.",
           [(-1.8, -2.0, 0), (-1.75, -0.3, 0), (-1.4, 0.6, 0.05)], 0.19, BLUE)
add_vessel("Pulmonary vein", "Brings oxygen-rich blood from the lungs to the left atrium.",
           [(2.4, 1.5, -0.95), (1.6, 1.1, -0.6), (1.2, 0.9, -0.5)], 0.13, RED)
add_vessel("Pulmonary vein", "Brings oxygen-rich blood from the lungs to the left atrium.",
           [(2.3, 0.2, -1.0), (1.5, 0.55, -0.6), (1.2, 0.7, -0.5)], 0.13, RED)

# --------------------------------------------------------------------------
# Blood particles that flow along each circuit
# --------------------------------------------------------------------------
rng = np.random.default_rng(1)
R_MID = [(-0.85, 0.0, 0.2), (-0.5, -0.6, 0.3)] + PA_PTS
L_MID = [(0.7, 0.0, -0.2), (0.55, -0.7, 0.0)] + AORTA_PTS

STREAM_DEFS = [  # (control points, color)
    ([(-1.35, 2.9, 0), (-1.25, 1.8, 0.02), (-1.15, 0.85, 0.05)] + R_MID, (0.62, 0.80, 1.0)),
    ([(-1.8, -2.0, 0), (-1.75, -0.3, 0), (-1.4, 0.6, 0.05), (-1.15, 0.85, 0.05)] + R_MID, (0.62, 0.80, 1.0)),
    ([(2.4, 1.5, -0.95), (1.6, 1.1, -0.6), (1.0, 0.85, -0.45)] + L_MID, (1.0, 0.70, 0.66)),
    ([(2.3, 0.2, -1.0), (1.5, 0.55, -0.6), (1.0, 0.85, -0.45)] + L_MID, (1.0, 0.70, 0.66)),
]


class Stream:
    def __init__(self, ctrl, color, n=55):
        self.path = pv.Spline(np.array(ctrl, float), 600).points
        self.u = np.linspace(0, 1, n, endpoint=False)
        self.jit = rng.normal(0, 0.05, (n, 3))
        self.mesh = pv.PolyData(self._pos())
        pl.add_mesh(self.mesh, color=color, point_size=7,
                    render_points_as_spheres=True)

    def _pos(self):
        idx = np.minimum((self.u * (len(self.path) - 1)).astype(int), len(self.path) - 1)
        return self.path[idx] + self.jit

    def step(self, du):
        self.u = (self.u + du) % 1.0
        self.mesh.points = self._pos()


streams = [Stream(c, col) for c, col in STREAM_DEFS]

# --------------------------------------------------------------------------
# Physiology model
# --------------------------------------------------------------------------
st = dict(ex=0.0, ca=0.0, tex=0.0, tca=0.0, phase=0.0, last=time.perf_counter(),
          hud_t=0.0)


def vitals(ex, ca):
    hr = min(200.0, 68 + ex * 105 + ca * 4 + ex * ca * 2)   # beats per minute
    sv = 70 + ex * 50 + ca * 2                              # stroke volume, mL
    sys_bp = 118 + ex * 55 + ca * 4                         # mmHg
    dia_bp = 78 - ex * 5 + ca * 2
    return hr, sv, sys_bp, dia_bp, hr * sv / 1000.0


def beat(t, hr):
    """Atrial squeeze first, then ventricular systole. Returns (atrial, ventricular)."""
    sf = max(0.28, 0.42 - (hr - 60) * 0.001)
    a = np.sin(np.pi * t / 0.12) if t < 0.12 else 0.0
    v = np.sin(np.pi * (t - 0.12) / sf) if 0.12 <= t < 0.12 + sf else 0.0
    return a, v


def story(ex, ca):
    if ca > 0 and ex < 0.15:
        s = ("Caffeine blocks adenosine, which normally calms the heart, and nudges "
             "adrenaline up. Beats get a little faster and stronger and blood pressure "
             "creeps up. Real effects peak 30-60 minutes after drinking; here they "
             "build over a few seconds. Sensitive people may feel palpitations.")
    elif ex >= 0.15 and ca > 0:
        s = ("Caffeine on top of hard exercise stacks load on load: higher rate, "
             "higher pressure, more work per minute. Healthy hearts cope, but this is "
             "where undiagnosed heart conditions can show up.")
    elif ex >= 0.15:
        s = ("Working muscles need oxygen, so the nervous system speeds the heart up "
             "and makes each beat stronger. Cardiac output can rise four- to five-fold. "
             "At very high rates there is less time between beats to fill.")
    else:
        s = ("At rest the heart beats about 70 times a minute and moves roughly "
             "5 litres of blood every minute.")
    pl.add_text("\n".join(textwrap.wrap(s, 40)), position="upper_right",
                font_size=10, color="#333333", name="story")


# --------------------------------------------------------------------------
# Animation tick
# --------------------------------------------------------------------------
def tick():
    now = time.perf_counter()
    dt = min(0.05, now - st["last"])
    st["last"] = now

    st["ex"] += (st["tex"] - st["ex"]) * (1 - np.exp(-dt * 1.5))   # fast response
    st["ca"] += (st["tca"] - st["ca"]) * (1 - np.exp(-dt * 0.35))  # caffeine builds slowly
    hr, sv, sbp, dbp, co = vitals(st["ex"], st["ca"])

    st["phase"] += dt * hr / 60.0
    a, v = beat(st["phase"] % 1.0, hr)

    depth = 0.08 + 0.10 * np.clip((sv - 60) / 80, 0, 1)
    for ch in (RA, LA):
        s = 1 - 0.08 * a
        ch.scale = (s, s, s)
    for ch in (RV, LV):
        ch.scale = (1 - depth * v, 1 - depth * 0.6 * v, 1 - depth * v)

    aorta.prop.color = tuple(RED * (1 - 0.45 * v) + LIGHT * 0.45 * v)
    pulm_art.prop.color = tuple(BLUE * (1 - 0.45 * v) + np.array([0.6, 0.8, 1.0]) * 0.45 * v)

    speed = 0.18 * (co / 4.8) * (0.6 + 0.8 * v) * dt
    for s_ in streams:
        s_.step(speed)

    st["hud_t"] += dt
    if st["hud_t"] > 0.1:
        st["hud_t"] = 0
        wl = hr * sbp / (68 * 118)
        pl.add_text(
            f"Heart rate       {hr:5.0f} bpm\n"
            f"Blood pressure  {sbp:3.0f}/{dbp:.0f} mmHg\n"
            f"Blood pumped    {co:5.1f} L/min\n"
            f"Heart workload  {wl:5.1f}x rest",
            position="upper_left", font_size=12, font="courier",
            color="#222222", name="hud")
    pl.render()


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
def set_ex(v):
    st["tex"] = v / 100.0
    story(st["tex"], st["tca"])


def set_ca(v):
    st["tca"] = float(v)
    story(st["tex"], st["tca"])


slider_ex = pl.add_slider_widget(set_ex, [0, 100], value=0, title="Exercise intensity (0 = rest, 100 = sprint)",
                                 pointa=(0.05, 0.12), pointb=(0.45, 0.12), color="#222222",
                                 title_height=0.02, style="modern")
slider_ca = pl.add_slider_widget(set_ca, [0, 4], value=0, title="Energy drinks (cans, ~80 mg caffeine each)",
                                 pointa=(0.55, 0.12), pointb=(0.95, 0.12), color="#222222",
                                 title_height=0.02, style="modern")


def preset(ex, ca):
    slider_ex.GetRepresentation().SetValue(ex)
    slider_ca.GetRepresentation().SetValue(ca)
    set_ex(ex)
    set_ca(ca)


pl.add_key_event("1", lambda: preset(0, 0))
pl.add_key_event("2", lambda: preset(0, 2))
pl.add_key_event("3", lambda: preset(90, 0))
pl.add_key_event("4", lambda: preset(90, 2))

pl.add_text("Keys 1-4: rest, energy drink, heavy run, run + drink  |  "
            "Drag: rotate  |  Right-click: identify part",
            position="lower_edge", font_size=8, color="#555555", name="help")
pl.add_text("Blue = oxygen-poor blood   Red = oxygen-rich blood", position=(20, 560),
            font_size=9, color="#555555", name="legend")


def on_pick(actor):
    if actor is not None and id(actor) in parts:
        name, info = parts[id(actor)]
        txt = "\n".join(textwrap.wrap(f"{name}: {info}", 44))
        pl.add_text(txt, position=(20, 170), font_size=10, color="#8a2a12", name="part")


try:
    pl.enable_mesh_picking(on_pick, use_actor=True, show=False, show_message=False,
                         left_clicking=False)
except Exception as exc:  # picking API differs slightly between PyVista versions
    print("Part picking unavailable:", exc)

story(0, 0)
pl.camera_position = [(0.3, 0.1, 17.0), (0.3, 0.1, 0.0), (0, 1, 0)]

if "--smoke" in sys.argv:           # headless self-test: python heart_3d.py --smoke
    preset(90, 2)
    for _ in range(120):
        tick()
    pl.show(screenshot="heart_smoke.png")
else:
    pl.add_callback(tick, interval=30)
    pl.show()
