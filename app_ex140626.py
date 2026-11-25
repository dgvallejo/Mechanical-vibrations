"""
Simulación del sistema de ensayo de impactos (polea + barra + muelles-amortiguadores + masa m0)
Mecánica de Máquinas y Vibraciones – 3º Grado en Ingeniería Aeroespacial

Modelo lineal de 1 GDL en el ángulo de la polea θ(t) (θ > 0 antihorario, la punta de la barra
se mueve hacia la derecha):

    I·θ'' + cθ·θ' + kθ·θ = 0
    I  = I_Gp + m_b (2L/5)² + I_Gb
    kθ = 2 k1 R² + k2 (2L/3)² + m_b g (2L/5)
    cθ = 2 ξ √(I kθ),   ξ a partir del decremento logarítmico

Secuencia del ensayo:
  1) Par estático M0 → la barra queda girada θ0 (sentido horario, θ0 < 0).  M0 = kθ |θ0|
  2) Se elimina M0 en t = 0: vibración libre desde el reposo hasta θ = 0 (impacto con m0).
  3) Impacto con coeficiente de restitución e:
        (I + m0 L²) θ'⁺ = (I − e m0 L²) θ'⁻ ,   x0'⁺ = L (θ'⁺ + e θ'⁻)
  4) Vibración libre tras el impacto con θ(0) = 0, θ'(0) = θ'⁺.

Ejecutar:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Ensayo de impactos – vibraciones", page_icon="🔩", layout="wide")

# =============================================================================
# Parámetros (barra lateral)
# =============================================================================
DEF = dict(L=1.0, R=0.2, k1=100.0, k2=30.0, mb=1.0, IGb=0.05, IGp=0.05, m0=1.0,
           red=95.0, ciclos=20, th0=5.0, e=0.2, g=9.81)

for _k, _v in DEF.items():
    st.session_state.setdefault(_k, _v)

if st.sidebar.button("↺ Restaurar valores del enunciado", use_container_width=True):
    for k, v in DEF.items():
        st.session_state[k] = v

st.sidebar.header("Geometría")
L = st.sidebar.number_input("Longitud de la barra L [m]", 0.1, 5.0, step=0.05, key="L")
R = st.sidebar.number_input("Radio de la polea R [m]", 0.01, 1.0, step=0.01, key="R")

st.sidebar.header("Rigideces")
k1 = st.sidebar.number_input("k1 [N/m] (cada muelle del cable)", 0.0, 1e5, step=5.0, key="k1")
k2 = st.sidebar.number_input("k2 [N/m]", 0.0, 1e5, step=5.0, key="k2")

st.sidebar.header("Masas e inercias")
mb = st.sidebar.number_input("Masa de la barra m_b [kg]", 0.0, 100.0, step=0.1, key="mb")
IGb = st.sidebar.number_input("I_Gb [kg·m²]", 0.0, 10.0, step=0.01, format="%.3f", key="IGb")
IGp = st.sidebar.number_input("I_Gp [kg·m²]", 0.0, 10.0, step=0.01, format="%.3f", key="IGp")
m0 = st.sidebar.number_input("Masa impactada m0 [kg]", 0.0, 100.0, step=0.1, key="m0")

st.sidebar.header("Amortiguamiento")
red = st.sidebar.slider("Reducción de amplitud [%]", 1.0, 99.9, step=0.1, key="red")
ciclos = st.sidebar.number_input("… al cabo de N ciclos", 1, 500, step=1, key="ciclos")

st.sidebar.header("Ensayo")
th0_deg = st.sidebar.slider("Giro inicial impuesto por M0 [°] (horario)", 0.5, 30.0, step=0.5, key="th0")
e = st.sidebar.slider("Coeficiente de restitución e", 0.0, 1.0, step=0.05, key="e")
g = st.sidebar.number_input("g [m/s²]", 0.0, 30.0, step=0.01, key="g")

# =============================================================================
# Cálculos (apartados del examen)
# =============================================================================
I = IGp + mb * (2 * L / 5) ** 2 + IGb
k_cab, k_k2, k_grav = 2 * k1 * R**2, k2 * (2 * L / 3) ** 2, mb * g * 2 * L / 5
kt = k_cab + k_k2 + k_grav

if I <= 0 or kt <= 0:
    st.error("La inercia y la rigidez equivalentes deben ser positivas.")
    st.stop()

delta = np.log(1 / (1 - red / 100)) / ciclos
xi = delta / np.sqrt(4 * np.pi**2 + delta**2)
wn = np.sqrt(kt / I)
wd = wn * np.sqrt(1 - xi**2)
ct = 2 * xi * np.sqrt(I * kt)
c1 = ct / (2 * R**2)  # amortiguador de cada rama del cable

th0 = -np.radians(th0_deg)        # giro horario → θ0 < 0
M0 = kt * abs(th0)
dl_min = abs(th0) * R

# Fase 1: hasta el impacto
t_imp = (np.pi - np.arctan(wd / (xi * wn))) / wd
thd_m = -(wn**2 / wd) * th0 * np.exp(-xi * wn * t_imp) * np.sin(wd * t_imp)

# Impacto
thd_p = (I - e * m0 * L**2) * thd_m / (I + m0 * L**2)
x0d_p = L * (thd_p + e * thd_m)
Ic = m0 * x0d_p

# Fase 2: máximo tras el impacto
t_pk = np.arctan(wd / (xi * wn)) / wd
th_max = np.exp(-xi * wn * t_pk) * thd_p / wd * np.sin(wd * t_pk)

# Pérdida de energía en el impacto
E_antes = 0.5 * I * thd_m**2
E_desp = 0.5 * I * thd_p**2 + 0.5 * m0 * x0d_p**2

# =============================================================================
# Respuesta temporal
# =============================================================================
def respuesta(t):
    t = np.asarray(t, dtype=float)
    th, thd, x0 = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    a = t < t_imp
    ta = t[a]
    env = np.exp(-xi * wn * ta)
    th[a] = env * th0 * (np.cos(wd * ta) + xi * wn / wd * np.sin(wd * ta))
    thd[a] = -(wn**2 / wd) * th0 * env * np.sin(wd * ta)
    b = ~a
    tb = t[b] - t_imp
    env = np.exp(-xi * wn * tb)
    th[b] = env * thd_p / wd * np.sin(wd * tb)
    thd[b] = thd_p * env * (np.cos(wd * tb) - xi * wn / wd * np.sin(wd * tb))
    x0[b] = x0d_p * tb  # m0 desliza sin rozamiento tras el impacto
    return th, thd, x0


# =============================================================================
# Cabecera y resultados
# =============================================================================
st.title("Ensayo de impactos: vibración del sistema polea-barra")
st.caption("Mecánica de Máquinas y Vibraciones · Modelo lineal de 1 GDL en el giro de la polea θ(t)")

st.subheader("Resultados de los apartados")
filas = [
    ("1. Ecuación del movimiento", "Inercia equivalente I", f"{I:.4f}", "kg·m²"),
    ("1. Ecuación del movimiento", "Amortiguamiento cθ", f"{ct:.4f}", "N·m·s/rad"),
    ("1. Ecuación del movimiento", "Rigidez kθ", f"{kt:.3f}", "N·m/rad"),
    ("1. Ecuación del movimiento", "ξ  ·  ωn  ·  ωd", f"{xi:.4f}  ·  {wn:.3f}  ·  {wd:.3f}", "–  ·  rad/s  ·  rad/s"),
    ("2. Par estático", "M0", f"{M0:.3f}", "N·m"),
    ("3. Precarga", "Δl mínima", f"{dl_min*1000:.2f}", "mm"),
    ("4. Antes del impacto", "Instante de impacto t*", f"{t_imp:.4f}", "s"),
    ("4. Antes del impacto", "Velocidad angular de impacto θ'⁻", f"{thd_m:.4f}", "rad/s"),
    ("5. Después del impacto", "Velocidad angular θ'⁺", f"{thd_p:.5f}", "rad/s"),
    ("5. Después del impacto", "Velocidad de m0 ẋ0⁺", f"{x0d_p:.4f}", "m/s"),
    ("5. Después del impacto", "Ángulo máximo |θ|máx",
     f"{abs(th_max):.5f}  ({np.degrees(abs(th_max)):.3f}°)", "rad"),
]
st.dataframe(pd.DataFrame(filas, columns=["Apartado", "Magnitud", "Valor", "Unidad"]),
             hide_index=True, use_container_width=True, height=36 * (len(filas) + 1) + 3)

if thd_p < 0:
    st.info("Con estos parámetros I < e·m0·L²: la barra **rebota** (θ'⁺ < 0) y el máximo tras el impacto "
            "se alcanza en sentido horario.")

with st.expander("Detalle de kθ, impulso y energía"):
    st.markdown(
        f"- kθ = 2k₁R² + k₂(2L/3)² + m_b·g·(2L/5) = **{k_cab:.3f}** + **{k_k2:.3f}** + **{k_grav:.3f}** "
        f"= **{kt:.3f} N·m/rad**\n"
        f"- Decremento logarítmico δ = ln(1/(1−{red/100:.3f}))/{ciclos} = **{delta:.4f}**\n"
        f"- Amortiguador equivalente de cada rama: c₁ = cθ/(2R²) = **{c1:.3f} N·s/m**\n"
        f"- Impulso de percusión sobre m0: I_c = m0·ẋ0⁺ = **{Ic:.4f} N·s**\n"
        f"- Energía cinética antes del impacto: **{E_antes*1000:.3f} mJ** · después: **{E_desp*1000:.3f} mJ** "
        f"· disipada: **{(E_antes-E_desp)*1000:.3f} mJ** ({(1-E_desp/E_antes)*100 if E_antes else 0:.1f} %)\n"
        f"- Instante del máximo tras el impacto: t = t* + **{t_pk:.4f} s**"
    )

# =============================================================================
# Animación
# =============================================================================
st.subheader("Animación del ensayo")
a1, a2, a3 = st.columns(3)
T_sim = a1.slider("Tiempo simulado [s]", 0.5, 10.0, 2.0, 0.5)
amp = a2.slider("Amplificación visual de los desplazamientos", 1.0, 30.0, 1.0, 0.5,
                help="Los giros tras el impacto son muy pequeños; amplifícalos para verlos. "
                     "Afecta solo al dibujo del mecanismo, no a la gráfica de θ(t).")
lento = a3.slider("Cámara lenta (×)", 1, 20, 5)

n_frames = 160
t_fr = np.linspace(0, T_sim, n_frames)
th_fr, _, x0_fr = respuesta(t_fr)

t_pl = np.linspace(0, T_sim, 2000)
th_pl, thd_pl, x0_pl = respuesta(t_pl)

# --- geometría fija del dibujo ---
XW = -0.9 * max(L, 0.6)          # pared de los muelles del cable
XA = XW + 0.55 * max(L, 0.6)     # posición en reposo del extremo del cable
XW2 = 0.45 * L                   # pared del muelle k2
r0 = 0.05 * L                    # radio de la masa m0
YC = -2 * L / 3
off = 0.035 * max(L, 0.6)        # separación muelle/amortiguador


def zigzag(x1, x2, y, n=8, a=0.018):
    xs = np.linspace(x1, x2, 2 * n + 3)
    ys = np.full_like(xs, y)
    ys[1:-1:2] += a
    ys[2:-1:2] -= a
    ys[1], ys[-2] = y, y
    return xs, ys


def rama(y, sgn, phi):
    """Muelle + amortiguador + conexión + cable de una rama. sgn=+1 superior, -1 inferior."""
    xa = XA - sgn * R * phi  # superior: se acorta con φ>0 ; inferior: se alarga
    X, Y = [], []
    zx, zy = zigzag(XW, xa, y - off, a=0.012 * max(L, 0.6) / 0.6)
    X += list(zx) + [None]; Y += list(zy) + [None]
    xp = xa - 0.22 * (XA - XW)  # pistón
    X += [xa, xp, None, xp, xp, None]; Y += [y + off, y + off, None, y + off - 0.015, y + off + 0.015, None]
    X += [xa, xa, None, xa, 0, None]; Y += [y - off, y + off, None, y, y, None]
    return X, Y


def mecanismo(phi, x0):
    s, cth = np.sin(phi), np.cos(phi)
    tip = (L * s, -L * cth)
    G = (2 * L / 5 * s, -2 * L / 5 * cth)
    C = (2 * L / 3 * s, -2 * L / 3 * cth)
    # radios de la polea
    ang = phi + np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2]) + np.pi / 4
    spx, spy = [], []
    for a_ in ang:
        spx += [0, 0.9 * R * np.cos(a_), None]
        spy += [0, 0.9 * R * np.sin(a_), None]
    kx, ky = zigzag(C[0], XW2, C[1], n=7, a=0.02 * L)
    x1, y1 = rama(R, +1, phi)
    x2, y2 = rama(-R, -1, phi)
    cc = np.linspace(0, 2 * np.pi, 30)
    mx, my = r0 + x0 + r0 * np.cos(cc), -L + r0 * np.sin(cc) - 0.0
    return [
        dict(x=spx, y=spy),
        dict(x=[0, tip[0]], y=[0, tip[1]]),
        dict(x=[G[0], C[0]], y=[G[1], C[1]]),
        dict(x=list(kx), y=list(ky)),
        dict(x=list(mx), y=list(my)),
        dict(x=x1 + x2, y=y1 + y2),
        dict(x=[t_fr_actual[0]], y=[t_fr_actual[1]]),
    ]


fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.08,
                    subplot_titles=("Mecanismo", "Giro de la polea θ(t)"))

# --- elementos estáticos ---
cc = np.linspace(0, 2 * np.pi, 60)
fig.add_trace(go.Scatter(x=R * np.cos(cc), y=R * np.sin(cc), fill="toself", fillcolor="#bdbdbd",
                         line=dict(color="#555"), hoverinfo="skip", showlegend=False), 1, 1)
hy = np.linspace(-R - 2.5 * off, R + 2.5 * off, 12)
fig.add_trace(go.Scatter(x=sum([[XW, XW - 0.04, None] for _ in hy], []), y=sum([[v, v - 0.04, None] for v in hy], []),
                         line=dict(color="#777", width=1), hoverinfo="skip", showlegend=False), 1, 1)
cyl_x, cyl_y = [], []
for y in (R, -R):
    x_end = XA - 0.12 * (XA - XW)
    cyl_x += [XW, x_end, None, XW, x_end, None, x_end, x_end, None]
    cyl_y += [y + off + 0.02, y + off + 0.02, None, y + off - 0.02, y + off - 0.02, None,
              y + off + 0.02, y + off + 0.008, None]
fig.add_trace(go.Scatter(x=[XW, XW] + [None] + cyl_x, y=[-R - 2.5 * off, R + 2.5 * off] + [None] + cyl_y,
                         line=dict(color="#333", width=2), hoverinfo="skip", showlegend=False), 1, 1)
fig.add_trace(go.Scatter(x=[XW2, XW2, None] + sum([[XW2, XW2 + 0.04, None] for _ in range(6)], []),
                         y=[YC - 0.08 * L, YC + 0.08 * L, None] +
                           sum([[YC - 0.08 * L + i * 0.032 * L, YC - 0.08 * L + i * 0.032 * L - 0.04, None]
                                for i in range(6)], []),
                         line=dict(color="#333", width=2), hoverinfo="skip", showlegend=False), 1, 1)
yg = -L - r0
fig.add_trace(go.Scatter(x=[-0.35 * L, 1.6 * L], y=[yg, yg], line=dict(color="#333", width=2),
                         hoverinfo="skip", showlegend=False), 1, 1)
fig.add_trace(go.Scatter(x=[0, 0], y=[0, -L], line=dict(color="#bbb", dash="dot"),
                         hoverinfo="skip", showlegend=False), 1, 1)

# --- curva θ(t) estática ---
fig.add_trace(go.Scatter(x=t_pl, y=np.degrees(th_pl), line=dict(color="#1f77b4", width=2),
                         name="θ(t)", showlegend=False), 1, 2)
fig.add_vline(x=t_imp, line=dict(color="#d62728", dash="dash"), row=1, col=2,
              annotation_text="impacto", annotation_position="top right")

n_static = len(fig.data)

# --- elementos animados ---
t_fr_actual = (0.0, np.degrees(th_fr[0]))
d0 = mecanismo(amp * th_fr[0], amp * x0_fr[0])
estilos = [
    dict(mode="lines", line=dict(color="#333", width=3)),                       # radios polea
    dict(mode="lines", line=dict(color="#1f4e79", width=8)),                    # barra
    dict(mode="markers+text", marker=dict(size=8, color=["#000", "#d62728"]),
         text=["G", "C"], textposition="middle left"),                            # G y C
    dict(mode="lines", line=dict(color="#2ca02c", width=2)),                    # muelle k2
    dict(mode="lines", fill="toself", fillcolor="#ff7f0e", line=dict(color="#a65000")),  # m0
    dict(mode="lines", line=dict(color="#555", width=2)),                       # ramas del cable
    dict(mode="markers", marker=dict(size=11, color="#d62728")),                # marcador θ(t)
]
for i, (d, s) in enumerate(zip(d0, estilos)):
    col = 2 if i == 6 else 1
    fig.add_trace(go.Scatter(**d, **s, hoverinfo="skip", showlegend=False), 1, col)

idx_anim = list(range(n_static, n_static + 7))
frames = []
for i, t in enumerate(t_fr):
    t_fr_actual = (t, np.degrees(th_fr[i]))
    frames.append(go.Frame(data=[go.Scatter(**d) for d in mecanismo(amp * th_fr[i], amp * x0_fr[i])],
                           traces=idx_anim, name=str(i)))
fig.frames = frames

dur = max(int(T_sim / n_frames * lento * 1000), 10)
fig.update_xaxes(range=[XW - 0.08, 1.15 * L], showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
fig.update_yaxes(range=[-L - 0.2 * L, R + 0.25 * L], showgrid=False, zeroline=False, showticklabels=False,
                 scaleanchor="x", scaleratio=1, row=1, col=1)
fig.update_xaxes(title_text="t [s]", row=1, col=2)
fig.update_yaxes(title_text="θ [°]", row=1, col=2)
fig.update_layout(
    height=600, margin=dict(l=20, r=20, t=50, b=40), plot_bgcolor="white",
    updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.08, xanchor="left",
                      buttons=[
                          dict(label="▶ Reproducir", method="animate",
                               args=[None, dict(frame=dict(duration=dur, redraw=False),
                                                transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                          dict(label="⏸ Pausa", method="animate",
                               args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                      ])],
    sliders=[dict(active=0, x=0.2, len=0.8, y=-0.04, currentvalue=dict(prefix="t = ", suffix=" s"),
                  steps=[dict(method="animate", label=f"{t:.2f}",
                              args=[[str(i)], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                         for i, t in enumerate(t_fr)])],
)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"El giro positivo θ es antihorario (la punta se mueve hacia la derecha). La masa m0 desliza sin rozamiento "
           f"tras el impacto. Desplazamientos del dibujo amplificados ×{amp:g}.")

# =============================================================================
# Gráficas adicionales
# =============================================================================
st.subheader("Respuesta y tensión en los cables")
g1, g2 = st.columns(2)

with g1:
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=t_pl, y=thd_pl, name="θ'(t) [rad/s]"))
    f2.add_trace(go.Scatter(x=t_pl, y=np.where(t_pl >= t_imp, x0d_p, 0) / L, name="ẋ0 / L [1/s]",
                            line=dict(dash="dot")))
    f2.add_vline(x=t_imp, line=dict(color="#d62728", dash="dash"))
    f2.update_layout(title="Velocidad angular de la polea", xaxis_title="t [s]", height=380,
                     legend=dict(orientation="h", y=-0.25), margin=dict(t=50, b=10))
    st.plotly_chart(f2, use_container_width=True)

with g2:
    usar_min = st.checkbox("Usar Δl mínima del apartado 3", value=True)
    dl = dl_min if usar_min else st.number_input("Δl [mm]", 0.0, 500.0, round(dl_min * 1000, 2), 0.5) / 1000
    incl_c = st.checkbox("Incluir la fuerza de los amortiguadores en la tensión", value=False,
                         help="La solución del examen considera solo los muelles. Con amortiguadores, "
                              "el salto de velocidad en el impacto modifica la tensión.")
    cc_ = c1 if incl_c else 0.0
    T_sup = k1 * (dl - R * th_pl) - cc_ * R * thd_pl
    T_inf = k1 * (dl + R * th_pl) + cc_ * R * thd_pl
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=t_pl, y=T_sup, name="Rama superior"))
    f3.add_trace(go.Scatter(x=t_pl, y=T_inf, name="Rama inferior"))
    f3.add_hline(y=0, line=dict(color="black", width=1))
    f3.add_vline(x=t_imp, line=dict(color="#d62728", dash="dash"))
    f3.update_layout(title="Tensión en el cable", xaxis_title="t [s]", yaxis_title="T [N]", height=380,
                     legend=dict(orientation="h", y=-0.25), margin=dict(t=50, b=10))
    st.plotly_chart(f3, use_container_width=True)
    Tmin = min(T_sup.min(), T_inf.min())
    if Tmin < -1e-9:
        st.error(f"El cable pierde tensión (T mín = {Tmin:.3f} N). Aumenta Δl.")
    else:
        st.success(f"El cable se mantiene tenso (T mín = {Tmin:.3f} N).")

# =============================================================================
# Teoría
# =============================================================================
with st.expander("Fundamento teórico"):
    st.markdown(r"""
**Ecuación del movimiento (Lagrange, linealizada)**

$$I\ddot\theta + c_\theta\dot\theta + k_\theta\theta = 0,\qquad
I = I_{Gp} + m_b\left(\tfrac{2L}{5}\right)^2 + I_{Gb},\qquad
k_\theta = 2k_1R^2 + k_2\left(\tfrac{2L}{3}\right)^2 + m_b g\tfrac{2L}{5}$$

**Amortiguamiento** (reducción $p$ de la amplitud en $N$ ciclos):
$\delta = \frac{1}{N}\ln\frac{1}{1-p}$, $\;\xi = \dfrac{\delta}{\sqrt{4\pi^2+\delta^2}}$, $\;c_\theta = 2\xi\sqrt{I k_\theta}$

**Par estático y precarga:** $M_0 = k_\theta|\theta_0|$, $\quad \Delta l \ge R|\theta_0|$

**Antes del impacto** (desde el reposo en $\theta_0$):
$\theta(t) = e^{-\xi\omega_n t}\theta_0\left(\cos\omega_d t + \frac{\xi\omega_n}{\omega_d}\sin\omega_d t\right)$,
impacto en $\omega_d t^* = \pi - \arctan\frac{\omega_d}{\xi\omega_n}$, con
$\dot\theta^- = -\frac{\omega_n^2}{\omega_d}\theta_0 e^{-\xi\omega_n t^*}\sin\omega_d t^*$

**Impacto** (percusión $I_c$ en la punta, coeficiente de restitución $e$):
$I(\dot\theta^+-\dot\theta^-) = -I_c L$, $\; I_c = m_0\dot x_0^+$, $\; \dot x_0^+ - L\dot\theta^+ = eL\dot\theta^-$
$$\Rightarrow\quad \dot\theta^+ = \frac{I - e\,m_0L^2}{I + m_0L^2}\,\dot\theta^-$$

**Después del impacto:** $\theta(t) = e^{-\xi\omega_n t}\frac{\dot\theta^+}{\omega_d}\sin\omega_d t$,
máximo en $\omega_d t = \arctan\frac{\omega_d}{\xi\omega_n}$.
""")
