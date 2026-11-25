"""
Vibración libre de una viga biapoyada (teoría de Euler-Bernoulli)
-----------------------------------------------------------------
Ecuación:  EI · ∂⁴w/∂x⁴ + ρA · ∂²w/∂t² = 0
Condiciones de contorno (biapoyada): w(0)=w(L)=0, w''(0)=w''(L)=0

Solución modal:
    φ_n(x) = sin(nπx/L)
    ω_n    = (nπ/L)² · sqrt(EI / ρA)       [rad/s]
    f_n    = ω_n / 2π                      [Hz]
    w(x,t) = Σ A_n · φ_n(x) · cos(ω_n t)

Ejecutar:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Vibración de viga biapoyada", page_icon="〰️", layout="wide")

# ---------------------------------------------------------------------------
# Datos de materiales y secciones
# ---------------------------------------------------------------------------
MATERIALES = {
    "Acero (S275)": {"E": 210e9, "rho": 7850.0},
    "Aluminio": {"E": 70e9, "rho": 2700.0},
    "Hormigón (C30)": {"E": 33e9, "rho": 2500.0},
    "Madera (C24)": {"E": 11e9, "rho": 420.0},
    "Personalizado": {"E": 200e9, "rho": 7800.0},
}


def propiedades_seccion(tipo, p):
    """Devuelve (A [m²], I [m⁴]) para la sección elegida."""
    if tipo == "Rectangular":
        b, h = p["b"], p["h"]
        return b * h, b * h**3 / 12
    if tipo == "Circular maciza":
        d = p["d"]
        return np.pi * d**2 / 4, np.pi * d**4 / 64
    if tipo == "Tubo circular":
        D, t = p["D"], p["t"]
        d = max(D - 2 * t, 0.0)
        return np.pi * (D**2 - d**2) / 4, np.pi * (D**4 - d**4) / 64
    # Introducir A e I directamente
    return p["A"], p["I"]


def frecuencias(n, L, E, I, rho, A):
    omega = (n * np.pi / L) ** 2 * np.sqrt(E * I / (rho * A))
    return omega, omega / (2 * np.pi)


# ---------------------------------------------------------------------------
# Barra lateral: parámetros
# ---------------------------------------------------------------------------
st.sidebar.header("Parámetros de la viga")

L = st.sidebar.number_input("Longitud L [m]", 0.1, 100.0, 5.0, 0.1)

mat = st.sidebar.selectbox("Material", list(MATERIALES))
if mat == "Personalizado":
    E = st.sidebar.number_input("Módulo de Young E [GPa]", 0.1, 1000.0, 200.0) * 1e9
    rho = st.sidebar.number_input("Densidad ρ [kg/m³]", 1.0, 30000.0, 7800.0)
else:
    E, rho = MATERIALES[mat]["E"], MATERIALES[mat]["rho"]
    st.sidebar.caption(f"E = {E/1e9:.0f} GPa · ρ = {rho:.0f} kg/m³")

tipo_sec = st.sidebar.selectbox(
    "Sección", ["Rectangular", "Circular maciza", "Tubo circular", "Introducir A e I"]
)
p = {}
if tipo_sec == "Rectangular":
    p["b"] = st.sidebar.number_input("Ancho b [mm]", 1.0, 5000.0, 100.0) / 1000
    p["h"] = st.sidebar.number_input("Canto h [mm]", 1.0, 5000.0, 200.0) / 1000
elif tipo_sec == "Circular maciza":
    p["d"] = st.sidebar.number_input("Diámetro d [mm]", 1.0, 5000.0, 100.0) / 1000
elif tipo_sec == "Tubo circular":
    p["D"] = st.sidebar.number_input("Diámetro exterior D [mm]", 1.0, 5000.0, 168.3) / 1000
    p["t"] = st.sidebar.number_input("Espesor t [mm]", 0.1, 500.0, 8.0) / 1000
else:
    p["A"] = st.sidebar.number_input("Área A [cm²]", 0.01, 1e5, 28.5, format="%.2f") / 1e4
    p["I"] = st.sidebar.number_input("Inercia I [cm⁴]", 0.01, 1e8, 1940.0, format="%.2f") / 1e8

A, I = propiedades_seccion(tipo_sec, p)
st.sidebar.caption(f"A = {A*1e4:.2f} cm² · I = {I*1e8:.1f} cm⁴ · m = ρA = {rho*A:.2f} kg/m")

# ---------------------------------------------------------------------------
# Cabecera y selección de modo
# ---------------------------------------------------------------------------
st.title("Vibración libre de una viga biapoyada")
st.markdown(
    r"Teoría de **Euler-Bernoulli**: "
    r"$\;\phi_n(x)=\sin\!\left(\dfrac{n\pi x}{L}\right),\quad "
    r"\omega_n=\left(\dfrac{n\pi}{L}\right)^2\sqrt{\dfrac{EI}{\rho A}}$"
)

c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    vista = st.radio("Visualización", ["Modo único", "Superposición de modos"], horizontal=True)
with c3:
    amp_rel = st.slider("Amplitud visual (% de L)", 1, 20, 8)

if vista == "Modo único":
    with c2:
        n = st.select_slider("Modo de vibración n", options=list(range(1, 11)), value=1)
    modos = {n: 1.0}
else:
    with c2:
        sel = st.multiselect("Modos a combinar", list(range(1, 11)), default=[1, 2, 3])
    if not sel:
        st.warning("Selecciona al menos un modo.")
        st.stop()
    cols = st.columns(len(sel))
    modos = {}
    for col, m in zip(cols, sorted(sel)):
        modos[m] = col.slider(f"A{m}", -1.0, 1.0, round(1.0 / m, 2), 0.05, key=f"amp{m}")

# ---------------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------------
x = np.linspace(0, L, 301)
omega = {m: frecuencias(m, L, E, I, rho, A)[0] for m in modos}
f = {m: w / (2 * np.pi) for m, w in omega.items()}

# Duración de la animación: un periodo del modo más bajo implicado
w_min = min(omega.values())
T_ref = 2 * np.pi / w_min
n_frames = 60
t_vals = np.linspace(0, T_ref, n_frames, endpoint=False)


def deformada(t):
    return sum(a * np.sin(m * np.pi * x / L) * np.cos(omega[m] * t) for m, a in modos.items())


# Escalado visual
max_abs = max(np.max(np.abs(deformada(t))) for t in t_vals) or 1.0
escala = (amp_rel / 100) * L / max_abs

# Métricas
if vista == "Modo único":
    m0 = next(iter(modos))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Frecuencia natural fₙ", f"{f[m0]:.3f} Hz")
    k2.metric("Frecuencia angular ωₙ", f"{omega[m0]:.2f} rad/s")
    k3.metric("Periodo Tₙ", f"{1/f[m0]*1000:.2f} ms" if 1 / f[m0] < 1 else f"{1/f[m0]:.3f} s")
    k4.metric("Nodos interiores", f"{m0 - 1}")

# ---------------------------------------------------------------------------
# Gráfico animado (Plotly, animación en el navegador)
# ---------------------------------------------------------------------------
y_lim = (amp_rel / 100) * L * 1.35
w0 = deformada(0) * escala

# Envolvente (máximo desplazamiento posible)
env = np.max(np.abs(np.array([deformada(t) for t in t_vals])), axis=0) * escala

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=env, mode="lines", line=dict(color="rgba(120,120,120,0.35)", dash="dot"),
                         name="Envolvente", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=x, y=-env, mode="lines", line=dict(color="rgba(120,120,120,0.35)", dash="dot"),
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=[0, L], y=[0, 0], mode="lines", line=dict(color="lightgray", width=2),
                         name="Posición de reposo", hoverinfo="skip"))
# Traza animada (índice 3)
fig.add_trace(go.Scatter(x=x, y=w0, mode="lines", line=dict(color="#1f77b4", width=5),
                         name="Deformada w(x,t)"))

# Nodos (solo modo único)
if vista == "Modo único":
    m0 = next(iter(modos))
    xn = [k * L / m0 for k in range(1, m0)]
    if xn:
        fig.add_trace(go.Scatter(x=xn, y=[0] * len(xn), mode="markers",
                                 marker=dict(size=10, color="#d62728", symbol="circle-open", line=dict(width=2)),
                                 name="Nodos"))

# Apoyos: articulado (izq.) y deslizante (dcha.) dibujados como triángulos
tri = 0.06 * y_lim / 0.35 if y_lim else 0.05
for xa, label in [(0, "Apoyo fijo"), (L, "Apoyo móvil")]:
    fig.add_trace(go.Scatter(
        x=[xa, xa - 0.025 * L, xa + 0.025 * L, xa], y=[0, -0.18 * y_lim, -0.18 * y_lim, 0],
        fill="toself", mode="lines", line=dict(color="#444"), fillcolor="#888",
        showlegend=False, hoverinfo="text", text=label))
fig.add_shape(type="line", x0=L - 0.035 * L, x1=L + 0.035 * L, y0=-0.23 * y_lim, y1=-0.23 * y_lim,
              line=dict(color="#444", width=2))

frames = [go.Frame(data=[go.Scatter(x=x, y=deformada(t) * escala)], traces=[3], name=f"{i}")
          for i, t in enumerate(t_vals)]
fig.frames = frames

dur = st.session_state.get("vel", 40)
fig.update_layout(
    height=480,
    xaxis=dict(title="x [m]", range=[-0.05 * L, 1.05 * L], zeroline=False),
    yaxis=dict(title="w (escalada)", range=[-y_lim, y_lim], showticklabels=False, zeroline=False),
    legend=dict(orientation="h", y=1.12),
    margin=dict(l=40, r=20, t=60, b=40),
    updatemenus=[dict(
        type="buttons", direction="left", x=0, y=-0.15, xanchor="left",
        buttons=[
            dict(label="▶ Reproducir", method="animate",
                 args=[None, dict(frame=dict(duration=dur, redraw=False), transition=dict(duration=0),
                                  fromcurrent=True, mode="immediate")]),
            dict(label="⏸ Pausa", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
        ],
    )],
    sliders=[dict(
        active=0, x=0.25, len=0.75, y=-0.12, currentvalue=dict(prefix="t / T = "),
        steps=[dict(method="animate", label=f"{i/n_frames:.2f}",
                    args=[[f"{i}"], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
               for i in range(n_frames)],
    )],
)

st.plotly_chart(fig, use_container_width=True)
st.slider("Velocidad de animación (ms por fotograma; menor = más rápido)", 10, 150, 40, key="vel")
st.caption(
    "La animación está ralentizada para poder verse: un ciclo completo corresponde a un periodo "
    f"real de {T_ref:.4g} s del modo más bajo mostrado."
)

# ---------------------------------------------------------------------------
# Formas modales y tabla de frecuencias
# ---------------------------------------------------------------------------
st.subheader("Primeros modos de vibración")
izq, der = st.columns([3, 2])

with izq:
    nmax = 6
    fig2 = go.Figure()
    for m in range(1, nmax + 1):
        fig2.add_trace(go.Scatter(x=x, y=np.sin(m * np.pi * x / L) - 2.5 * (m - 1), mode="lines",
                                  name=f"Modo {m}",
                                  line=dict(width=4 if m in modos else 2)))
        fig2.add_shape(type="line", x0=0, x1=L, y0=-2.5 * (m - 1), y1=-2.5 * (m - 1),
                       line=dict(color="lightgray", width=1))
    fig2.update_layout(height=520, showlegend=True, margin=dict(l=40, r=20, t=20, b=40),
                       xaxis_title="x [m]",
                       yaxis=dict(showticklabels=False, zeroline=False))
    st.plotly_chart(fig2, use_container_width=True)

with der:
    ns = np.arange(1, 11)
    w_all, f_all = frecuencias(ns, L, E, I, rho, A)
    df = pd.DataFrame({
        "Modo n": ns,
        "ωₙ [rad/s]": w_all.round(2),
        "fₙ [Hz]": f_all.round(3),
        "Tₙ [s]": (1 / f_all).round(5),
        "fₙ / f₁": (f_all / f_all[0]).round(1),
    })
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.info("En una viga biapoyada fₙ crece con n²: f₂ = 4·f₁, f₃ = 9·f₁, …")

with st.expander("Fundamento teórico"):
    st.markdown(r"""
**Ecuación del movimiento (Euler-Bernoulli, sin amortiguamiento):**

$$EI\,\frac{\partial^4 w}{\partial x^4} + \rho A\,\frac{\partial^2 w}{\partial t^2} = 0$$

**Separación de variables** $w(x,t)=\phi(x)\,q(t)$ y condiciones de contorno de viga biapoyada
$w(0)=w(L)=0$, $\;M(0)=M(L)=0 \Rightarrow w''(0)=w''(L)=0$, lo que conduce a la ecuación característica
$\sin(\beta L)=0 \Rightarrow \beta_n L = n\pi$.

**Modos y frecuencias:**

$$\phi_n(x)=\sin\frac{n\pi x}{L},\qquad
\omega_n=\beta_n^2\sqrt{\frac{EI}{\rho A}}=\left(\frac{n\pi}{L}\right)^2\sqrt{\frac{EI}{\rho A}},\qquad
f_n=\frac{\omega_n}{2\pi}$$

**Respuesta libre** (partiendo del reposo con deformada inicial): $w(x,t)=\sum_n A_n\,\phi_n(x)\cos(\omega_n t)$.

El modo $n$ tiene $n-1$ nodos interiores situados en $x_k = kL/n$.
Hipótesis: material elástico lineal, pequeñas deformaciones, se desprecian la deformación por cortante y la inercia rotacional
(válido para vigas esbeltas, $L/h \gtrsim 10$, y modos bajos).
""")
