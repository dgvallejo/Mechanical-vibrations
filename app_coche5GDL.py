"""
Modos de vibración de un modelo de medio vehículo pesado de tres ejes (5 GDL)
Problema 3.8 – Respuesta de sistemas de varios GDL con amortiguamiento no proporcional

Coordenadas:  x = [x_a, x_b, x_d, x_t, x_m]
    x_a  desplazamiento del anclaje de la suspensión delantera en el bastidor
    x_b  desplazamiento del anclaje de la suspensión trasera en el bastidor
    x_d  masa semisuspendida delantera
    x_t  masa semisuspendida trasera
    x_m  masa semisuspendida central

Modos no amortiguados:  (k − ω² m) φ = 0
Modos amortiguados (complejos, amortiguamiento no proporcional): espacio de estados 2N.

Ejecutar:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.linalg as sl
import streamlit as st

st.set_page_config(page_title="Modos de vibración – vehículo de 3 ejes", page_icon="🚛", layout="wide")

# =============================================================================
# Parámetros (Tabla 3.1)
# =============================================================================
DEF = dict(
    m=22000.0, IG=21000.0, mssd=900.0, mssm=1400.0, msst=1400.0,
    ksd=610000.0, ksm=2600000.0, kst=2600000.0, csd=15400.0, csm=15400.0, cst=15400.0,
    krd=1360000.0, krm=5430000.0, krt=5430000.0, crd=150.0, crm=150.0, crt=150.0,
    ld=4.44, lt=1.71, la=4.80, lb=1.35,
)
for _k, _v in DEF.items():
    st.session_state.setdefault(_k, _v)

sb = st.sidebar
sb.header("Propiedades del modelo")
if sb.button("↺ Restaurar valores de la Tabla 3.1", use_container_width=True):
    for _k, _v in DEF.items():
        st.session_state[_k] = _v


def num(label, key, step, fmt="%.0f", container=sb):
    return container.number_input(label, min_value=0.0, step=step, format=fmt, key=key)


with sb.expander("Masas e inercia", expanded=False):
    m = num("m – masa del bastidor [kg]", "m", 500.0, container=st)
    IG = num("I_G – inercia de cabeceo [kg·m²]", "IG", 500.0, container=st)
    mssd = num("m_ssd – eje delantero [kg]", "mssd", 50.0, container=st)
    mssm = num("m_ssm – eje central [kg]", "mssm", 50.0, container=st)
    msst = num("m_sst – eje trasero [kg]", "msst", 50.0, container=st)

with sb.expander("Suspensiones", expanded=False):
    ksd = num("k_sd [N/m]", "ksd", 10000.0, container=st)
    ksm = num("k_sm [N/m]", "ksm", 10000.0, container=st)
    kst = num("k_st [N/m]", "kst", 10000.0, container=st)
    csd = num("c_sd [N·s/m]", "csd", 500.0, container=st)
    csm = num("c_sm [N·s/m]", "csm", 500.0, container=st)
    cst = num("c_st [N·s/m]", "cst", 500.0, container=st)

with sb.expander("Neumáticos", expanded=False):
    krd = num("k_rd [N/m]", "krd", 10000.0, container=st)
    krm = num("k_rm [N/m]", "krm", 10000.0, container=st)
    krt = num("k_rt [N/m]", "krt", 10000.0, container=st)
    crd = num("c_rd [N·s/m]", "crd", 10.0, container=st)
    crm = num("c_rm [N·s/m]", "crm", 10.0, container=st)
    crt = num("c_rt [N·s/m]", "crt", 10.0, container=st)

with sb.expander("Geometría", expanded=False):
    ld = num("l_d – cdg ↔ eje delantero [m]", "ld", 0.05, "%.2f", container=st)
    lt = num("l_t – cdg ↔ eje trasero [m]", "lt", 0.05, "%.2f", container=st)
    la = num("l_a – eje delantero ↔ central [m]", "la", 0.05, "%.2f", container=st)
    lb = num("l_b – eje central ↔ trasero [m]", "lb", 0.05, "%.2f", container=st)
    l = ld + lt
    st.caption(f"l = l_d + l_t = {l:.2f} m")
    if abs((la + lb) - l) > 1e-6:
        st.warning(f"l_a + l_b = {la+lb:.2f} m ≠ l = {l:.2f} m. Se usa l = l_d + l_t.")

if l <= 0 or min(m, mssd, mssm, msst) <= 0:
    st.error("Las masas y la batalla deben ser positivas.")
    st.stop()


# =============================================================================
# Matrices (ec. 3.167 – 3.169)
# =============================================================================
def matriz_rig(sd, sm, st_, rd, rm, rt):
    """Misma estructura para k y c."""
    K = np.zeros((5, 5))
    K[0, 0] = sd + sm * lb**2 / l**2
    K[1, 1] = st_ + sm * la**2 / l**2
    K[0, 1] = K[1, 0] = sm * la * lb / l**2
    K[0, 2] = K[2, 0] = -sd
    K[0, 4] = K[4, 0] = -sm * lb / l
    K[1, 3] = K[3, 1] = -st_
    K[1, 4] = K[4, 1] = -sm * la / l
    K[2, 2] = rd + sd
    K[3, 3] = rt + st_
    K[4, 4] = rm + sm
    return K


M = np.zeros((5, 5))
M[0, 0] = (m * lt**2 + IG) / l**2
M[1, 1] = (m * ld**2 + IG) / l**2
M[0, 1] = M[1, 0] = -(IG - ld * lt * m) / l**2
M[2, 2], M[3, 3], M[4, 4] = mssd, msst, mssm
K = matriz_rig(ksd, ksm, kst, krd, krm, krt)
C = matriz_rig(csd, csm, cst, crd, crm, crt)

try:
    w2, Phi = sl.eigh(K, M)
except np.linalg.LinAlgError:
    st.error("La matriz de masa no es definida positiva con estos parámetros.")
    st.stop()
wn = np.sqrt(np.clip(w2, 0, None))

# Modos complejos (espacio de estados)
Minv = np.linalg.inv(M)
A = np.block([[np.zeros((5, 5)), np.eye(5)], [-Minv @ K, -Minv @ C]])
lam, Vs = np.linalg.eig(A)
sel = lam.imag > 1e-9
lam_c, Psi = lam[sel], Vs[:5, sel]
orden = np.argsort(np.abs(lam_c))
lam_c, Psi = lam_c[orden], Psi[:, orden]

NOMBRES = ["x_a", "x_b", "x_d", "x_t", "x_m"]
ETIQ = ["x_a (bastidor, delante)", "x_b (bastidor, detrás)", "x_d (eje delantero)",
        "x_t (eje trasero)", "x_m (eje central)"]


def normaliza(v):
    v = np.asarray(v, dtype=complex)
    i = np.argmax(np.abs(v))
    return v / v[i]


def energia_cinetica(v):
    """Reparto de energía cinética (máx.) entre bastidor y ejes, en %."""
    v = np.real(v)
    xa, xb = v[0], v[1]
    xg = (lt * xa + ld * xb) / l
    th = (xa - xb) / l
    e = np.array([m * xg**2, IG * th**2, mssd * v[2]**2, msst * v[3]**2, mssm * v[4]**2])
    return 100 * e / e.sum() if e.sum() > 0 else e


def describe(v):
    vr = np.real(v)
    e = energia_cinetica(vr)
    if e[0] + e[1] > 50:
        return "Bote del bastidor (traslación vertical dominante)" if e[0] >= e[1] else \
               "Cabeceo del bastidor (giro dominante)"
    ejes = {"delantero": e[2], "trasero": e[3], "central": e[4]}
    if e[3] > 15 and e[4] > 15:
        fase = "en fase" if vr[3] * vr[4] > 0 else "en contrafase"
        return f"Salto de los ejes trasero y central {fase}"
    return f"Salto del eje {max(ejes, key=ejes.get)}"


# =============================================================================
# Interfaz principal
# =============================================================================
st.title("Modos de vibración de un vehículo pesado de tres ejes")
st.caption("Modelo de medio coche de 5 GDL (Problema 3.8) · coordenadas x = [x_a, x_b, x_d, x_t, x_m]ᵀ")

c1, c2 = st.columns([3, 2])
with c1:
    modo = st.radio("Modo de vibración", [1, 2, 3, 4, 5], horizontal=True,
                    format_func=lambda i: f"Modo {i} · {wn[i-1]/(2*np.pi):.2f} Hz")
with c2:
    tipo = st.radio("Tipo de modo", ["No amortiguado (real)", "Amortiguado (complejo)"], horizontal=True,
                    help="El amortiguamiento es no proporcional: los modos amortiguados son complejos y "
                         "cada coordenada tiene su propio desfase.")

i = modo - 1
if tipo.startswith("No"):
    phi = normaliza(Phi[:, i])
    w_anim = wn[i]
    xi_i = None
else:
    phi = normaliza(Psi[:, i])
    w_anim = lam_c[i].imag
    xi_i = -lam_c[i].real / abs(lam_c[i])

k1, k2, k3, k4 = st.columns(4)
k1.metric("Frecuencia natural fₙ", f"{wn[i]/(2*np.pi):.3f} Hz")
k2.metric("ωₙ", f"{wn[i]:.2f} rad/s")
if xi_i is None:
    k3.metric("Periodo Tₙ", f"{2*np.pi/wn[i]*1000:.1f} ms")
    k4.metric("Tipo", "Real")
else:
    k3.metric("Frec. amortiguada f_d", f"{w_anim/(2*np.pi):.3f} Hz")
    k4.metric("Factor de amortiguamiento ξ", f"{xi_i*100:.2f} %")
st.markdown(f"**Forma modal:** {describe(phi)}")

a1, a2, a3 = st.columns(3)
amp = a1.slider("Amplitud visual [m]", 0.05, 0.6, 0.25, 0.05)
vel = a2.slider("Duración de un ciclo en pantalla [s]", 0.5, 5.0, 1.5, 0.25,
                help="Todos los modos se muestran a la misma velocidad aparente; la frecuencia real está arriba.")
fantasma = a3.checkbox("Mostrar posición de equilibrio", value=True)

# =============================================================================
# Dibujo del vehículo
# =============================================================================
# Posiciones horizontales (eje trasero en X = 0, delantero en X = l, marcha hacia la derecha)
XT, XM, XD, XG = 0.0, lb, l, lt
Y_RUEDA, Y_EJE, Y_BAST = 0.0, 0.9, 1.9   # cotas de referencia
H_BAST, H_EJE, W_EJE = 0.45, 0.28, 0.6
DX_AM = 0.18                              # separación muelle-amortiguador


def zigzag(x, y1, y2, n=6, a=0.09):
    ys = np.linspace(y1, y2, 2 * n + 3)
    xs = np.full_like(ys, x)
    xs[2:-2:2] += a
    xs[3:-2:2] -= a
    return list(xs) + [None], list(ys) + [None]


def amortiguador(x, y1, y2, w=0.08):
    """y1 inferior, y2 superior. Cilindro unido abajo, vástago unido arriba."""
    ym = 0.5 * (y1 + y2)
    hc = 0.55 * (y2 - y1) if y2 > y1 else 0.1
    yc_top = y1 + hc
    X = [x, x, None, x - w, x - w, x + w, x + w, None, x - 0.7 * w, x + 0.7 * w, None, x, x, None]
    Y = [y1, y1 + 0.08, None, yc_top, y1 + 0.08, y1 + 0.08, yc_top, None, ym, ym, None, ym, y2, None]
    return X, Y


def rect(xc, yc, w, h):
    return [xc - w / 2, xc + w / 2, xc + w / 2, xc - w / 2, xc - w / 2, None], \
           [yc - h / 2, yc - h / 2, yc + h / 2, yc + h / 2, yc - h / 2, None]


def geometria(u):
    """u = desplazamientos reales [xa, xb, xd, xt, xm] (ya escalados)."""
    xa, xb, xd, xt, xm = u
    yb = lambda X: Y_BAST + xb + (xa - xb) * X / l    # cara inferior del bastidor
    ejes = [(XD, Y_EJE + xd), (XM, Y_EJE + xm), (XT, Y_EJE + xt)]

    # Bastidor (polígono deformado como sólido rígido con giro pequeño)
    x0, x1 = -0.5, l + 0.5
    bx = [x0, x1, x1, x0, x0]
    by = [yb(x0), yb(x1), yb(x1) + H_BAST, yb(x0) + H_BAST, yb(x0)]

    mx, my = [], []
    for X, Y in ejes:
        a_, b_ = rect(X, Y, W_EJE, H_EJE)
        mx += a_; my += b_

    sx, sy, dx, dy = [], [], [], []
    for X, Y in ejes:
        # suspensión: de la masa semisuspendida al bastidor
        zx, zy = zigzag(X - DX_AM, Y + H_EJE / 2, yb(X - DX_AM))
        sx += zx; sy += zy
        ax_, ay_ = amortiguador(X + DX_AM, Y + H_EJE / 2, yb(X + DX_AM))
        dx += ax_; dy += ay_
        # neumático: del suelo a la masa semisuspendida
        zx, zy = zigzag(X - DX_AM, Y_RUEDA, Y - H_EJE / 2, n=4)
        sx += zx; sy += zy
        ax_, ay_ = amortiguador(X + DX_AM, Y_RUEDA, Y - H_EJE / 2)
        dx += ax_; dy += ay_

    pts_x = [XD, XT, XM, XG]
    pts_y = [yb(XD), yb(XT), yb(XM), yb(XG) + H_BAST / 2]
    return dict(bast=(bx, by), masas=(mx, my), muelles=(sx, sy), amort=(dx, dy), pts=(pts_x, pts_y))


n_fr = 48
fases = np.linspace(0, 2 * np.pi, n_fr, endpoint=False)
desp = [amp * np.real(phi * np.exp(1j * f)) for f in fases]

fig = go.Figure()
# Elementos fijos: suelo y posición de equilibrio
fig.add_trace(go.Scatter(x=[-1.0, l + 1.0], y=[Y_RUEDA, Y_RUEDA], mode="lines",
                         line=dict(color="#333", width=3), hoverinfo="skip", showlegend=False))
hx = np.linspace(-1.0, l + 0.8, 40)
fig.add_trace(go.Scatter(x=sum([[v, v - 0.15, None] for v in hx], []),
                         y=sum([[Y_RUEDA, Y_RUEDA - 0.15, None] for _ in hx], []),
                         mode="lines", line=dict(color="#999", width=1), hoverinfo="skip", showlegend=False))
if fantasma:
    g0 = geometria(np.zeros(5))
    fig.add_trace(go.Scatter(x=g0["bast"][0], y=g0["bast"][1], mode="lines",
                             line=dict(color="#aaa", dash="dot"), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=g0["masas"][0], y=g0["masas"][1], mode="lines",
                             line=dict(color="#aaa", dash="dot"), hoverinfo="skip", showlegend=False))
fig.add_annotation(x=l + 0.6, y=Y_RUEDA - 0.35, text="v →", showarrow=False, font=dict(size=16))
n_static = len(fig.data)

estilos = dict(
    bast=dict(mode="lines", fill="toself", fillcolor="rgba(31,78,121,0.35)", line=dict(color="#1f4e79", width=2)),
    masas=dict(mode="lines", fill="toself", fillcolor="rgba(255,127,14,0.6)", line=dict(color="#a65000", width=2)),
    muelles=dict(mode="lines", line=dict(color="#2ca02c", width=2)),
    amort=dict(mode="lines", line=dict(color="#555", width=2)),
    pts=dict(mode="markers+text", marker=dict(size=[9, 9, 9, 11], color=["#d62728"] * 3 + ["#000"]),
             text=["a", "b", "c", "G"], textposition=["bottom right", "bottom left", "bottom right", "top center"]),
)
orden_tr = ["bast", "masas", "muelles", "amort", "pts"]
g = geometria(desp[0])
for k in orden_tr:
    fig.add_trace(go.Scatter(x=g[k][0], y=g[k][1], **estilos[k], hoverinfo="skip", showlegend=False))

frames = []
for j, u in enumerate(desp):
    g = geometria(u)
    frames.append(go.Frame(data=[go.Scatter(x=g[k][0], y=g[k][1]) for k in orden_tr],
                           traces=list(range(n_static, n_static + len(orden_tr))), name=str(j)))
fig.frames = frames

dur = int(vel * 1000 / n_fr)
fig.update_layout(
    height=520, plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(range=[-1.1, l + 1.1], visible=False),
    yaxis=dict(range=[-0.5, Y_BAST + H_BAST + amp + 0.4], visible=False, scaleanchor="x", scaleratio=1),
    updatemenus=[dict(type="buttons", direction="left", x=0.0, y=1.08, xanchor="left",
                      buttons=[
                          dict(label="▶ Animar", method="animate",
                               args=[[str(j) for j in range(n_fr)] * 200,  # bucle continuo
                                     dict(frame=dict(duration=dur, redraw=False), transition=dict(duration=0),
                                          mode="immediate")]),
                          dict(label="⏸ Pausa", method="animate",
                               args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                      ])],
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Desplazamientos exagerados para la visualización (componente máxima = amplitud visual). "
           "Rojo: puntos de anclaje a, b, c; negro: centro de gravedad G del bastidor.")

# =============================================================================
# Forma modal, energía y tabla de frecuencias
# =============================================================================
st.subheader("Forma modal y frecuencias")
b1, b2, b3 = st.columns([1.1, 1, 1.1])

with b1:
    f2 = go.Figure(go.Bar(x=NOMBRES, y=np.real(phi), marker_color=["#1f4e79"] * 2 + ["#ff7f0e"] * 3,
                          text=[f"{v:.3f}" for v in np.real(phi)], textposition="outside"))
    f2.update_layout(title=f"Vector modal φ{modo} (normalizado, máx = 1)" +
                     (" — parte real" if xi_i is not None else ""),
                     yaxis=dict(range=[-1.3, 1.3]), height=340, margin=dict(t=50, b=20))
    st.plotly_chart(f2, use_container_width=True)
    if xi_i is not None:
        st.dataframe(pd.DataFrame({"Coord.": NOMBRES, "|φ|": np.abs(phi).round(4),
                                   "Fase [°]": np.degrees(np.angle(phi)).round(2)}),
                     hide_index=True, use_container_width=True)

with b2:
    e = energia_cinetica(np.real(phi))
    f3 = go.Figure(go.Bar(x=e, y=["Bastidor: bote", "Bastidor: cabeceo", "Eje delantero", "Eje trasero",
                                   "Eje central"], orientation="h", marker_color="#2ca02c",
                          text=[f"{v:.1f} %" for v in e], textposition="outside"))
    f3.update_layout(title="Reparto de energía cinética", xaxis=dict(range=[0, 115], title="%"),
                     yaxis=dict(autorange="reversed"), height=340, margin=dict(t=50, b=20))
    st.plotly_chart(f3, use_container_width=True)

with b3:
    filas = []
    for j in range(5):
        lj = lam_c[j] if j < len(lam_c) else np.nan
        filas.append({
            "Modo": j + 1,
            "fₙ [Hz]": round(wn[j] / (2 * np.pi), 3),
            "ωₙ [rad/s]": round(wn[j], 2),
            "f_d [Hz]": round(lj.imag / (2 * np.pi), 3),
            "ξ [%]": round(-lj.real / abs(lj) * 100, 2),
            "Descripción": describe(normaliza(Phi[:, j])),
        })
    st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True, height=36 * 6 + 3)
    st.caption("f_d y ξ proceden del problema de autovalores en espacio de estados (amortiguamiento no proporcional).")

with st.expander("Matrices del sistema y fundamento teórico"):
    fmt = lambda A_: pd.DataFrame(A_, index=NOMBRES, columns=NOMBRES).style.format("{:.4g}")
    t1, t2, t3 = st.tabs(["Masa m", "Rigidez k", "Amortiguamiento c"])
    t1.dataframe(fmt(M), use_container_width=True)
    t2.dataframe(fmt(K), use_container_width=True)
    t3.dataframe(fmt(C), use_container_width=True)
    st.markdown(r"""
Las ecuaciones de Lagrange conducen a $\;\boldsymbol m\ddot{\boldsymbol x} + \boldsymbol c\dot{\boldsymbol x}
+ \boldsymbol k\boldsymbol x = \boldsymbol Q(t)$ (ec. 3.166–3.170), con las variables del bastidor

$$x = \frac{l_t x_a + l_d x_b}{l},\qquad \theta = \frac{x_a - x_b}{l},\qquad x_c = \frac{l_b x_a + l_a x_b}{l}.$$

**Modos no amortiguados:** $\;(\boldsymbol k - \omega_n^2\,\boldsymbol m)\,\boldsymbol\phi = \boldsymbol 0$.

**Modos amortiguados:** como $\boldsymbol c$ no es proporcional, se resuelve el problema en espacio de estados de
dimensión $2N$:
$$\dot{\boldsymbol z} = \begin{pmatrix}\boldsymbol 0 & \boldsymbol I\\ -\boldsymbol m^{-1}\boldsymbol k & -\boldsymbol m^{-1}\boldsymbol c\end{pmatrix}\boldsymbol z,
\qquad \lambda_r = -\xi_r\omega_r \pm j\,\omega_r\sqrt{1-\xi_r^2}.$$
Los vectores modales son complejos: cada coordenada vibra con su propio desfase, y la animación muestra
$\operatorname{Re}\{\boldsymbol\psi_r e^{j\omega_d t}\}$ (sin el decaimiento exponencial).
""")
