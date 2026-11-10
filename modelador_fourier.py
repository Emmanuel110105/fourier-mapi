"""
Modelador de Series de Fourier  ·  MAPI (Matemáticas Avanzadas para Ingeniería)
Convenciones del manual (Unidad 7):
  f(t) = a0/2 + Σ [ an cos(nω0 t) + bn sen(nω0 t) ],   ω0 = 2π/T
  a0 = (2/T)∫f dt,  an = (2/T)∫f cos(nω0 t) dt,  bn = (2/T)∫f sen(nω0 t) dt
  forma amplitud-fase:  f(t) = a0/2 + Σ Cn cos(nω0 t − φn),  Cn = √(an²+bn²),  φn = arctan(bn/an)
  valor eficaz del armónico n:  Cn' = Cn/√2
  forma compleja:  f(t) = Σ cn e^{j n ω0 t},  cn = (an − j bn)/2,  c0 = a0/2

Ejecutar:   streamlit run modelador_fourier.py
"""
import numpy as np
import sympy as sp
import warnings
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
t = sp.Symbol("t", real=True)
n = sp.Symbol("n", integer=True, positive=True)
LOC = {"t": t, "pi": sp.pi, "n": n}

PRESETS = {
    "Onda cuadrada (±1, T=2π)": "-1 ; -pi ; 0\n1 ; 0 ; pi",
    "Diente de sierra  f(t)=t  en (−π, π)": "t ; -pi ; pi",
    "Triangular  f(t)=|t|  en (−π, π)": "-t ; -pi ; 0\nt ; 0 ; pi",
    "Parábola  f(t)=t²  en (−π, π)": "t**2 ; -pi ; pi",
    "Pulso (1 en [0,1], 0 en [1,4])": "1 ; 0 ; 1\n0 ; 1 ; 4",
    "Rectificada de media onda (sen t)": "sin(t) ; 0 ; pi\n0 ; pi ; 2*pi",
    "Personalizada": "t ; -pi ; pi",
}


# ---------------------------------------------------------------- núcleo
def parse_pieces(text):
    """Cada línea:  expresión ; desde ; hasta   (tramos contiguos, en un periodo)."""
    pieces = []
    for raw in text.strip().splitlines():
        if not raw.strip():
            continue
        parts = [p.strip() for p in raw.split(";")]
        if len(parts) != 3:
            raise ValueError(f"Línea mal escrita: «{raw}». Formato: expresión ; desde ; hasta")
        expr = sp.sympify(parts[0].replace("^", "**"), locals=LOC)
        a = sp.sympify(parts[1].replace("^", "**"), locals=LOC)
        b = sp.sympify(parts[2].replace("^", "**"), locals=LOC)
        if not (a < b):
            raise ValueError(f"En «{raw}» debe cumplirse desde < hasta.")
        pieces.append((expr, a, b))
    if not pieces:
        raise ValueError("Escribe al menos un tramo.")
    for (_, _, b1), (_, a2, _) in zip(pieces, pieces[1:]):
        if sp.simplify(b1 - a2) != 0:
            raise ValueError("Los tramos deben ser contiguos (el 'hasta' de uno = el 'desde' del siguiente).")
    return pieces


def lambdify_pieces(pieces):
    return [(sp.lambdify(t, e, "numpy"), float(a), float(b), e) for e, a, b in pieces]


def f_numeric(tt, lp, t0, T):
    tt = np.asarray(tt, dtype=float)
    x = t0 + np.mod(tt - t0, T)
    out = np.zeros_like(x)
    for fn, a, b, _ in lp:
        m = (x >= a) & (x < b)
        out[m] = fn(x[m]) * np.ones(m.sum())
    return out


def detect_parity(lp, t0, T, a_sym):
    rng = np.random.default_rng(1)
    ts = rng.uniform(0.01, T / 2 - 0.01, 200)
    fp, fm = f_numeric(ts, lp, t0, T), f_numeric(-ts, lp, t0, T)
    if np.allclose(fp, fm, atol=1e-9):
        return "par"
    if np.allclose(fp, -fm, atol=1e-9):
        return "impar"
    return "ninguna"


def coef_symbolic(pieces, w):
    """Devuelve (a0, an, bn) simbólicos y las integrales sin evaluar por tramo."""
    T = pieces[-1][2] - pieces[0][1]
    k = 2 / T
    ints = {"a0": [], "an": [], "bn": []}
    res = {}
    for key, kern in (("a0", sp.Integer(1)), ("an", sp.cos(n * w * t)), ("bn", sp.sin(n * w * t))):
        total = 0
        for e, a, b in pieces:
            ints[key].append(sp.Integral(e * kern, (t, a, b)))
            total += sp.integrate(e * kern, (t, a, b))
        res[key] = sp.simplify(k * total)
    return res, ints


def coef_numeric(lp, t0, T, N):
    tt = np.linspace(t0, t0 + T, 200001)
    ft = f_numeric(tt, lp, t0, T)
    w0 = 2 * np.pi / T
    a0 = 2 / T * np.trapezoid(ft, tt)
    ks = np.arange(1, N + 1)
    an = np.array([2 / T * np.trapezoid(ft * np.cos(k * w0 * tt), tt) for k in ks])
    bn = np.array([2 / T * np.trapezoid(ft * np.sin(k * w0 * tt), tt) for k in ks])
    return a0, an, bn


def to_arrays(expr, N):
    ks = np.arange(1, N + 1)
    fn = sp.lambdify(n, expr, "numpy")
    return np.broadcast_to(np.asarray(fn(ks), dtype=float), ks.shape).copy()


def partial_sum(tt, a0, an, bn, w0, N):
    s = np.full_like(tt, a0 / 2, dtype=float)
    for k in range(1, N + 1):
        s += an[k - 1] * np.cos(k * w0 * tt) + bn[k - 1] * np.sin(k * w0 * tt)
    return s


# ---------------------------------------------------------------- interfaz
st.set_page_config(page_title="Modelador de Fourier · MAPI", layout="wide")
st.title("Modelador de Series de Fourier")
st.caption("Matemáticas Avanzadas para Ingeniería · Unidad 7 · pasos explicados + gráficas")

with st.sidebar:
    st.header("1. Función")
    preset = st.selectbox("Ejemplo", list(PRESETS))
    text = st.text_area(
        "Tramos de UN periodo (una línea por tramo):  expresión ; desde ; hasta",
        PRESETS[preset], height=130, key=preset,
    )
    st.caption("Usa t, pi, sin(t), cos(t), exp(t), t**2 ... Los tramos deben ser contiguos.")
    st.header("2. Visualización")
    N = st.slider("Número de armónicos N", 1, 60, 5)
    force_num = st.checkbox("Forzar cálculo numérico (si la integral simbólica tarda)")

try:
    pieces = parse_pieces(text)
except Exception as ex:
    st.error(str(ex))
    st.stop()

t0 = pieces[0][1]
t1 = pieces[-1][2]
T = sp.simplify(t1 - t0)
w = sp.simplify(2 * sp.pi / T)
lp = lambdify_pieces(pieces)
T_f, t0_f, w0_f = float(T), float(t0), float(w)

# --- coeficientes
symbolic_ok = False
if not force_num:
    try:
        res, ints = coef_symbolic(pieces, w)
        symbolic_ok = not any(r.has(sp.Integral) for r in res.values())
    except Exception:
        symbolic_ok = False

if symbolic_ok:
    a0_e, an_e, bn_e = res["a0"], res["an"], res["bn"]
    a0 = float(a0_e)
    an_all, bn_all = to_arrays(an_e, 200), to_arrays(bn_e, 200)
else:
    a0, an_all, bn_all = coef_numeric(lp, t0_f, T_f, 200)
    st.warning("Cálculo numérico (no se obtuvo forma cerrada): los pasos simbólicos no están disponibles.")

an, bn = an_all[:N], bn_all[:N]
paridad = detect_parity(lp, t0_f, T_f, t0)

tab1, tab2, tab3, tab4 = st.tabs(["Procedimiento", "Aproximación", "Armónicos y espectros", "Forma compleja"])

# =============== TAB 1: procedimiento ===============
with tab1:
    st.subheader("Paso 1. Datos")
    st.latex(rf"f(t)=\begin{{cases}}"
             + r"\\".join(rf"{sp.latex(e)} & {sp.latex(a)}<t<{sp.latex(b)}" for e, a, b in pieces)
             + r"\end{cases}")
    st.latex(rf"T = {sp.latex(t1)}-({sp.latex(t0)}) = {sp.latex(T)} \qquad "
             rf"\omega_0=\frac{{2\pi}}{{T}}={sp.latex(w)}")

    st.subheader("Paso 2. Paridad")
    if paridad == "par":
        st.success("f(−t) = f(t) → **función par** ⇒ bₙ = 0 (solo cosenos).")
    elif paridad == "impar":
        st.success("f(−t) = −f(t) → **función impar** ⇒ a₀ = aₙ = 0 (solo senos).")
    else:
        st.info("No es par ni impar (o el intervalo no es simétrico): se calculan aₙ y bₙ.")

    st.subheader("Paso 3. Fórmulas de los coeficientes")
    st.latex(r"a_0=\frac{2}{T}\int_{T} f(t)\,dt \qquad a_n=\frac{2}{T}\int_{T} f(t)\cos(n\omega_0 t)\,dt \qquad b_n=\frac{2}{T}\int_{T} f(t)\sin(n\omega_0 t)\,dt")

    if symbolic_ok:
        for key, nombre in (("a0", "a_0"), ("an", "a_n"), ("bn", "b_n")):
            st.subheader(f"Paso 4. Cálculo de ${nombre}$")
            suma = " + ".join(sp.latex(i) for i in ints[key])
            st.latex(rf"{nombre}=\frac{{2}}{{{sp.latex(T)}}}\left[{suma}\right]")
            st.latex(rf"{nombre}={sp.latex(res[key])}")
        st.caption("Aquí n es entero positivo, por eso sen(nπ)=0 y cos(nπ)=(−1)ⁿ.")
        st.subheader("Paso 5. Serie de Fourier")
        st.latex(rf"f(t)\sim \frac{{a_0}}{{2}}+\sum_{{n=1}}^{{\infty}}\left[a_n\cos(n\omega_0 t)+b_n\sin(n\omega_0 t)\right]")
        terms = sp.Rational(1, 2) * a0_e
        st.latex(rf"\frac{{a_0}}{{2}}={sp.latex(sp.simplify(terms))}")
    else:
        st.write("Coeficientes calculados numéricamente (regla del trapecio).")

    st.subheader("Primeros coeficientes")
    k = np.arange(1, min(N, 10) + 1)
    st.dataframe(pd.DataFrame({"n": k, "aₙ": an_all[:len(k)], "bₙ": bn_all[:len(k)]}).round(5), hide_index=True)
    st.write(f"a₀ = {a0:.5f}  →  a₀/2 = {a0 / 2:.5f} (valor medio)")

# =============== TAB 2: aproximación ===============
with tab2:
    tt = np.linspace(t0_f, t0_f + 2 * T_f, 4000)
    ft = f_numeric(tt, lp, t0_f, T_f)
    sN = partial_sum(tt, a0, an, bn, w0_f, N)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(tt, ft, "k", lw=1.6, label="f(t)")
    ax[0].plot(tt, sN, "tab:red", lw=1.4, label=f"Suma parcial, N = {N}")
    ax[0].grid(alpha=.3); ax[0].legend(); ax[0].set_ylabel("f(t)")
    ax[1].plot(tt, ft - sN, "tab:blue", lw=1)
    ax[1].grid(alpha=.3); ax[1].set_ylabel("error"); ax[1].set_xlabel("t")
    st.pyplot(fig)
    ecm = np.sqrt(np.mean((ft - sN) ** 2))
    st.write(f"Error cuadrático medio ≈ **{ecm:.4f}**. Mueve el slider de N: el error baja, pero en las "
             "discontinuidades queda un pico de ~9 % que no desaparece (**fenómeno de Gibbs**), y la serie converge "
             "al promedio de los límites laterales (Teorema de Dirichlet).")

# =============== TAB 3: armónicos y espectros ===============
with tab3:
    Cn = np.hypot(an_all, bn_all)
    phin = np.arctan2(bn_all, an_all)
    k = np.arange(1, N + 1)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.stem(np.r_[0, k], np.r_[abs(a0) / 2, Cn[:N]], basefmt=" ")
        ax.set_xlabel("n (armónico)"); ax.set_ylabel("Cₙ"); ax.set_title("Espectro de amplitud"); ax.grid(alpha=.3)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        mask = Cn[:N] > 1e-9
        ax.stem(k[mask], phin[:N][mask], basefmt=" ")
        ax.set_xlabel("n (armónico)"); ax.set_ylabel("φₙ [rad]"); ax.set_title("Espectro de fase"); ax.grid(alpha=.3)
        st.pyplot(fig)
    st.latex(r"f(t)=\frac{a_0}{2}+\sum C_n\cos(n\omega_0 t-\varphi_n),\quad C_n=\sqrt{a_n^2+b_n^2},\quad \varphi_n=\arctan\frac{b_n}{a_n}")
    m = min(N, 15)
    df = pd.DataFrame({
        "n": np.arange(1, m + 1),
        "ω = nω₀ [rad/s]": np.arange(1, m + 1) * w0_f,
        "f = n/T [Hz]": np.arange(1, m + 1) / T_f,
        "Cₙ (amplitud)": Cn[:m],
        "φₙ [rad]": phin[:m],
        "φₙ [°]": np.degrees(phin[:m]),
        "Cₙ′ = Cₙ/√2 (eficaz)": Cn[:m] / np.sqrt(2),
    }).round(5)
    st.dataframe(df, hide_index=True)
    tt_i = np.linspace(t0_f, t0_f + T_f, 200001)
    frms = float(np.sqrt(np.trapezoid(f_numeric(tt_i, lp, t0_f, T_f) ** 2, tt_i) / T_f))
    frms_s = float(np.sqrt((a0 / 2) ** 2 + np.sum(Cn[:N] ** 2) / 2))
    st.write(f"**Valor eficaz** de f(t): {frms:.5f} (directo)  ·  {frms_s:.5f} (con {N} armónicos, Parseval: "
             "F² = (a₀/2)² + Σ Cₙ²/2).")
    st.write(f"**Componente fundamental** (n = 1): amplitud C₁ = {Cn[0]:.5f}, frecuencia f₀ = 1/T = {1 / T_f:.5f} Hz.")

# =============== TAB 4: forma compleja ===============
with tab4:
    st.latex(r"f(t)=\sum_{n=-\infty}^{\infty} c_n e^{jn\omega_0 t},\qquad c_n=\frac{1}{T}\int_T f(t)e^{-jn\omega_0 t}dt=\frac{a_n-jb_n}{2},\quad c_0=\frac{a_0}{2},\quad c_{-n}=\overline{c_n}")
    cn = (an_all - 1j * bn_all) / 2
    ks = np.arange(-N, N + 1)
    cvals = np.array([a0 / 2 if q == 0 else (cn[q - 1] if q > 0 else np.conj(cn[-q - 1])) for q in ks])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].stem(ks, np.abs(cvals), basefmt=" "); ax[0].set_title("|cₙ|"); ax[0].set_xlabel("n"); ax[0].grid(alpha=.3)
    mk = np.abs(cvals) > 1e-9
    ax[1].stem(ks[mk], np.angle(cvals)[mk], basefmt=" "); ax[1].set_title("arg(cₙ) [rad]"); ax[1].set_xlabel("n"); ax[1].grid(alpha=.3)
    st.pyplot(fig)
    q = ks[(ks >= -min(N, 6)) & (ks <= min(N, 6))]
    cc = np.array([cvals[list(ks).index(i)] for i in q])
    st.dataframe(pd.DataFrame({"n": q, "cₙ": [f"{z.real:.5f} {'+' if z.imag >= 0 else '−'} j{abs(z.imag):.5f}" for z in cc],
                               "|cₙ|": np.abs(cc).round(5), "arg [rad]": np.angle(cc).round(5)}), hide_index=True)
    tt = np.linspace(t0_f, t0_f + 2 * T_f, 3000)
    rec = sum(cvals[i] * np.exp(1j * ks[i] * w0_f * tt) for i in range(len(ks)))
    st.write("Comprobación: la suma compleja reconstruye la misma señal que la forma trigonométrica "
             f"(diferencia máxima = {np.max(np.abs(rec.imag)) + np.max(np.abs(rec.real - partial_sum(tt, a0, an, bn, w0_f, N))):.2e}).")
