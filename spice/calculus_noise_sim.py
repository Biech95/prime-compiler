#!/usr/bin/env python3
"""
ngspice arbiter for `docs/exp_calculus_noise.md`:  the mismatch calculus under
THERMAL NOISE and a TEMPERATURE GRADIENT -- the two error classes
`docs/mismatch_calculus.md` §7 explicitly excludes.

Nothing here draws a mismatch sample.  Noise is a linear-response quantity and
is obtained exactly from ngspice `.noise`; the loop's treatment of it is
obtained exactly from ngspice `.ac`; the temperature gradient is obtained from
paired `.op` runs (gradient vs uniform control).  The only Monte Carlo is the
cheap numpy propagation of the measured sigmas into the Table-7 metric.

Benches (both already validated for static mismatch):
  [G]  GELU = P11 . P6          -- `predict_gelu_layernorm_sim.build_gelu_netlist`
  [A]  AGC RMSNorm, N = 8       -- `agc_mismatch_sweep` detector + the closed
                                   loop of `rmsnorm_agc_sim.build_agc_netlist`
                                   + a two-junction current-log VGA (new; §2.2
                                   of the experiment document)

Usage:
  OPENBLAS_NUM_THREADS=1 python3 calculus_noise_sim.py
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mismatch_calculus as MC                                   # noqa: E402

# --- constants, all declared in the pre-registration -----------------------
VT = 0.025864890          # ngspice SPICE3-legacy kT/q at 27 C
ISAT = 1e-16
IUNIT = 1e-6
ITAIL = 4e-6
IREF = 4e-6
Q = 1.602176634e-19
KB = 1.380649e-23
T0_C = 27.0               # ngspice default; T0 = 300.15 K
T0_K = 300.15
DT = 2.0                  # gradient across the array, K
RL = 1e9                  # noise-sense load; 4kT/RL = 1.66e-29 A^2/Hz
B_GELU = 1e8              # open-loop band, brief
NCH = 8

WORKDIR = Path(__file__).parent / "out"
DIODE = "BF=1e6 VAF=1e12 RB=0 RE=0 RC=0 CJE=0 CJC=0 TF=0 TR=0"
OPTS = ".options gmin=1e-14 reltol=1e-6 abstol=1e-16"


# ---------------------------------------------------------------------------
# ngspice plumbing
# ---------------------------------------------------------------------------

def run(netlist, tag, timeout=300):
    WORKDIR.mkdir(exist_ok=True)
    cir = WORKDIR / f"{tag}.cir"
    cir.write_text(netlist)
    res = subprocess.run(["ngspice", "-b", str(cir)], capture_output=True,
                         text=True, timeout=timeout, cwd=WORKDIR)
    if res.returncode != 0:
        sys.stderr.write(res.stdout[-3000:] + res.stderr[-3000:])
        raise RuntimeError(f"ngspice failed for {tag}")
    return res.stdout


def kv(txt):
    """parse 'name = value' lines from an ngspice print dump."""
    out = {}
    for line in txt.splitlines():
        p = line.split("=")
        if len(p) == 2:
            try:
                out[p[0].strip().lower()] = float(p[1])
            except ValueError:
                pass
    return out


def temps(i, n=NCH):
    """device temperature of channel i, in degrees C (gradient arm)."""
    return T0_C + DT * i / n


# ---------------------------------------------------------------------------
# [G] GELU bench
# ---------------------------------------------------------------------------

def gelu_currents(x):
    u = 1.702 * x
    sig = 1.0 / (1.0 + np.exp(-u))
    return dict(i_x=np.abs(x) * IUNIT, i_s=ITAIL * sig,
                i_o=np.abs(x) * sig * IUNIT, i_ref=IREF, sig=sig, u=u)


def build_gelu(x, tag, grad=False, freeze_pair=False, ac_ref=False,
               scale=1.0):
    """Nominal-device GELU.  `grad` applies T_i = T0 + 2 i/N to channel i's
    devices (shared Dref at the array mean); `freeze_pair` replaces the
    differential pair by an ideal current source carrying its nominal current
    (isolating the translinear chain); `scale` multiplies the x operand only."""
    N = len(x)
    cur = gelu_currents(x)
    tref = f" temp={T0_C + DT/2:.6f}" if grad else ""
    L = [f"* GELU noise/temp bench N={N} ({tag})", OPTS,
         "Vcc vcc 0 DC 2.0", "Vee vee 0 DC -2.0",
         f"Iref 0 nref DC {IREF:.8e}" + (" AC 1" if ac_ref else ""),
         f"Dref nref 0 DMR{tref}"]
    models = [f".model DMR D(IS={ISAT:.8e} N=1.0)"]
    for i in range(N):
        t = f" temp={temps(i, N):.6f}" if grad else ""
        vd = cur["u"][i] * VT
        if freeze_pair:
            L += [f"Is{i} 0 ns{i} DC {cur['i_s'][i]:.8e}"]
        else:
            L += [f"Vbp{i} bp{i} 0 DC {vd/2:.10e}",
                  f"Vbn{i} bn{i} 0 DC {-vd/2:.10e}",
                  f"Q{i}a qa{i} bp{i} e{i} QA{i}{t}",
                  f"Q{i}b qb{i} bn{i} e{i} QB{i}{t}",
                  f"It{i} e{i} vee DC {ITAIL:.8e}",
                  f"Vca{i} vcc qa{i} DC 0",
                  f"Vcb{i} vcc qb{i} DC 0",
                  f"Fs{i} 0 ns{i} Vca{i} 1.0"]
        L += [f"Ds{i} ns{i} 0 DMS{i}{t}",
              f"Ix{i} 0 nx{i} DC {abs(x[i])*scale*IUNIT:.8e}",
              f"Dx{i} nx{i} 0 DMX{i}{t}",
              f"Eo{i} eo{i} 0 VALUE={{V(nx{i})+V(ns{i})-V(nref)}}",
              f"Do{i} eo{i} nd{i} DMO{i}{t}",
              f"Vd{i} nd{i} 0 DC 0",
              f"Fo{i} 0 no{i} Vd{i} 1.0",
              f"RL{i} no{i} 0 {RL:.6e}"]
        models += [f".model QA{i} NPN(IS={ISAT:.8e} NF=1.0 {DIODE})",
                   f".model QB{i} NPN(IS={ISAT:.8e} NF=1.0 {DIODE})",
                   f".model DMS{i} D(IS={ISAT:.8e} N=1.0)",
                   f".model DMX{i} D(IS={ISAT:.8e} N=1.0)",
                   f".model DMO{i} D(IS={ISAT:.8e} N=1.0)"]
    return L, models


def gelu_op(x, tag, **kw):
    L, models = build_gelu(x, tag, **kw)
    pr = " ".join(f"v(no{i})" for i in range(len(x)))
    L += models + [".control", "op", f"print {pr} > {tag}.txt",
                   "quit", ".endc", ".end"]
    run("\n".join(L), tag)
    d = kv((WORKDIR / f"{tag}.txt").read_text())
    return np.array([d[f"v(no{i})"] for i in range(len(x))]) / RL


def gelu_noise(x, freeze_pair=False):
    """.noise per channel; returns (sigma_rel, white_flag, i_out)."""
    sig, white, iout = [], [], []
    for i in range(len(x)):
        tag = f"gnoise_{i}_{int(freeze_pair)}"
        L, models = build_gelu(x, tag, ac_ref=True, freeze_pair=freeze_pair)
        L += models + [".control", "op",
                       f"print v(no{i}) > {tag}_op.txt",
                       f"noise v(no{i}) Iref dec 20 1e3 {B_GELU:.6e}",
                       "setplot noise1",
                       f"print onoise_spectrum[0] onoise_spectrum[60] "
                       f"onoise_spectrum[99] > {tag}_s.txt",
                       "setplot noise2",
                       f"print onoise_total > {tag}_t.txt",
                       "quit", ".endc", ".end"]
        run("\n".join(L), tag)
        io = kv((WORKDIR / f"{tag}_op.txt").read_text())[f"v(no{i})"] / RL
        sp = kv((WORKDIR / f"{tag}_s.txt").read_text())
        tot = kv((WORKDIR / f"{tag}_t.txt").read_text())["onoise_total"]
        vals = [sp[k] for k in sorted(sp)]
        white.append(max(vals) / min(vals))
        sig.append(tot / RL / abs(io))
        iout.append(io)
    return np.array(sig), np.array(white), np.array(iout)


def gelu_pair_temp(u_grid, dT):
    """differential pair alone: I_a/I_tail vs u at T0 + dT."""
    t = f" temp={T0_C + dT:.6f}" if dT else ""
    L = [f"* pair temp {dT}", OPTS, "Vcc vcc 0 DC 2.0", "Vee vee 0 DC -2.0"]
    models = []
    for i, u in enumerate(u_grid):
        vd = u * VT
        L += [f"Vbp{i} bp{i} 0 DC {vd/2:.10e}",
              f"Vbn{i} bn{i} 0 DC {-vd/2:.10e}",
              f"Q{i}a qa{i} bp{i} e{i} QA{i}{t}",
              f"Q{i}b qb{i} bn{i} e{i} QB{i}{t}",
              f"It{i} e{i} vee DC {ITAIL:.8e}",
              f"Vca{i} vcc qa{i} DC 0", f"Vcb{i} vcc qb{i} DC 0"]
        models += [f".model QA{i} NPN(IS={ISAT:.8e} NF=1.0 {DIODE})",
                   f".model QB{i} NPN(IS={ISAT:.8e} NF=1.0 {DIODE})"]
    pr = " ".join(f"i(Vca{i})" for i in range(len(u_grid)))
    tag = f"pairT{int(dT*10)}"
    L += models + [".control", "op", f"print {pr} > {tag}.txt",
                   "quit", ".endc", ".end"]
    run("\n".join(L), tag)
    d = kv((WORKDIR / f"{tag}.txt").read_text())
    return np.array([abs(d[f"i(vca{i})"]) for i in range(len(u_grid))]) / ITAIL


# ---------------------------------------------------------------------------
# [A] AGC bench
# ---------------------------------------------------------------------------

def build_detector(y_abs, tag, grad=False, ac=False):
    N = len(y_abs)
    tsh = f" temp={T0_C + DT/2:.6f}" if grad else ""
    L = [f"* junction RMS detector N={N} ({tag})", OPTS,
         f"Iref 0 nref DC {IUNIT:.8e}" + (" AC 1" if ac else ""),
         f"Dref nref 0 DMR{tsh}"]
    models = [f".model DMR D(IS={ISAT:.8e} N=1.0)"]
    for i in range(N):
        t = f" temp={temps(i, N):.6f}" if grad else ""
        L += [f"Iy{i} 0 ny{i} DC {max(y_abs[i],1e-4)*IUNIT:.8e}",
              f"Dy{i} ny{i} 0 DMY{i}{t}",
              f"Esq{i} esq{i} 0 VALUE={{2*V(ny{i})-V(nref)}}",
              f"Dsq{i} esq{i} msum DMSQ{i}{t}"]
        models += [f".model DMY{i} D(IS={ISAT:.8e} N=1.0)",
                   f".model DMSQ{i} D(IS={ISAT:.8e} N=1.0)"]
    L += ["Vsum msum 0 DC 0", f"Fms 0 nmss Vsum {1.0/N:.10f}",
          f"RL nmss 0 {RL:.6e}"]
    return L, models


def detector_ms(y_abs, tag, grad=False):
    L, models = build_detector(y_abs, tag, grad=grad)
    L += models + [".control", "op", f"print v(nmss) > {tag}.txt",
                   "quit", ".endc", ".end"]
    run("\n".join(L), tag)
    return kv((WORKDIR / f"{tag}.txt").read_text())["v(nmss)"] / RL / IUNIT


def settle_gain(x, tag, grad=False, iters=16):
    lo, hi = 1e-3, 1e3
    for k in range(iters):
        G = float(np.sqrt(lo * hi))
        ms = detector_ms(np.abs(G * x), f"{tag}_{k}", grad=grad)
        if ms > 1.0:
            hi = G
        else:
            lo = G
    return float(np.sqrt(lo * hi))


def detector_noise(y_abs):
    tag = "dnoise"
    L, models = build_detector(y_abs, tag, ac=True)
    L += models + [".control", "op", f"print v(nmss) > {tag}_op.txt",
                   f"noise v(nmss) Iref dec 20 1e3 1e9",
                   "setplot noise1",
                   f"print onoise_spectrum[0] onoise_spectrum[60] "
                   f"onoise_spectrum[110] > {tag}_s.txt",
                   "setplot noise2", f"print onoise_total > {tag}_t.txt",
                   "quit", ".endc", ".end"]
    run("\n".join(L), tag)
    ims = kv((WORKDIR / f"{tag}_op.txt").read_text())["v(nmss)"] / RL
    sp = kv((WORKDIR / f"{tag}_s.txt").read_text())
    vals = [sp[k] for k in sorted(sp)]
    tot = kv((WORKDIR / f"{tag}_t.txt").read_text())["onoise_total"]
    # onoise_total is integrated over 1e3..1e9 -> density = tot/sqrt(B)
    dens = tot / np.sqrt(1e9 - 1e3)
    return dens / RL / abs(ims), max(vals) / min(vals), abs(ims)


def build_vga(i_in, tag, vg=0.0, dT=None, ac=False):
    """two-junction current-log VGA: Dlog (current driven) -> +Vg -> Dexp."""
    t = f" temp={dT:.6f}" if dT is not None else ""
    L = [f"* current-log VGA ({tag})", OPTS,
         f"Iin 0 nlog DC {i_in:.8e}" + (" AC 1" if ac else ""),
         f"Dlog nlog 0 DL{t}",
         f"Eg ng 0 VALUE={{V(nlog)+{vg:.10e}}}",
         f"Dexp ng nd DE{t}", "Vd nd 0 DC 0",
         "Fo 0 no Vd 1.0", f"RL no 0 {RL:.6e}",
         f".model DL D(IS={ISAT:.8e} N=1.0)",
         f".model DE D(IS={ISAT:.8e} N=1.0)"]
    return L


def vga_noise(i_in, B):
    tag = "vnoise"
    L = build_vga(i_in, tag, ac=True)
    L += [".control", "op", f"print v(no) > {tag}_op.txt",
          f"noise v(no) Iin dec 20 1e3 {B:.6e}",
          "setplot noise1",
          f"print onoise_spectrum[0] onoise_spectrum[60] > {tag}_s.txt",
          "setplot noise2", f"print onoise_total > {tag}_t.txt",
          "quit", ".endc", ".end"]
    run("\n".join(L), tag)
    io = kv((WORKDIR / f"{tag}_op.txt").read_text())["v(no)"] / RL
    tot = kv((WORKDIR / f"{tag}_t.txt").read_text())["onoise_total"]
    sp = kv((WORKDIR / f"{tag}_s.txt").read_text())
    vals = [sp[k] for k in sorted(sp)]
    return tot / RL / abs(io), max(vals) / min(vals)


def vga_temp(i_in, dT, vg):
    tag = f"vgaT{int(dT*1000)}"
    L = build_vga(i_in, tag, vg=vg, dT=T0_C + dT)
    L += [".control", "op", f"print v(no) > {tag}.txt",
          "quit", ".endc", ".end"]
    run("\n".join(L), tag)
    return kv((WORKDIR / f"{tag}.txt").read_text())["v(no)"] / RL


def pair_noise_probe(u, load, B=B_GELU):
    """the differential pair alone, read out through a LINEAR load (`R`)
    or through a log diode (`D`) -- the interface it has inside the GELU."""
    vd = u * VT
    L = [f"* pair probe u={u} load={load}", OPTS,
         "Vcc vcc 0 DC 2.0", "Vee vee 0 DC -2.0",
         f"Vbp bp 0 DC {vd/2:.10e} AC 1", f"Vbn bn 0 DC {-vd/2:.10e}",
         "Qa qa bp e QA", "Qb qb bn e QB", f"It e vee DC {ITAIL:.8e}",
         "Vca vcc qa DC 0", "Vcb vcc qb DC 0", "Fs 0 ns Vca 1.0"]
    L += ([f"Rs ns 0 {RL:.6e}"] if load == "R"
          else ["Ds ns 0 DD", f".model DD D(IS={ISAT:.8e} N=1.0)"])
    L += [f".model QA NPN(IS={ISAT:.8e} NF=1.0 {DIODE})",
          f".model QB NPN(IS={ISAT:.8e} NF=1.0 {DIODE})",
          ".control", "op", "print v(ns) i(Vca) > pnp_op.txt",
          f"noise v(ns) Vbp dec 20 1e3 {B:.6e}", "setplot noise2",
          "print onoise_total > pnp_t.txt", "quit", ".endc", ".end"]
    run("\n".join(L), f"pnp_{load}")
    d = kv((WORKDIR / "pnp_op.txt").read_text())
    tot = kv((WORKDIR / "pnp_t.txt").read_text())["onoise_total"]
    ia = abs(d["i(vca)"])
    return (tot / RL / ia if load == "R" else tot / VT), ia


def build_agc_ac(x, which, tau_det_ns=0.5, c_pf=1.0, k_gain=2e-3):
    """closed AGC loop with two unit-AC relative injection nodes."""
    N = len(x)
    G0 = 1.0 / np.sqrt(np.mean(x ** 2))
    ac_d = "1" if which == "det" else "0"
    ac_c = "1" if which == "ch0" else "0"
    L = [f"* AGC closed loop, AC probe ({which})", OPTS,
         f"Ved ed 0 DC 0 AC {ac_d}", f"Ve0 e0 0 DC 0 AC {ac_c}"]
    for i in range(N):
        L.append(f"Vx{i} x{i} 0 DC {x[i]:.10e}")
    for i in range(N):
        pert = "*(1+V(e0))" if i == 0 else ""
        L.append(f"By{i} y{i} 0 V={{V(g)*V(x{i}){pert}}}")
    sq = "+".join(f"V(y{i})*V(y{i})" for i in range(N))
    L.append(f"Bmsq msqr 0 V={{(1+V(ed))*({sq})/{N}}}")
    r_det = tau_det_ns * 1e-9 / 1e-12
    L += [f"Rdet msqr msq {r_det:.6f}", "Cdet msq 0 1p",
          f"Bint 0 g I={{{k_gain}*(1.0 - V(msq))}}", f"Cg g 0 {c_pf}p",
          f".nodeset v(g)={G0:.8f} v(msq)=1.0"]
    return L


def agc_ac(x, which):
    tag = f"agcac_{which}"
    L = build_agc_ac(x, which)
    L += [".control", "op", "print v(g) > " + f"{tag}_op.txt",
          "ac dec 40 1e5 1e11",
          f"wrdata {tag}.txt v(g) v(y0) v(y1)", "quit", ".endc", ".end"]
    run("\n".join(L), tag)
    g0 = kv((WORKDIR / f"{tag}_op.txt").read_text())["v(g)"]
    d = np.loadtxt(WORKDIR / f"{tag}.txt")
    # wrdata for complex ac: freq re im freq re im ...
    f = d[:, 0]
    def cx(c):
        return d[:, c] + 1j * d[:, c + 1]
    return f, cx(1), cx(4), cx(7), g0


def enbw(f, H):
    h0 = abs(H[0])
    if h0 == 0:
        return 0.0
    return float(np.trapezoid(np.abs(H) ** 2, f) / h0 ** 2)


# ---------------------------------------------------------------------------
# calculus side (no ngspice)
# ---------------------------------------------------------------------------

def sigma_shot(I, B):
    return np.sqrt(2 * Q * B / I)


def mc_decompose(ref, sig, n=20000, seed=77):
    rng = np.random.default_rng(seed)
    cm, pc = [], []
    for _ in range(n):
        y = ref * (1 + rng.normal(0.0, sig))
        a, p = MC.decompose(y, ref)
        cm.append(a)
        pc.append(p)
    return dict(pc_p50=float(np.percentile(pc, 50)) * 100,
                pc_p95=float(np.percentile(pc, 95)) * 100,
                cm_p50=float(np.percentile(cm, 50)) * 100,
                cm_p95=float(np.percentile(cm, 95)) * 100)


# ---------------------------------------------------------------------------

def main():
    if not shutil.which("ngspice"):
        sys.exit("ngspice not found in PATH")
    WORKDIR.mkdir(exist_ok=True)
    out = {}
    W = 92
    print("=" * W)
    print("  CALCULUS UNDER NOISE + TEMPERATURE GRADIENT   "
          "(docs/exp_calculus_noise.md)")
    print("=" * W)

    # ================= [G] GELU ==========================================
    xg = MC.gelu_inputs()
    cur = gelu_currents(xg)
    ref_g = xg * cur["sig"]

    print("\n[G0] bench control (nominal devices, uniform T)")
    y0 = gelu_op(xg, "gctrl")
    ctl = float(np.max(np.abs(np.sign(xg) * y0 / IUNIT - ref_g)
                       / np.abs(ref_g)))
    print(f"   max rel. deviation from x*sigma(1.702x): {ctl:.3e}")

    print(f"\n[G1] .noise per channel, B = {B_GELU/1e6:.0f} MHz")
    sig_m, white, iout = gelu_noise(xg)
    sig_tl, _, _ = gelu_noise(xg, freeze_pair=True)      # translinear only
    s4 = sigma_shot(1.0, B_GELU) * np.sqrt(
        1 / cur["i_x"] + 1 / cur["i_s"] + 1 / cur["i_o"] + 1 / cur["i_ref"])
    s_pair_naive = np.sqrt(s4 ** 2 + 2 * Q * B_GELU / cur["i_s"])
    s_pair_corr = np.sqrt(s4 ** 2 + 2 * Q * B_GELU / cur["i_s"]
                          * (1 - cur["sig"]))
    s_pair_meas = np.sqrt(s4 ** 2 + 2 * Q * B_GELU / cur["i_s"]
                          * (1 - cur["sig"]) * (2 - cur["sig"]))
    print(f"   {'ch':>3} {'I_out/uA':>9} {'meas %':>8} {'TL-only%':>9} "
          f"{'4j %':>7} {'m_TL/4j':>8} {'+naive':>7} {'+(1-s)':>7} "
          f"{'+emp':>7} {'m/emp':>6} {'white':>7}")
    for i in range(NCH):
        print(f"   {i:>3} {iout[i]*1e6:9.4f} {sig_m[i]*100:8.4f} "
              f"{sig_tl[i]*100:9.4f} {s4[i]*100:7.4f} "
              f"{sig_tl[i]/s4[i]:8.4f} {s_pair_naive[i]*100:7.4f} "
              f"{s_pair_corr[i]*100:7.4f} {s_pair_meas[i]*100:7.4f} "
              f"{sig_m[i]/s_pair_meas[i]:6.3f} {white[i]:7.4f}")
    g_meas = mc_decompose(ref_g, sig_m)
    g_p4 = mc_decompose(ref_g, s4)
    g_pn = mc_decompose(ref_g, s_pair_naive)
    g_pc = mc_decompose(ref_g, s_pair_corr)
    g_pe = mc_decompose(ref_g, s_pair_meas)
    print(f"   metric  decompose():  measured pc_p50 = {g_meas['pc_p50']:.4f} %"
          f"  cm_p50 = {g_meas['cm_p50']:.4f} %")
    print(f"     predicted (4j)            {g_p4['pc_p50']:.4f} %"
          f"   ratio {g_p4['pc_p50']/g_meas['pc_p50']:.3f}")
    print(f"     predicted (+naive pair)   {g_pn['pc_p50']:.4f} %"
          f"   ratio {g_pn['pc_p50']/g_meas['pc_p50']:.3f}")
    print(f"     predicted (+(1-sig) pair) {g_pc['pc_p50']:.4f} %"
          f"   ratio {g_pc['pc_p50']/g_meas['pc_p50']:.3f}")
    print(f"     predicted (+empirical)    {g_pe['pc_p50']:.4f} %"
          f"   ratio {g_pe['pc_p50']/g_meas['pc_p50']:.3f}")
    print(f"     static mismatch on record  8.518 %  -> noise/mismatch = "
          f"{g_meas['pc_p50']/8.518:.4f}")
    out["gelu_noise"] = dict(
        ctrl=ctl, sigma_meas=(sig_m * 100).tolist(),
        sigma_tl_only=(sig_tl * 100).tolist(),
        sigma_4j=(s4 * 100).tolist(),
        sigma_pair_naive=(s_pair_naive * 100).tolist(),
        sigma_pair_corr=(s_pair_corr * 100).tolist(),
        sigma_pair_emp=(s_pair_meas * 100).tolist(),
        white=white.tolist(), i_out_uA=(iout * 1e6).tolist(),
        metric_meas=g_meas, metric_4j=g_p4, metric_pair_naive=g_pn,
        metric_pair_corr=g_pc, metric_pair_emp=g_pe)

    # ---- [G1b] where does the pair term come from? ----------------------
    print("\n[G1b] pair read out through a linear load vs through the log "
          "diode it drives inside the GELU")
    print(f"   {'u':>8} {'sigma':>8} {'I_a/uA':>8} {'R: meas %':>10} "
          f"{'(1-s) model':>12} {'D: meas %':>10} {'D: +own':>9} "
          f"{'D: emp law':>11}")
    pr_rows = []
    for u in (-3.228, -2.0, -1.35, -0.497, 0.0, 0.409, 1.0, 2.0):
        s = 1.0 / (1.0 + np.exp(-u))
        sr, ia = pair_noise_probe(u, "R")
        sd, _ = pair_noise_probe(u, "D")
        nv = sigma_shot(ia, B_GELU)
        m_r = nv * np.sqrt(1 - s)
        m_d = nv * np.sqrt(1 + (1 - s))
        m_e = nv * np.sqrt(1 + (1 - s) * (2 - s))
        print(f"   {u:8.3f} {s:8.4f} {ia*1e6:8.4f} {sr*100:10.4f} "
              f"{m_r*100:12.4f} {sd*100:10.4f} {m_d*100:9.4f} "
              f"{m_e*100:11.4f}")
        pr_rows.append(dict(u=u, sigma=s, i_a=ia, R_meas=sr * 100,
                            R_model=m_r * 100, D_meas=sd * 100,
                            D_naive_own=m_d * 100, D_emp=m_e * 100))
    out["pair_noise_probe"] = pr_rows

    # ---- [G2] temperature gradient --------------------------------------
    print(f"\n[G2] temperature gradient {DT} K across N = {NCH}")
    tg = {}
    for lbl, fp in (("translinear chain only", True), ("full GELU", False)):
        row = {}
        for s in (1.0, 0.25):
            a = gelu_op(xg, f"gt_u_{int(s*100)}_{int(fp)}",
                        grad=False, freeze_pair=fp, scale=s)
            b = gelu_op(xg, f"gt_g_{int(s*100)}_{int(fp)}",
                        grad=True, freeze_pair=fp, scale=s)
            rel = b / a - 1.0
            row[f"scale_{s}"] = dict(rel=(rel * 100).tolist(),
                                     abs_err=(np.abs(b - a)).tolist())
            print(f"   {lbl:<24} scale={s:<5} rel. error per channel (%): "
                  f"{np.round(rel*100, 3).tolist()}")
        r1 = np.array(row["scale_1.0"]["rel"])
        r2 = np.array(row["scale_0.25"]["rel"])
        a1 = np.array(row["scale_1.0"]["abs_err"])
        a2 = np.array(row["scale_0.25"]["abs_err"])
        m = np.abs(r1) > 1e-6          # ch. at the reference temperature: 0/0
        row["rel_ratio"] = float(np.median(r2[m] / r1[m]))
        row["abs_ratio"] = float(np.median(a2[m] / a1[m]))
        print(f"   {lbl:<24} median rel-error ratio (0.25x / 1x) = "
              f"{row['rel_ratio']:.4f}   [1.0 => GAIN-type, "
              f"4.0 => OFFSET-type]")
        print(f"   {lbl:<24} median abs-error ratio (0.25x / 1x) = "
              f"{row['abs_ratio']:.4f}   [0.25 => GAIN, 1.0 => OFFSET]")
        # decomposition of the gradient error
        a = gelu_op(xg, f"gt_u_dec_{int(fp)}", grad=False, freeze_pair=fp)
        b = gelu_op(xg, f"gt_g_dec_{int(fp)}", grad=True, freeze_pair=fp)
        cm, pc = MC.decompose(b, a)
        row["cm"] = cm * 100
        row["pc"] = pc * 100
        print(f"   {lbl:<24} decompose(): common-mode {cm*100:.4f} %, "
              f"per-channel {pc*100:.4f} %")
        tg[lbl] = row
    out["gelu_temp"] = tg

    # ---- [G3] the pair alone --------------------------------------------
    print(f"\n[G3] subthreshold pair alone: offset or slope?")
    ug = np.linspace(-4, 4, 17)
    p0 = gelu_pair_temp(ug, 0.0)
    p2 = gelu_pair_temp(ug, DT)
    # fit sigma((u - delta)/s) by least squares in logit space
    good = (p2 > 1e-6) & (p2 < 1 - 1e-6) & (p0 > 1e-6) & (p0 < 1 - 1e-6)
    lg0 = np.log(p0[good] / (1 - p0[good]))
    lg2 = np.log(p2[good] / (1 - p2[good]))
    A = np.vstack([lg0, np.ones(good.sum())]).T
    coef, *_ = np.linalg.lstsq(A, lg2, rcond=None)
    slope, inter = float(coef[0]), float(coef[1])
    delta = -inter / slope * 1.0
    print(f"   logit(T0+2K) = {slope:.6f} * logit(T0) + {inter:.3e}")
    print(f"   => slope change {(slope-1)*100:+.4f} %  "
          f"(V_T gain: expected {-DT/T0_K*100:+.4f} % / "
          f"{DT/T0_K*100:+.4f} %),  input-referred offset "
          f"delta = {delta:.3e} units of u")
    out["pair_temp"] = dict(u=ug.tolist(), p_T0=p0.tolist(),
                            p_T2=p2.tolist(), slope=slope, inter=inter,
                            delta_u=delta)

    # ================= [A] AGC ===========================================
    rng0 = np.random.default_rng(2026)
    xa = rng0.standard_normal(NCH) * 1.5
    xa = np.where(np.abs(xa) < 0.05, 0.05 * np.sign(xa) + (xa == 0) * 0.05, xa)
    ref_a = xa / np.sqrt(np.mean(xa ** 2))
    G0 = 1.0 / np.sqrt(np.mean(xa ** 2))
    print(f"\n[A0] AGC input draw (Table-7 draw, seed 2026): "
          f"{np.round(xa,4).tolist()}")
    print(f"     G0 = {G0:.6f},  channel currents |y_i| (uA) = "
          f"{np.round(np.abs(ref_a),4).tolist()}")

    # ---- [A1] closed-loop AC: what does the loop do to a spectrum? ------
    print("\n[A1] closed-loop AC transfer functions")
    f, Hg_d, Hy0_d, Hy1_d, g_op = agc_ac(xa, "det")
    _, Hg_c, Hy0_c, Hy1_c, _ = agc_ac(xa, "ch0")
    w = ref_a ** 2 / np.sum(ref_a ** 2)
    print(f"   op-point gain node V(g) = {g_op:.6f}  (ideal {G0:.6f})")
    # |H| at DC and at the top of the sweep
    def lohi(H):
        return abs(H[0]), abs(H[-1])
    print(f"   {'transfer':<34} {'|H| @100 kHz':>13} {'|H| @100 GHz':>13} "
          f"{'ENBW/MHz':>10}")
    rows = [("detector rel.err -> dG/G", Hg_d / g_op),
            ("detector rel.err -> y0", Hy0_d / (g_op * xa[0])),
            ("detector rel.err -> y1", Hy1_d / (g_op * xa[1])),
            ("ch0 VGA rel.err -> y0", Hy0_c / (g_op * xa[0])),
            ("ch0 VGA rel.err -> y1", Hy1_c / (g_op * xa[1]))]
    ac_rows = {}
    for lbl, H in rows:
        lo, hi = lohi(H)
        bw = enbw(f, H)
        print(f"   {lbl:<34} {lo:13.6f} {hi:13.3e} {bw/1e6:10.2f}")
        ac_rows[lbl] = dict(dc=lo, hf=hi, enbw_MHz=bw / 1e6)
    # -3 dB corner of the detector->gain path
    Hd = np.abs(Hg_d / g_op)
    i3 = int(np.argmax(Hd < Hd[0] / np.sqrt(2)))
    f3 = float(f[i3])
    B_loop = ac_rows["detector rel.err -> dG/G"]["enbw_MHz"] * 1e6
    print(f"   loop -3 dB corner (detector path): {f3/1e6:.2f} MHz;  "
          f"ENBW_loop = {B_loop/1e6:.2f} MHz   "
          f"[pre-registered estimate 318 MHz]")
    print(f"   R2-for-noise check: H(ch0->y0) DC = "
          f"{ac_rows['ch0 VGA rel.err -> y0']['dc']:.5f}  "
          f"(1 - w0 = {1-w[0]:.5f});  HF = "
          f"{ac_rows['ch0 VGA rel.err -> y0']['hf']:.5f}")
    print(f"   cross-channel demotion H(ch0->y1) DC = "
          f"{ac_rows['ch0 VGA rel.err -> y1']['dc']:.5f}  "
          f"(-w0 = {-w[0]:.5f})")
    out["agc_ac"] = dict(rows=ac_rows, f3db=f3, enbw_loop=B_loop,
                         g_op=g_op, G0=G0, w=w.tolist())

    # ---- [A2] noise densities -------------------------------------------
    print("\n[A2] .noise: detector (shared) and VGA (per channel)")
    s_det, white_det, i_ms = detector_noise(np.abs(ref_a))
    print(f"   detector relative noise density = {s_det:.6e} /sqrt(Hz), "
          f"whiteness {white_det:.4f}, I_ms = {i_ms*1e6:.5f} uA")
    # R3 walk: d ln I_ms = sum_i w_i (2 b_y,i + b_sq,i) - b_ref,
    # w_i = y_i^2 / sum y_j^2  (energy weights at the summed node)
    wdet = ref_a ** 2 / np.sum(ref_a ** 2)
    i_y = np.abs(ref_a) * IUNIT
    i_sq = ref_a ** 2 * IUNIT
    s_det_pred = np.sqrt(2 * Q * (np.sum(wdet ** 2 * (4 / i_y + 1 / i_sq))
                                  + 1 / IUNIT))
    s_det_naive = np.sqrt(2 * Q * (np.sum(4 / i_y + 1 / i_sq) + 1 / IUNIT))
    print(f"   calculus, R3 energy weights      = {s_det_pred:.6e} "
          f"/sqrt(Hz)   ratio {s_det_pred/s_det:.3f}")
    print(f"   calculus, weights FORGOTTEN      = {s_det_naive:.6e} "
          f"/sqrt(Hz)   ratio {s_det_naive/s_det:.3f}")

    B_tab = [("ENBW_loop (measured)", B_loop),
             ("1/(4 t_AGC) = 56.8 MHz", 1.0 / (4 * 4.4e-9)),
             ("100 MHz", 1e8)]
    agc_res = {}
    for lbl, B in B_tab:
        sm, wh = [], []
        for i in range(NCH):
            s, wf = vga_noise(abs(ref_a[i]) * IUNIT, B)
            sm.append(s)
            wh.append(wf)
        sm = np.array(sm)
        sp = np.sqrt(2.0) * sigma_shot(np.abs(ref_a) * IUNIT, B)
        m = mc_decompose(ref_a, sm)
        p = mc_decompose(ref_a, sp)
        sg = 0.5 * s_det * np.sqrt(B) * ac_rows[
            "detector rel.err -> dG/G"]["dc"]
        print(f"   B = {lbl:<24} VGA sigma/ch (%) meas "
              f"{np.round(sm*100,3).tolist()}")
        print(f"   {'':<28} calculus       "
              f"{np.round(sp*100,3).tolist()}")
        print(f"   {'':<28} pc_p50 meas {m['pc_p50']:.4f} % | "
              f"calculus {p['pc_p50']:.4f} % | ratio "
              f"{p['pc_p50']/m['pc_p50']:.3f} | K1 line 0.690 % -> "
              f"{'TRIGGERED' if m['pc_p50'] > 0.690 else 'not triggered'}")
        print(f"   {'':<28} detector-induced common-mode gain jitter "
              f"{sg*100:.4f} %")
        agc_res[lbl] = dict(B=B, sigma_meas=(sm * 100).tolist(),
                            sigma_calc=(sp * 100).tolist(),
                            white=wh, metric_meas=m, metric_calc=p,
                            cm_from_detector=sg * 100)
    out["agc_noise"] = dict(sigma_det_density=s_det,
                            sigma_det_density_calc=s_det_pred,
                            sigma_det_density_unweighted=s_det_naive,
                            white_det=white_det, by_band=agc_res)
    # kT/C on the gain node
    ktc = np.sqrt(KB * T0_K / 1e-12)
    print(f"   kT/C floor on C_g = 1 pF: {ktc*1e6:.2f} uV on V(g) = "
          f"{g_op:.4f} V -> {ktc/g_op*100:.4f} % common-mode")
    out["agc_noise"]["ktc_pct"] = float(ktc / g_op * 100)

    # ---- [A3] temperature gradient on the AGC bench ----------------------
    print(f"\n[A3] temperature gradient {DT} K on the AGC bench")
    G_u = settle_gain(xa, "agT_u", grad=False)
    G_g = settle_gain(xa, "agT_g", grad=True)
    print(f"   detector-side gradient: G {G_u:.6f} -> {G_g:.6f}  "
          f"({(G_g/G_u-1)*100:+.4f} % common-mode gain shift)")
    # the VGA actually applies the loop gain G0: V_g = VT ln(G0)
    vg = VT * np.log(G0)
    y_u = np.array([vga_temp(abs(xa[i]) * IUNIT, 0.0, vg) for i in range(NCH)])
    y_g = np.array([vga_temp(abs(xa[i]) * IUNIT, DT * i / NCH, vg)
                    for i in range(NCH)])
    rel = y_g / y_u - 1.0
    cm_v, pc_v = MC.decompose(np.sign(xa) * y_g, np.sign(xa) * y_u)
    pred = -np.log(G0) * (DT * np.arange(NCH) / NCH) / T0_K
    print(f"   VGA at gain G0 = {G0:.5f} (V_g = {vg*1e3:.4f} mV)")
    print(f"   VGA-side gradient, per-channel rel. error (%): "
          f"{np.round(rel*100,4).tolist()}")
    print(f"   calculus  -ln(G0)*dT_i/T (%):                 "
          f"{np.round(pred*100,4).tolist()}")
    print(f"   VGA-side decompose(): common-mode {cm_v*100:.4f} %, "
          f"per-channel {pc_v*100:.4f} %")
    out["agc_temp"] = dict(G_uniform=G_u, G_grad=G_g,
                           cm_detector=(G_g / G_u - 1) * 100,
                           vga_rel=(rel * 100).tolist(),
                           vga_pred=(pred * 100).tolist(),
                           vga_cm=cm_v * 100, vga_pc=pc_v * 100)

    p = WORKDIR / "calculus_noise.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\n  raw: {p}")
    print("=" * W)


if __name__ == "__main__":
    main()
