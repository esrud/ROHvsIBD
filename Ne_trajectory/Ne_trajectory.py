#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 02 10:01:39 2026

@author: enrique

===============================================================================
 Ne_trajectory.py  --  stand-alone inference of the N(t) trajectory from the
 observed ROH length spectrum
===============================================================================

 -------------------------------------------------------------
 Breakage rate per Mb and generation            B = c + m          (Eq. A2)
     c = recombination events per Mb
     m = (mutation + gene conversion) per Mb = (mu + g) * 1e6
 Boundary displacement to the nearest het site  gamma = c * D/H    (Eq. 10)
     g = ln[1/(1 - gamma)]
 Coverage of the genome by ROH of length x      (Eqs. A14, A24, A25)
     p_ROH(x|h) = 4 h B^2 x * fhat''(s),   s = 2 B x - 4 g
 where fhat(s) = E[exp(-sT)] is the Laplace transform of the coalescence-time
 density.  For a piecewise-constant history fhat'' is a finite sum (Eq. A16):
     fhat''(s) = sum_k lam_k S_{k-1} e^{+lam_k tau_{k-1}}
                 [Phi(tau_{k-1}; s+lam_k) - Phi(tau_k; s+lam_k)]
     Phi(t; a)  = e^{-at} (t^2/a + 2t/a^2 + 2/a^3)

 PER-EPOCH INVERSION  (Appendix, Section 5; closed form, no fitting)
 -------------------------------------------------------------------
 Epoch k is read at the tract length that probes the middle of the epoch,
     t = exp(psi(3)) / (2 B x*),   x* = x - 2g/B                   (Eq. A7)
 The epochs already determined are subtracted; with
     w     = lam_k/(s+lam_k) = 1/(1 + 4 B N_k x)
     sigma = 2 B x tau_{k-1}
     V_k   = (s^2/2) [p_ROH/(4 h B^2 x) - G_{<k}] / (S_{k-1} e^{-sigma})
 w is the root in (0,1) of the cubic                               (Eq. A17)
     2w^3 - 2(sigma+2) w^2 + [(sigma+1)^2 + 1] w - 2 V_k = 0
 solved explicitly by Cardano, and                                 (Eq. A18)
     N_k = (1 - w) / (4 B x w)
 The only approximation is the one built into A17-A18: the epoch being read is
 treated as terminal, so whatever happens after tau_k leaks into N_k.

 INPUT
 -----
 SIMULATED_PARAMS.txt   every value sits on the line after the comment that
                        describes it.  Entries are located by a keyword in the
                        comment and their units are taken from it:
                          "mutation rate"         mu, per site per generation
                          "gene conversion"       per site per generation
                          "recombination"         cM/Mb (or per bp if the
                                                  comment says so)
                          "distance between"      D, the mean distance between
                                                  polymorphic sites, in Mb
                          "heterozygosity"        H, the mean heterozygosity of
                                                  those polymorphic sites
                          "trajectory"            file with the true N(t)
                          "chromosome number", "chromosome size (bp)"
                        D and H enter only through D/H, the mean distance
                        between HETEROZYGOUS sites: with 1/D polymorphic sites
                        per Mb, each heterozygous with probability H, an
                        individual carries H/D heterozygous sites per Mb.  So
                        gamma = c * (D/H) and theta_H = H/D.  The program also
                        prints E[T] = theta_H/(2m) as an independent check on
                        mu, D and H; it is not used in the inversion.
                        (If the recombination rate is absent 1 cM/Mb is
                        assumed; if D is absent, gamma = 0.)
 OBSERVED_ROH_TABLE.csv the observed ROH length spectrum, as produced by
                        nerohstats (nerohstats_v5.cpp), which calls the ROH and
                        tabulates the fraction of the genome they cover by
                        length class.  Two tab-separated columns with a header:
                        the class mark of the length bin (cM) and the fraction
                        of the genome covered by ROH of that class.  Rows with
                        an empty first field are ignored (trailing checksum).
                        Both end classes are ordinary bins: neither pools the
                        remaining ROH down to 0 nor up to infinity.

 OUTPUT
 ------
 OUTPUT.txt             two columns, Generation and Ne_estimate.
 Ne_trajectory.pdf      the inferred trajectory (and the true one if the
 Ne_trajectory.png      parameter file names a trajectory file).

 REQUIREMENTS
 ------------
 Python 3.6 or later

 USAGE
 -----
     python Ne_trajectory.py
     python Ne_trajectory.py --step 50         # epochs of 50 generations
     python Ne_trajectory.py --grid geometric # resolution-matched epochs
===============================================================================
"""
import argparse
import math
import os
import sys

import numpy as np

EPSI3 = math.exp(1.5 - 0.5772156649015329)     # e^{psi(3)} = 2.516289...
BP_PER_MB = 1.0e6


# =============================================================================
# 1. INPUT
# =============================================================================
def read_params(path):
    """Reads SIMULATED_PARAMS.txt: every value sits on the line following the
    comment that describes it.  Values are matched to keys by keyword, and
    units are taken from the comment when it states them."""
    if not os.path.exists(path):
        sys.exit(f"parameter file not found: {path}")
    label, raw = None, []
    for line in open(path, encoding="utf-8"):
        s_ = line.strip()
        if not s_:
            continue
        if s_.startswith("#"):
            label = s_.lstrip("#").strip().lower()
        elif label is not None:
            raw.append((label, s_))
            label = None

    def find(*keys, exclude=()):
        """(value, comment) of the first entry whose comment matches."""
        for k, v in raw:
            if any(c in k for c in keys) and not any(e in k for e in exclude):
                return v, k
        return None, None

    p = {}
    p["trajectory"] = find("trajectory")[0]
    p["mu"] = float(find("mutation rate")[0] or 0.0)
    p["gc"] = float(find("gene conversion")[0] or 0.0)
    nch = find("chromosome number")[0]
    p["nchrom"] = int(float(nch)) if nch else None
    siz = find("chromosome size")[0]
    p["chromsize"] = float(siz) if siz else None

    # --- recombination: cM/Mb, or por bp when stated or when tiny ----------
    rec, com = find("recombination")
    if rec is None:
        p["rec_cM_Mb"] = 1.0
        p["rec_source"] = "ABSENT from the file: 1 cM/Mb assumed"
    else:
        v = float(rec)
        por_pb = ("per bp" in com or "per site" in com or
                  ("cm/mb" not in com and v < 1e-4))
        if por_pb:
            p["rec_cM_Mb"] = v * BP_PER_MB * 100.0
            p["rec_source"] = f"{v:g} per bp  ->  {p['rec_cM_Mb']:g} cM/Mb"
        else:
            p["rec_cM_Mb"] = v
            p["rec_source"] = f"{v:g} cM/Mb"

    # --- D and H ------------------------------------------------------------
    # The theory needs only the ratio D/H: with 1/D segregating sites per Mb,
    # each heterozygous in an individual with probability H, that individual
    # carries H/D heterozygous sites per Mb, so D/H is the mean distance
    # between HETEROZYGOUS sites and gamma = c * (D/H).
    Dv, Dcom = find("distance between", "marker spacing", "snp spacing",
                     "site spacing")
    Hv, _ = find("heterozygosity", exclude=("distance", "spacing"))
    p["D_bp"] = p["H"] = p["DH"] = None
    if Dv is not None:
        v = float(Dv)
        flag = ("(mb)" in Dcom) or ("in mb" in Dcom) or (v < 1e-2)
        p["D_bp"] = v * BP_PER_MB if flag else v
    if Hv is not None:
        h = float(Hv)
        if 0.0 < h <= 1.0:
            p["H"] = h

    if p["D_bp"] is None:
        p["dh_source"] = ("ABSENT from the file: gamma = 0 assumed (no "
                          "boundary displacement)")
    elif p["H"] is None:
        p["DH"] = p["D_bp"] / BP_PER_MB
        p["dh_source"] = (f"no H in the file: D = {p['D_bp']:g} bp taken AS "
                          f"the distance between heterozygous sites")
    else:
        p["DH"] = (p["D_bp"] / p["H"]) / BP_PER_MB
        p["dh_source"] = (f"D = {p['D_bp']:g} bp between polymorphic sites, "
                          f"H = {p['H']:g}  ->  D/H = "
                          f"{p['DH'] * BP_PER_MB:.1f} bp between het sites")
    return p


def read_table(path, rec_cM_Mb):
    """Reads OBSERVED_ROH_TABLE.csv.  Returns (xc, masa, lo, hi) in Mb plus the
    checksum row if present.  Class boundaries are the midpoints between
    consecutive class marks, with the two outer edges mirrored: neither end
    class pools the remaining ROH."""
    if not os.path.exists(path):
        sys.exit(f"ROH table not found: {path}")
    xc, masa, extra = [], [], []
    for i, line in enumerate(open(path, encoding="utf-8")):
        fields = line.rstrip("\n").replace(",", "\t").split("\t")
        fields = [c.strip() for c in fields]
        if i == 0 or not fields:
            continue
        if not fields[0]:                          # trailing checksum / blanks
            if len(fields) > 1 and fields[1]:
                extra.append(float(fields[1]))
            continue
        xc.append(float(fields[0]))
        masa.append(float(fields[1]))
    xc = np.asarray(xc, float) / rec_cM_Mb         # cM -> Mb
    masa = np.asarray(masa, float)
    marca = 0.5 * (xc[1:] + xc[:-1])
    lo, hi = np.empty_like(xc), np.empty_like(xc)
    hi[:-1] = marca
    hi[-1] = 2 * xc[-1] - marca[-1]
    lo[1:] = marca
    lo[0] = 2 * xc[0] - marca[0]
    return xc, masa, lo, hi, extra


def read_trajectory(path):
    """Reads the true trajectory: two columns, generation and N."""
    if not path or not os.path.exists(path):
        return None, None
    g, n = [], []
    for i, line in enumerate(open(path, encoding="utf-8")):
        c = line.split()
        if len(c) < 2:
            continue
        try:
            gg, nn = float(c[0]), float(c[1])
        except ValueError:
            continue                               # header
        g.append(gg); n.append(nn)
    return np.asarray(g), np.asarray(n)


# =============================================================================
# 2. THEORY
# =============================================================================
def Phi(t, a):
    """Primitive of -t^2 e^{-a t}; Phi(inf; a) = 0."""
    if t is None or not np.isfinite(t):
        return 0.0
    return math.exp(-a * t) * (t * t / a + 2.0 * t / a ** 2 + 2.0 / a ** 3)


def fhat2_epochs(s, N_list, tau_list):
    """Equation A16 differentiated twice, restricted to the listed epochs.

    tau_list holds the upper bound of each epoch.  Returns
    (fhat'' of those epochs, S at the end of the last one, that end time)."""
    tot, Lacc, t0 = 0.0, 0.0, 0.0
    for Nk, tk in zip(N_list, tau_list):
        lk = 1.0 / (2.0 * Nk)
        a = s + lk
        tot += lk * math.exp(-Lacc) * math.exp(lk * t0) * (Phi(t0, a) - Phi(tk, a))
        if tk is None or not np.isfinite(tk):
            return tot, 0.0, np.inf
        Lacc += lk * (tk - t0)
        t0 = tk
    return tot, math.exp(-Lacc), t0


def w_roots(V, sigma=0.0):
    """Roots in (0,1) of the cubic A17, by Cardano.  Smallest w first, i.e.
    largest N first."""
    b2 = -(sigma + 2.0)
    b1 = ((sigma + 1.0) ** 2 + 1.0) / 2.0
    b0 = -V
    p = b1 - b2 * b2 / 3.0
    q = 2.0 * b2 ** 3 / 27.0 - b2 * b1 / 3.0 + b0
    disc = 4.0 * p ** 3 + 27.0 * q * q
    out = []
    if p < 0.0 and disc < 0.0:                     # three real roots
        th = math.acos(max(-1.0, min(1.0, (3.0 * q / (2.0 * p))
                                     * math.sqrt(-3.0 / p))))
        for k in (0, 1, 2):
            out.append(2.0 * math.sqrt(-p / 3.0)
                       * math.cos(th / 3.0 - 2.0 * math.pi * k / 3.0) - b2 / 3.0)
    elif p < 0.0:                                  # one real root, cosh
        sg = 1.0 if q >= 0.0 else -1.0
        out.append(-2.0 * sg * math.sqrt(-p / 3.0)
                   * math.cosh(math.acosh(-3.0 * abs(q) / (2.0 * p)
                                          * math.sqrt(-3.0 / p)) / 3.0)
                   - b2 / 3.0)
    else:                                          # one real root, sinh
        sg = 1.0 if q >= 0.0 else -1.0
        out.append(-2.0 * sg * math.sqrt(p / 3.0)
                   * math.sinh(math.asinh(3.0 * abs(q) / (2.0 * p)
                                          * math.sqrt(3.0 / p)) / 3.0)
                   - b2 / 3.0)
    return sorted(w for w in out if 1e-13 < w < 1.0)


def N_from_w(w, x, B):
    """Equation A18."""
    return (1.0 - w) / (4.0 * B * x * w)


# =============================================================================
# 3. INVERSION
# =============================================================================
class Spectrum:
    """Observed coverage density P(x), per Mb, interpolated in log-log."""

    def __init__(self, xc, masa, lo, hi):
        self.xm = np.sqrt(lo * hi)
        self.P = masa / (hi - lo)
        ok = self.P > 0
        self.lx = np.log(self.xm[ok])
        self.lP = np.log(self.P[ok])
        self.xmin, self.xmax = float(lo[0]), float(hi[-1])

    def __call__(self, x):
        return float(np.exp(np.interp(np.log(x), self.lx, self.lP)))


def ET_piecewise(N_list, tau_list):
    """E[T] = integral of S(t) for a piecewise-constant history."""
    S, ET, t0 = 1.0, 0.0, 0.0
    for Nk, tk in zip(N_list, tau_list):
        lam = 1.0 / (2.0 * Nk)
        if tk is None or not np.isfinite(tk):
            return ET + S * 2.0 * Nk
        ET += S * 2.0 * Nk * (1.0 - math.exp(-lam * (tk - t0)))
        S *= math.exp(-lam * (tk - t0))
        t0 = tk
    return ET + S * 2.0 * N_list[-1]


def x_probe(t, B, g):
    """Tract length in Mb that probes generation t (inverse of Eq. A7)."""
    return EPSI3 / (2.0 * B * t) + 2.0 * g / B


def t_from_x(x, B, g):
    """Generation probed by a tract of length x Mb."""
    return EPSI3 / (2.0 * B * max(x - 2.0 * g / B, 1e-12))


def uniform_grid(step, t_deep):
    """Epoch upper bounds every `step` generations up to t_deep."""
    n = max(1, int(math.ceil(t_deep / step)))
    return np.arange(1, n + 1, dtype=float) * step


def geometric_grid(t_shallow, t_deep, factor=3.0):
    """Epoch upper bounds spaced geometrically, the resolution limit of the
    t^2 e^{-st} kernel (CV = 1/sqrt(3) always, so the 90 % window spans a
    factor 7.7 in t whatever the probe depth)."""
    bs, t = [], t_shallow * factor
    while t < t_deep:
        bs.append(t)
        t *= factor
    bs.append(t_deep)
    return np.asarray(bs, float)


def peel(spec, tau, B, g, tol=0.30, verbose=True):
    """Sequential closed-form inversion, longest tracts first.

    `tol` is the smallest fraction of the observed signal that an epoch must
    still account for.  Once the epochs already peeled explain more than
    (1 - tol) of the coverage at the probing length, the remainder is a small
    difference between large numbers: N_k stops being identifiable and the
    estimates run away.  The peel stops there rather than printing nonsense.

    Returns a list of rows (k, t_start, t_end, t_probe, x, clipped, sigma, V,
    roots, N) for the epochs that could be read."""
    N_est, tau_prev, rows = [], [], []
    t0 = 0.0
    for k, tk in enumerate(tau):
        t_c = math.sqrt(max(t0, tau[0] / 3.0) * tk)     # geometric mid-epoch
        x = x_probe(t_c, B, g)
        clipped = ""
        if x > spec.xmax:
            x, clipped = spec.xmax, "<"                # longest bin available
        elif x < spec.xmin:
            if verbose:
                print(f"  epoch {t0:.0f}-{tk:.0f}: the length that probes it "
                      f"({x:.4f} Mb) falls below the shortest class "
                      f"({spec.xmin:.4f} Mb) -- the peel stops here")
            break
        P = spec(x)
        s = max(2.0 * B * x - 4.0 * g, 1e-14)
        G_prev, S_prev, t_fin_prev = fhat2_epochs(s, N_est, tau_prev)
        if S_prev <= 0.0:
            break
        G_obs = P / (4.0 * B * B * x)
        signal = (G_obs - G_prev) / G_obs if G_obs > 0 else 0.0
        if N_est and signal < tol:
            if verbose:
                print(f"  epoch {t0:.0f}-{tk:.0f}: the epochs already peeled "
                      f"account for {100*(1-signal):.1f} % of the coverage "
                      f"observed at x = {x:.4f} Mb.\n  What is left "
                      f"({100*signal:.1f} %) is a difference between large "
                      f"numbers and N is no longer identifiable:\n  the peel "
                      f"stops here (threshold --tol = {100*tol:.0f} %)")
            break
        R = (G_obs - G_prev) / (S_prev * math.exp(-s * t_fin_prev))
        sigma, V = s * t_fin_prev, R * s * s / 2.0
        ws = w_roots(V, sigma)
        if not ws:
            if verbose:
                print(f"  epoch {t0:.0f}-{tk:.0f}: the cubic A17 has no root "
                      f"in (0,1) (V = {V:.4g}, sigma = {sigma:.3g}); the epochs "
                      f"already peeled absorb the whole coverage observed at "
                      f"that length -- the peel stops here")
            break
        cands = [N_from_w(w, x, B) for w in ws]
        if not N_est:
            Nk = max(cands)          # recent epoch: the branch with the larger N
        else:
            Nk = min(cands, key=lambda n: abs(math.log(n / N_est[-1])))
        rows.append((k, t0, tk, t_c, x, clipped, sigma, V, cands, Nk,
                      signal))
        N_est.append(Nk)
        tau_prev.append(tk)
        t0 = tk
    return rows


# =============================================================================
# 4. OUTPUT
# =============================================================================
def write_output(rows, path="OUTPUT.txt"):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Generation\tNe_estimate\n")
        for r in rows:
            t0, Nk = r[1], r[9]
            f.write(f"{t0:.0f}\t{Nk:.2f}\n")
    return path


def make_figure(rows, tau, gv, nv, B, g, spec, pars, ref=None,
           base="Ne_trajectory"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.linewidth": .8, "legend.frameon": False,
                         "figure.dpi": 170, "savefig.dpi": 400})
    C_EST, C_TRUE = "#c1440e", "#1b3a5c"

    t_start = np.array([r[1] for r in rows])
    t_end = np.array([r[2] for r in rows])
    Ne = np.array([r[9] for r in rows])

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    if gv is not None and len(gv):
        gg = np.r_[gv, gv[-1] * 1.0]
        nn = np.r_[nv, nv[-1]]
        ax.step(gg, nn, where="post", color= C_TRUE, lw=1.6, alpha=.6,
                label="simulated N(t)")

    edges = np.r_[t_start, t_end[-1]]
    ax.step(np.repeat(edges, 2)[1:-1], np.repeat(Ne, 2), where="pre",
            color=C_EST, lw=2.0, label="inferred from the ROH spectrum")

    t_lo = t_from_x(spec.xmax, B, g)
    t_hi = t_from_x(spec.xmin, B, g)
    ax.axvspan(0, t_lo, color="0.90", lw=0, zorder=0)
    ax.axvspan(t_hi, 1e6, color="0.90", lw=0, zorder=0)
    if ref:
        ti = np.array([r[1] for r in ref]); tf = np.array([r[2] for r in ref])
        nr = np.array([r[9] for r in ref])
        bd = np.r_[ti, tf[-1]]
        ax.step(np.repeat(bd, 2)[1:-1], np.repeat(nr, 2), where="pre",
                color="#0f6e5c", lw=1.2, ls="--",
                label="resolution-matched epochs")

    ax.set_yscale("log")                       # tiempo lineal, N logaritmico
    ax.set_xlim(0.0, float(t_end[-1]))         # hasta la ultima estima
    lo_y = min(Ne.min(), nv.min() if nv is not None else Ne.min()) * 0.6
    hi_y = max(Ne.max(), nv.max() if nv is not None else Ne.max()) * 1.8
    ax.set_ylim(lo_y, hi_y)
    ax.set_xlabel("generations before the present, $t$")
    ax.set_ylabel("effective population size $N$")
    # ax.set_title("N(t) from the ROH length spectrum (Equations A17-A18)",
    #              loc="left", fontsize=10, pad=7)
    # ax.legend(fontsize=8, loc="lower left", borderpad=.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    txt = (f"$B = c + m = {B:.4f}$ Mb$^{{-1}}$   "
           f"($c = {pars['rec_cM_Mb']/100:.4f}$, $m = {(pars['mu']+pars['gc'])*BP_PER_MB:.4f}$)   "
           f"$\\gamma = {pars['gamma']:.3g}$")
    ax.text(0.985, 0.975, txt, transform=ax.transAxes, fontsize=7,
            color="0.3", ha="right", va="top")

    fig.tight_layout(pad=.6)
    for ext in ("pdf", "png"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    return f"{base}.pdf"


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="N(t) from the observed ROH length spectrum "
                    "(From_IBD_to_ROH_v5, Appendix Section 5)")
    ap.add_argument("--params", default="SIMULATED_PARAMS.txt")
    ap.add_argument("--table", default="OBSERVED_ROH_TABLE.csv")
    ap.add_argument("--output", default="OUTPUT.txt")
    ap.add_argument("--grid", choices=("uniform", "geometric"),
                    default="uniform")
    ap.add_argument("--step", type=float, default=25.0,
                    help="epoch width in generations (--grid uniform)")
    ap.add_argument("--factor", type=float, default=3.0,
                    help="geometric spacing (--grid geometric)")
    ap.add_argument("--tol", type=float, default=0.30,
                    help="minimum fraction of the signal an epoch must\n                          still explain for the peel to continue")
    ap.add_argument("--reference", action="store_true",
                    help="overlay the resolution-matched (geometric) "
                         "reading as a reference")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    pars = read_params(args.params)
    xc, masa, lo, hi, extra = read_table(args.table, pars["rec_cM_Mb"])
    spec = Spectrum(xc, masa, lo, hi)

    c = pars["rec_cM_Mb"] / 100.0
    m = (pars["mu"] + pars["gc"]) * BP_PER_MB
    B = c + m
    gamma = 0.0 if pars["DH"] is None else c * pars["DH"]
    g = -math.log(1.0 - gamma) if gamma > 0 else 0.0
    pars["gamma"] = gamma

    t_lo, t_hi = t_from_x(spec.xmax, B, g), t_from_x(spec.xmin, B, g)

    print("=" * 74)
    print("Input")
    print("=" * 74)
    print(f"  parameter file  : {args.params}")
    print(f"  ROH table       : {args.table}   ({len(xc)} classes)")
    print(f"  mutation        : {pars['mu']:g} per site per generation")
    print(f"  gene conversion : {pars['gc']:g} per site per generation")
    print(f"  recombination   : {pars['rec_source']}")
    print(f"  D/H             : {pars['dh_source']}")
    if pars["nchrom"] and pars["chromsize"]:
        print(f"  genome          : {pars['nchrom']} x "
              f"{pars['chromsize']/BP_PER_MB:.0f} Mb = "
              f"{pars['nchrom']*pars['chromsize']/BP_PER_MB:.0f} Mb")
    print(f"  -> c = {c:.4f}/Mb, m = {m:.4f}/Mb, B = c + m = {B:.4f}/Mb")
    print(f"  -> gamma = c D/H = {gamma:.4g}, g = {g:.4g}, "
          f"length shift 2g/B = {2*g/B:.5f} Mb")
    print(f"  classes         : {lo[0]:.4f} to {hi[-1]:.4f} Mb "
          f"({lo[0]*pars['rec_cM_Mb']:.4f} to {hi[-1]*pars['rec_cM_Mb']:.4f} cM)")
    print(f"  total coverage  : {masa.sum():.9f}", end="")
    if extra:
        d = abs(masa.sum() - extra[-1])
        verdict = "matches" if d < 1e-9 else "DIFFERS by %.2e" % d
        print(f"   (checksum row in the file: {extra[-1]:.9f}, "
              f"{verdict})")
    else:
        print()
    print(f"  window with length information: "
          f"t = {t_lo:.0f} to {t_hi:.0f} generations")

    gv, nv = read_trajectory(pars["trajectory"])
    if pars["DH"]:
        theta = 1.0 / pars["DH"]                 # H/D het sites per Mb
        ET_het = theta / (2.0 * m)
        print("\n  heterozygosity check (not used in the inversion, "
              "reported only):")
        print(f"    theta_H = H/D = {theta:.1f} het sites per Mb  ->  "
              f"E[T] = theta_H/(2m) = {ET_het:.0f} generations")
        if gv is not None:
            ET_sim = ET_piecewise(list(nv), list(gv[1:]) + [np.inf])
            print(f"    E[T] of the simulated trajectory = {ET_sim:.0f}  "
                  f"({100*(ET_het/ET_sim - 1):+.1f} %)")

    if args.grid == "uniform":
        tau = uniform_grid(args.step, t_hi)
        print(f"\n  grid: epochs of {args.step:.0f} generations "
              f"({len(tau)} up to t = {tau[-1]:.0f})")
        print(f"  WARNING: the kernel t^2 e^(-st) of Equation A14 has "
              f"CV = 1/sqrt(3) whatever s,\n  so its central 90 % window "
              f"always spans a factor 7.7 in t. Epochs of {args.step:.0f}"
              f"\n  generations are well below that: consecutive readings are "
              f"strongly correlated.\n  --grid geometric uses boundaries "
              f"matched to the resolution.")
    else:
        tau = geometric_grid(t_lo, t_hi, args.factor)
        print(f"\n  grid: geometric, factor {args.factor:g} "
              f"({len(tau)} epochs between {t_lo:.0f} and {t_hi:.0f})")

    print("\n" + "=" * 74)
    print("Direct per-epoch inversion (Equations A17-A18; closed form)")
    print("=" * 74)
    rows = peel(spec, tau, B, g, tol=args.tol)
    if not rows:
        sys.exit("no epoch could be read")

    print("\n    epoch (gen)    t probe   x (Mb)    sigma         V"
          "     roots of A18        N   signal")
    for (k, t0, tk, t_c, x, rec, sigma, V, cands, Nk, signal) in rows:
        cs = ", ".join(f"{v:.0f}" for v in cands)
        print(f" {t0:7.0f}-{tk:<7.0f} {t_c:8.1f} {x:8.4f}{rec:1s} "
              f"{sigma:8.3f} {V:10.6f} {cs:>17}  {Nk:9.0f} {100*signal:6.1f} %")
    print("  'signal' is the fraction of the coverage observed at that "
          "length which the earlier\n  epochs do NOT account for: it is what "
          "is left to determine this one. The smaller\n  it is, the less "
          "identifiable N becomes.")
    if any(r[5] for r in rows):
        print("  (<) the length that would probe that epoch exceeds the "
              "longest class observed;\n      it was read at the longest class "
              "available, so the value mixes that\n      interval with the "
              "immediately earlier past")

    out_path = write_output(rows, args.output)
    print(f"\n  written  {out_path}  ({len(rows)} epochs)")

    if gv is None and pars["trajectory"]:
        print(f"  ({pars['trajectory']} not found; the figure goes without "
              f"the true trajectory)")
    if not args.no_figure:
        ref = None
        if args.reference:
            ref = peel(spec, geometric_grid(t_lo, t_hi, args.factor),
                        B, g, tol=args.tol, verbose=False)
        f = make_figure(rows, tau, gv, nv, B, g, spec, pars, ref=ref)
        print(f"  written  {f}  and its .png")


if __name__ == "__main__":
    main()
