#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 12:43:19 2021

@author: enrique

===============================================================================
 PDF_IBD_tracts_BS.py  --  Computes and plots the probability density function of IBD tract
 lengths with deleterious mutations, neutral mutations and gene conversions. 
===============================================================================

 REQUIREMENTS
 ------------
 Python 3.6 or later

 USAGE
 -----
 python PDF_IBD_tracts_BS.py or python3 PDF_IBD_tracts_BS.py

===============================================================================

"""

import math

import matplotlib.pyplot as plt
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Auxiliary functions used as integrands by scipy.integrate.quad
# ---------------------------------------------------------------------------

def Q2_t_ratchet(x):
    """Squared Qc_t summed over ii generations (finite sum, t = 1..tt)."""
    global tt, UssV, Nhap
    c = (1 - math.exp(-2 * x)) / 2
    a = (1 - UssV) * (1 - c)
    ff = (1 - pow(a, (tt + 1))) / (1 - a)
    return ff * ff


def Q2ratchetexact(x):
    """Squared Qc summed over an infinite number of generations."""
    global UssV, Nhap
    c = (1 - math.exp(-2 * x)) / 2
    ff = 1 / (c + (1 - c) * UssV)
    return ff * ff


# ---------------------------------------------------------------------------
# INPUT
# ===========================================================================
Ndip   = 1000     # Diploid population size
s      = 0.02     # Deleterious effect in heterozygotes (twice in homozygotes)
U      = 1.0      # Deleterious mutation rate per chromosome
m      = 2.5      # Rate of neutral mutation and gene conversion per Morgan
# ===========================================================================
L      = 1      # Chromosome size in Morgans
basexM = 0.0      # Min X value to plot in Morgans
topexM = 0.01     # Max X value to plot in Morgans
h      = (topexM - basexM) / 1000  # Window size in Morgans
OUTFILE = "PDF_IBD_BS_N" + str(Ndip) + '_m' + str(m) + '_s' + str(s) + '_U' + str(U)
# ===========================================================================

# ---------------------------------------------------------------------------
# Derived quantities and accumulators
# ---------------------------------------------------------------------------
topey = 0
window_cM = h * 100
mucM      = m / 100
Nhap      = int(Ndip * 2)
maxMorgan = int(100 / window_cM + 0.5)   # one Morgan is split into maxMorgan segments

basex = basexM * 100 - window_cM
if basex <= 0:
    basex = window_cM                    # minimum x value (cM) shown in the plot
topex = topexM * 100                     # maximum x value (cM) shown in the plot
maxix = int(topex / window_cM + 0.5) + 1 # only the first maxix INTEGER bins are inspected
minix = int(basex / window_cM + 0.5) - 1

xx       = [(i+0.5) / maxMorgan for i in range(maxix)]   # bin centres in Morgans
xxcM     = [xx[i] * 100      for i in range(maxix)]  # bin centres in cM
Px       = [0 for i in range(maxix)]  # accumulator of Px for t = 0..inf, per ten-thousandth of a Morgan

aa = (1 - s) * (1 - 1 / 2 / Ndip)    # reduction due to selection and to sampling

t     = Ndip * 50
stept = 1
if t > 10000:
    stept = int((t - 5000) / 1000)
if stept < 1: stept = 1

print("Nehap_nosel:", Nhap, "   Uhap:", U, "   s:", s, "   L:", L)

# ---------------------------------------------------------------------------
# Ne with ratchet using equation 3 of SC16
# ---------------------------------------------------------------------------
Nehap = Nhap
for i in range(1, 2000):
    try:
        T = (math.exp(2 * s * Nehap) - 1) / (2 * U * s * Nehap) # eq 5 of SC16
    except:
        T = 999999999999999
    Ne1   = Nhap * math.exp(-2 * U * (1 - 1 / (U * T)) ** 3 / (2 * s + L)) # eq 3 of SC16 
    Nehap = (0.1 * Ne1 + 0.9 * Nehap)
    if Nehap < Ne1: break
Nedip  = Nehap / 2
Vantes = U * s
V      = Vantes - s / T
Veq3   = V
print("Nehap_asint_ratchet_eq3:", f"{Nehap:.2f}", "   log(T):", f"{math.log10(T):.2f}")
print("NeDIP_asint_ratchet_eq3:", f"{Nedip:.2f}")

# ---------------------------------------------------------------------------
# Ne with ratchet using the exact formula of Q (infinite summation) in SC16
# ---------------------------------------------------------------------------
Ne    = 2
incre = Nehap / 100
for i in range(0, 200):
    try:
        T = (math.exp(2 * s * Ne) - 1) / (2 * U * s * Ne)
    except:
        Ne     -= incre
        T       = (math.exp(2 * s * Ne) - 1) / (2 * U * s * Ne)
        xNehap  = Ne
        break
    Vantes = U * s
    V      = Vantes - s / T
    UssV   = U * s * s / V
    if L > 0:
        rest2, err2 = quad(Q2ratchetexact, 0, L / 2)
        Qcuad = rest2 / (L / 2)
    else:
        c     = 0
        ff    = 1 / (c + (1 - c) * UssV)
        rest2 = ff * ff
        Qcuad = rest2
    xNehap = Nhap * math.exp(-V * Qcuad)
    if (xNehap > Ne):
        if incre < 0:
            incre = -incre / 10
    else:
        if incre > 0:
            incre = -incre / 10
    Ne += incre
xNedip = xNehap / 2
print("Nehap_asint_ratchet_exact:", f"{xNehap:.2f}", "   log(T):", f"{math.log10(T):.2f}")
print("NeDIP_asint_ratchet_exact:", f"{xNedip:.2f}")
Nehap_asint_exact = xNehap
Vexact            = V
# lineasal += str(Nehap_asint_exact) + "\t"

# ---------------------------------------------------------------------------
# Distribution of IBD tract sizes, generation by generation
# ---------------------------------------------------------------------------
inf  = 0
sup  = L / 2
suma = 0
prod = 1
UssV = U * s * s / V

# tt iterates over generations
for tt in range(0, 10000, 1):
    if L > 0:
        rest2, err2 = quad(Q2_t_ratchet, inf, sup)
        Qcuad = rest2 / sup
    else:
        c     = 0
        a     = (1 - UssV) * (1 - c)
        ff    = (1 - pow(a, (tt + 1))) / (1 - a)
        rest2 = ff * ff
        Qcuad = rest2
    Nehap_tt = Nhap * math.exp(-Qcuad * V)
    if Nehap_tt < 1: break
    aa = 4 * tt * tt * (1 + m) * (1 + m) * (window_cM / 100) * prod / Nehap_tt
    bb = -2 * tt * (1 + m)
    for ix in range(minix, maxix):
        Px[ix] += aa * xx[ix] * math.exp(bb * xx[ix])
    prod   *= (1 - 1 / (Nehap_tt))
    prodant = prod
    suma   += prod

if (t > 10000):
    for tt in range(10000, t, stept):
        if L > 0:
            rest2, err2 = quad(Q2_t_ratchet, inf, sup)
            Qcuad = rest2 / sup
        else:
            c     = 0
            a     = (1 - UssV) * (1 - c)
            ff    = (1 - pow(a, (tt + 1))) / (1 - a)
            rest2 = ff * ff
            Qcuad = rest2
        Nehap_tt = Nhap * math.exp(-Qcuad * V)
        if Nehap_tt < 1: break
        prod *= pow(1 - 1 / (Nehap_tt), stept)
        suma += (prod + prodant) / 2 * stept
        aa = 4 * tt * tt * (1 + m) * (1 + m) * (window_cM / 100) * prod / Nehap_tt * stept
        bb = -2 * tt * (1 + m)
        for ix in range(minix, maxix):
            Px[ix] += aa * xx[ix] * math.exp(bb * xx[ix])

        prodant = prod

Nehap_het_ratchet = suma   # sum[(1-1/Nehap_tt)^i, i = 0..inf] = NeH_hap
Nedip_het_ratchet = Nehap_het_ratchet / 2

# Find the maximum y value to scale the plot
for ix in range(minix, maxix):
    if topey < Px[ix]:
        topey = Px[ix]

print("Nehap_asint_ratchet_2:", f"{Nehap_tt:.2f}")
print("NeDIP_asint_ratchet_2:", f"{Nehap_tt / 2:.2f}")
print("Nehap_het_ratchet:",     f"{Nehap_het_ratchet:.2f}")
print("NeDIP_het_ratchet:",     f"{Nedip_het_ratchet:.2f}")

# ---------------------------------------------------------------------------
# variance components
# ---------------------------------------------------------------------------
Vwh1 = Nehap_asint_exact * s * s / T
Vwh2 = Nehap_het_ratchet * s * s / T
print("Vw, Vw_h_asin, Vw_LE_asin: ", f"{Vexact:.8f}", "  ", f"{Vwh1:.8f}", "  ", f"{Vexact + Vwh1:.8f}")
print("Vw, Vw_h_het, Vw_LE_het: ",  f"{Vexact:.8f}", "  ", f"{Vwh2:.8f}", "  ", f"{Vexact + Vwh2:.8f}")

try:
    T1 = ((math.exp(2 * s * xNehap) - 1) / (2 * U * s * xNehap))
except:
    T1 = 999999999999999
try:
    T2 = ((math.exp(2 * s * Nehap_het_ratchet) - 1) / (2 * U * s * Nehap_het_ratchet))
except:
    T2 = 999999999999999
print(T1)
print(T2)

# ---------------------------------------------------------------------------
# Plot the IBD tract-length distribution
# ---------------------------------------------------------------------------
plt.ylim(0, topey * 1.1)
plt.yticks(fontsize=8)
plt.xlabel('IBD tract length (cM)', fontsize=12, labelpad=0)
plt.ylabel("$frequency_{\\ (" + f"{window_cM:.6f}" + "\\ cM\\ interval)}$", fontsize=11, labelpad=1)

plt.xlim(basex , topex)
plt.title('PDF of lengths or genome proportion that is IBD due to tract lengths ($N$ = ' + str(int(Ndip)) + ")", fontsize=8)
plt.plot(xxcM, Px, color="firebrick", lw=2, alpha=.7)
plt.text(basex + (topex - basex) / 2.3, topey / 10 * 10, "$s$ = "                    + f"{s:.3f}", fontsize=10, color='firebrick')
plt.text(basex + (topex - basex) / 2.3, topey / 10 *  8, "chr size (Morgans)  = "    + f"{L:.3f}", fontsize=10, color='firebrick')
plt.text(basex + (topex - basex) / 2.3, topey / 10 *  9, "deleterious mut/chr = "    + f"{U:.3f}", fontsize=10, color='firebrick')
plt.text(basex + (topex - basex) / 2.3, topey / 10 *  7, "mut & gene conv/Morgan = " + f"{m:.3f}", fontsize=10, color='firebrick')
plt.savefig(OUTFILE + '.jpg', dpi=600)
plt.savefig(OUTFILE + '.pdf', dpi=600)
plt.show()

# salida.close()
