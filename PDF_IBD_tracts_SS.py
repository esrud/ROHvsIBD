#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 12:43:19 2021

@author: enrique

===============================================================================
PDF_IBD_tracts_SS.py  --  Computes and plots the probability density function of IBD tracts
lengths with a selective sweep, neutral mutations and gene conversions. 
===============================================================================

 REQUIREMENTS
 ------------
 Python 3.6 or later

 USAGE
 -----
 python PDF_IBD_tracts_BS.py or python3 PDF_IBD_tracts_SS.py

===============================================================================

"""

import math
import sys
import matplotlib.pyplot as plt

# ===========================================================================
# INPUT
# ===========================================================================
Ndip   = 15000    # Diploid population size
alphaHet  = 0.05     # Favourable effect in heterozygotes (only positive values)
alphaHom  = 0.05     # Favourable effect in Homozygotes (only positive values)
rec    = 0.0020    # Recombination frequency between selected and neutral sites
fa     = 0.9      # Frequency of favourable allele at the sampling generation
gaf    = 0        # generation of sampling after fixation (valid if fa>=0.99)
m      = 2.5      # Rate of neutral mutation and gene conversion per Morgan
basexM = 0.00      # Min X value to plot in Morgans
topexM = 0.006     # Max X value to plot in Morgans
# ===========================================================================
h      = (topexM - basexM) / 30  # Window size in Morgans
if fa > 0.99: fa = 0.99
if fa < 0.99: gaf = 0
OUTFILE = "PDF_IBD_SS_N" + str(Ndip) + '_m' + str(m) + '_alphaHet' + str(alphaHet)+ '_alphaHom' + str(alphaHom)+ '_rec' + str(rec)
# ===========================================================================
if (alphaHet <= 0) or (alphaHom<alphaHet):
    sys.exit()
# ---------------------------------------------------------------------------
# Derived quantities and accumulators
# ---------------------------------------------------------------------------
topey = 0
topex = topexM * 100                     # maximum x value (cM) shown in the plot
window_cM = h * 100
mucM      = m / 100
Nhap      = int(Ndip * 2)
nmaxMorgan = int(topex / window_cM + 0.5)   # one Morgan is split into nmaxMorgan segments

basex = basexM * 100 # minimum x value (cM) shown in the plot

maxix = int(topex / window_cM + 0.5) 
minix = int(basex / window_cM + 0.5)

xxcM     = [(i+0.5) * window_cM for i in range(nmaxMorgan)]  # bin centres in cM
xx       = [xxcM[i] / 100  for i in range(nmaxMorgan)]   # bin centres in Morgans

Px_right = [0 for i in range(nmaxMorgan)]  # accumulator of Px_right for t = 0..inf
Px_left = [0 for i in range(nmaxMorgan)]  # accumulator of Px_left for t = 0..inf

t     = Ndip * 50
stept = 1
if t > 10000:
    stept = int((t - 5000) / 1000)
if stept < 1: stept = 1

print("Nehap_nosel:", Nhap, "   alphaHet:", alphaHet)

# ---------------------------------------------------------------------------
# Gene frequency of the favourable allele over generations
# ---------------------------------------------------------------------------
q        = 1 / Nhap
fgen     = [q]
ngen     = [0]
ng       = 1
while (q < fa) or (fa >= 0.99 and gaf > 0):
    DD=q*q*(1+alphaHom)
    HH=2*q*(1-q)*(1+alphaHet)
    RR=(1-q)*(1-q)
    MM=DD+HH+RR
    qq=(DD+HH/2)/MM
    increq=qq-q
    if (increq < 1 / Nhap) and q < 0.5:
        q += 1 / Nhap
    else:
        q = qq
    if q > 0.9999: q = 0.9999
    fgen.append(q)
    ng += 1
    ngen.append(ng)
    if q >= 0.99: gaf -= 1

plt.ylim(0, 1)
plt.xlabel('generations', fontsize=12, labelpad=0)
plt.ylabel("favourable allele frequency", fontsize=11, labelpad=1)

plt.xlim(0, ng)
basx = ng
basy = 0
topy= 0.9

plt.title("gene frequencies over the considered time span ($N$ = " + str(int(Ndip)) + ")", fontsize=9)
plt.plot(ngen, fgen, color="darkblue", lw=2, alpha=.7)
plt.text(basx / 20, topy / 10 * 10, "\u03b1_Het = "                    + f"{alphaHet:.4f}", fontsize=10, color='darkblue')
plt.text(basx / 20, topy / 10 * 9, "\u03b1_Hom = "                    + f"{alphaHom:.4f}", fontsize=10, color='darkblue')
plt.text(basx / 20, topy / 10 * 8, "rec rate = "    + f"{rec:.4f}", fontsize=10, color='darkblue')
plt.text(basx / 20, topy / 10 * 7, "mut & gene conv/Morgan = " + f"{m:.4f}", fontsize=10, color='darkblue')
plt.savefig(OUTFILE + '_FREQUENCIES.jpg', dpi=600)
plt.savefig(OUTFILE + '_FREQUENCIES.pdf', dpi=600)

plt.show()

# ---------------------------------------------------------------------------
# Ne using equation of Long Term Contributions
# ---------------------------------------------------------------------------
y1gen    = []
y2gen    = []
zgen     = []
for i in range (ng): # Projections from consecutive generations
    q     = fgen[i]
    p1    = 1
    p2    = 1
    z     = 0
    for j in range (i+1,ng): # Accumulation to the frontier
        z += (q*p1)*(1-z)*rec - (q*(1-p1))*z*rec# Correction for prop neutrals in non-carriers 
        # z=0
        p1 = p1 * (1 - (1 - q) * rec * (1 - z))
        p2 = p2 * (1 - (1 - q) * rec * (1 - z))**2+ (p1 - p2) / Nhap / q
        
        DD=q*q*(1+alphaHom)
        HH=2*q*(1-q)*(1+alphaHet)
        RR=(1-q)*(1-q)
        MM=DD+HH+RR
        qq=(DD+HH/2)/MM
        increq=qq-q
        if (increq < 1 / Nhap) and q < 0.5:
            q += 1 / Nhap
        else:
            q = qq

    y1gen.append(p1)
    y2gen.append(p2)
    zgen.append(z)
NeLT_neutralsite = []
qf1=fgen[ng-1]
qf2=fgen[ng-1]**2
maxNe=0
for i in range (ng):
    q0 = fgen[i]
    VLT=y2gen[i]*qf2/q0+(1+y2gen[i]*qf2-2*y1gen[i]*qf1)/(1-q0)-1
    xNe = Ndip / (1 + VLT)
    if xNe>maxNe:maxNe=xNe
    NeLT_neutralsite.append(xNe)


plt.ylim(0, maxNe*1.1)
plt.xlabel('generations', fontsize=12, labelpad=0)
plt.ylabel("$N_e$", fontsize=11, labelpad=1)

plt.xlim(0, ng)
basx = ng
basy = 0
topy= maxNe
plt.title("Ne over the considered time span ($N$ = " + str(int(Ndip)) + ")", fontsize=9)
plt.plot(ngen, NeLT_neutralsite, color="seagreen", lw=2, alpha=.7)
plt.text(basx / 20, topy / 10 * 10, "\u03b1_Het = "                    + f"{alphaHet:.4f}", fontsize=10, color='seagreen')
plt.text(basx / 20, topy / 10 * 9, "\u03b1_Hom = "                    + f"{alphaHom:.4f}", fontsize=10, color='seagreen')
plt.text(basx / 20, topy / 10 * 8, "rec rate = "    + f"{rec:.4f}", fontsize=10, color='seagreen')
plt.text(basx / 20, topy / 10 * 7, "mut & gene conv/Morgan = " + f"{m:.4f}", fontsize=10, color='seagreen')
plt.savefig(OUTFILE + '_Ne.jpg', dpi=600)
plt.savefig(OUTFILE + '_Ne.pdf', dpi=600)

plt.show()
    
# Flip of NeLT_neutralsite. Now, element 0 is the previous generation (the most recent one)
NeLT_neutralsite.reverse()
# tt iterates over generations
prod=1
for tt in range(0, 10000, 1):
    if tt>=ng:
        Nehap_tt = Nhap
    else:
        Nehap_tt = 2 * NeLT_neutralsite[tt]
    if Nehap_tt < 1: break
    aa = 2 * tt * (1 + m) * (window_cM / 100) * prod / Nehap_tt# Only right tracts. AUMENTO DE RESOLUCION DE LA LINEA
    bb = -2 * tt * (1 + m)
    for ix in range(0, nmaxMorgan):
        Px_right[ix] += aa * math.exp(bb * xx[ix]) # Only right tracts
    prod   *= (1 - 1 / (Nehap_tt))
    prodant = prod
if (t > 10000):
    for tt in range(10000, t, stept):
        if tt>=ng:
            Nehap_tt = Nhap
        else:
            Nehap_tt = 2 * NeLT_neutralsite[tt]
        if Nehap_tt < 1: break
        prod *= pow(1 - 1 / (Nehap_tt), stept)
        aa = 2 * tt * (1 + m) * (window_cM / 100) * prod / Nehap_tt * stept
        bb = -2 * tt * (1 + m)
        for ix in range(0, nmaxMorgan):
            Px_right[ix] += aa * math.exp(bb * xx[ix])
        prodant = prod
suma=sum(Px_right)
# Px_right=[x/suma for x in Px_right]

# Find the maximum y value to scale the plot
for ix in range(minix, maxix):
    if topey < Px_right[ix]:
        topey = Px_right[ix]

# ---------------------------------------------------------------------------
# Plot the IBD tract-length distribution
# ---------------------------------------------------------------------------
plt.ylim(0, topey * 1.01)
plt.yticks(fontsize=8)
plt.xlabel('IBD tract length (cM)', fontsize=12, labelpad=0)
plt.ylabel("$frequency_{\\ (" + f"{window_cM:.6f}" + "\\ cM\\ interval)}$", fontsize=11, labelpad=1)

kx = topex / 0.6
ky = topey / 0.52

plt.xlim(basex , topex)
plt.title('PDF of IBD tract lengths on the right side ($N$ = ' + str(int(Ndip)) + ")", fontsize=10)
plt.plot(xxcM, Px_right, color="firebrick", lw=2, alpha=.7)
plt.text((basex + (topex - basex) / 2.3), topey / 15 * 5, "\u03b1_Het = "                    + f"{alphaHet:.4f}", fontsize=10, color='firebrick')
plt.text((basex + (topex - basex) / 2.3), topey / 15 * 4, "\u03b1_Hom = "                    + f"{alphaHet:.4f}", fontsize=10, color='firebrick')
plt.text((basex + (topex - basex) / 2.3), topey / 15 * 3, "rec rate = "    + f"{rec:.4f}", fontsize=10, color='firebrick')
plt.text((basex + (topex - basex) / 2.3), topey / 15 * 2, "mut & gene conv/Morgan = " + f"{m:.4f}", fontsize=10, color='firebrick')

plt.plot([0.2*kx,0.55*kx],[0.24*ky*1.7,0.24*ky*1.7],color="gray", lw=4, alpha=.7)
plt.plot(0.375*kx,0.24*ky*1.7,linestyle='None',mfc="royalblue",marker="s",mec="navy",mew=.7,markersize=7,alpha=0.9)
plt.plot(0.275*kx,0.24*ky*1.7,linestyle='None',mfc="darkviolet",marker="s",mec="navy",mew=.7,markersize=7,alpha=0.9)
plt.text(0.25*kx, 0.217*ky*1.7, "\u2190 left tracts", fontsize=10,color='firebrick')
plt.text(0.39*kx, 0.217*ky*1.7, "right tracts \u2192", fontsize=10, fontweight="bold",color='firebrick')
plt.text(0.375*kx, 0.25*ky*1.7, "neutral\nfocal\nsite", fontsize=10, color='black')
plt.text(0.205*kx, 0.25*ky*1.7, "selected\n        site", fontsize=10, color='black')
plt.text(0.295*kx, 0.25*ky*1.7, "r="+ f"{rec:.3f}", fontsize=9, color='black')

plt.savefig(OUTFILE + '_PDFtracts.jpg', dpi=600)
plt.savefig(OUTFILE + '_PDFtracts.pdf', dpi=600)
plt.show()
