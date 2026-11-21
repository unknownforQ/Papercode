#!/usr/bin/env python3
"""
Reproduce the residual and Mermin A/B figures for the GHZ contract-signing
manuscript.  The script is an illustrative IID/independent-residual model,
not a full device-level simulation or a complete adjudicator implementation.
"""
from pathlib import Path
import csv, json, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, beta

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
DATA = OUT / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

SEED = 20261108
rng = np.random.default_rng(SEED)

# ----------------------- Shared manuscript parameters -----------------------
n = 16383
t = 1000
raw_thresholds = [800, 900, 1000]
e0 = 0.02
a0 = 0.04

# ----------------------- Residual simulation model --------------------------
def binom_cdf_int(r, n, p):
    return float(binom.cdf(int(r), int(n), float(p)))

def residual_pass_prob(e, a, r):
    qx = e + (1 - 2*e)*a
    qz = e
    return (binom_cdf_int(r, n, qx)**2) * (binom_cdf_int(r, n, qz)**2)

def residual_mc(e, a, r, reps, rng):
    qx = e + (1 - 2*e)*a
    qz = e
    kax = rng.binomial(n, qx, reps)
    kbx = rng.binomial(n, qx, reps)
    kaz = rng.binomial(n, qz, reps)
    kbz = rng.binomial(n, qz, reps)
    return float(np.mean((kax <= r) & (kbx <= r) & (kaz <= r) & (kbz <= r)))

# Panel 1: background residual rate e, no late flip.
e_grid = np.linspace(0.0, 0.10, 401)
res_a_rows = []
fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
for r in raw_thresholds:
    y = np.array([residual_pass_prob(e, 0.0, r) for e in e_grid])
    ax.plot(100*e_grid, y, label=fr"$r={r}$")
    for e in np.linspace(0.0, 0.10, 21):
        pth = residual_pass_prob(e, 0.0, r)
        pmc = residual_mc(e, 0.0, r, 20000, rng)
        res_a_rows.append([e, 0.0, r, pth, pmc, 20000])
        ax.scatter(100*e, pmc, s=13)
ax.set(xlabel="Baseline residual probability $e$ (%)",
       ylabel="Four-stream raw-pass probability",
       title="(a) Background noise")
ax.set_ylim(-0.03, 1.03)
ax.grid(True, alpha=.25)
ax.legend(title="Raw threshold")
fig.savefig(FIG/"residual_background.png", dpi=220)
fig.savefig(FIG/"residual_background.pdf")
plt.close(fig)

# Panel 2: late WORK-X report flips, baseline e fixed.
a_grid = np.linspace(0.0, 0.08, 401)
res_b_rows = []
fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
for r in raw_thresholds:
    y = np.array([residual_pass_prob(e0, a, r) for a in a_grid])
    ax.plot(100*a_grid, y, label=fr"$r={r}$")
    for a in np.linspace(0.0, 0.08, 17):
        pth = residual_pass_prob(e0, a, r)
        pmc = residual_mc(e0, a, r, 20000, rng)
        res_b_rows.append([e0, a, r, pth, pmc, 20000])
        ax.scatter(100*a, pmc, s=13)
ax.set(xlabel="WORK-$X$ report-flip probability $a$ (%)",
       ylabel="Four-stream raw-pass probability",
       title="(b) Late WORK-$X$ report flips")
ax.set_ylim(-0.03, 1.03)
ax.grid(True, alpha=.25)
ax.legend(title="Raw threshold")
fig.savefig(FIG/"residual_work_x_flips.png", dpi=220)
fig.savefig(FIG/"residual_work_x_flips.pdf")
plt.close(fig)

# Panel 3: log-scale rare-event view.
fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
a_rare = np.linspace(0.018, 0.075, 350)
for r in raw_thresholds:
    vals = np.array([residual_pass_prob(e0, a, r) for a in a_rare])
    ax.plot(100*a_rare, np.maximum(vals, 1e-14), label=fr"$r={r}$")
# Zero-hit 95% one-sided upper bound, with actual MC checks at selected points.
R_mc = 100000
upper95 = 1 - 0.05**(1/R_mc)
rare_mc_a = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
rare_mc = []
for aa in rare_mc_a:
    pp = residual_pass_prob(e0, aa, 1000)
    ff = residual_mc(e0, aa, 1000, R_mc, rng)
    hits = int(round(ff*R_mc))
    rare_mc.append([aa, pp, ff, hits, R_mc])
    if hits == 0:
        ax.scatter(100*aa, upper95, marker="v", facecolors="none", edgecolors="black", zorder=5)
    else:
        ax.scatter(100*aa, ff, marker="o", color="black", s=22, zorder=5)
ax.axhline(upper95, linestyle=":", label="0 hits in 100,000: 95% upper bound")
ax.set_yscale("log")
ax.set_ylim(1e-14, 1.2)
ax.set(xlabel="WORK-$X$ report-flip probability $a$ (%)",
       ylabel="Four-stream raw-pass probability (log scale)",
       title="(c) Rare-event resolution")
ax.grid(True, which="both", alpha=.25)
ax.legend()
fig.savefig(FIG/"residual_rare_event.png", dpi=220)
fig.savefig(FIG/"residual_rare_event.pdf")
plt.close(fig)
with open(DATA/"residual_rare_mc.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["e","a","r","P_raw_theory","P_raw_MC","hits","repetitions"]); w.writerows([[e0,*row[:1],1000,*row[1:]] for row in rare_mc])

# Panel 4: single X-stream error-count PMF.
qx0 = e0 + (1 - 2*e0)*a0
k = np.arange(700, 1161)
pmf = binom.pmf(k, n, qx0)
fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
ax.plot(k, pmf, linewidth=1.8, label=fr"$K_X\sim\mathrm{{Bin}}({n},{qx0:.4f})$")
ax.fill_between(k[k <= 800], pmf[k <= 800], alpha=.45, label=r"Raw accepted: $K_X\leq800$")
ax.fill_between(k[(k > 800) & (k <= t)], pmf[(k > 800) & (k <= t)],
                alpha=.35, label=fr"Within $t={t}$, rejected at $r=800$")
ax.fill_between(k[k > t], pmf[k > t], alpha=.30, label=fr"Outside radius $t={t}$")
for x, lab in [(800, "raw threshold 800"), (900, "raw threshold 900"), (1000, "correction radius 1000")]:
    ax.axvline(x, linestyle="--", linewidth=1.2, label=lab)
ax.set(xlabel="Residual count in one $X$ stream",
       ylabel="Probability mass",
       title="(d) Raw policy versus correction radius")
ax.grid(True, alpha=.25)
ax.legend(fontsize=8)
fig.savefig(FIG/"residual_count_distribution.png", dpi=220)
fig.savefig(FIG/"residual_count_distribution.pdf")
plt.close(fig)

# Residual overview, 2x2.
fig, axs = plt.subplots(2, 2, figsize=(13.0, 8.2), constrained_layout=True)
for r in raw_thresholds:
    axs[0,0].plot(100*e_grid, [residual_pass_prob(e,0,r) for e in e_grid], label=fr"$r={r}$")
    axs[0,1].plot(100*a_grid, [residual_pass_prob(e0,a,r) for a in a_grid], label=fr"$r={r}$")
    vals = [residual_pass_prob(e0,a,r) for a in a_rare]
    axs[1,0].plot(100*a_rare, np.maximum(vals,1e-14), label=fr"$r={r}$")
axs[0,0].set(title="(a) Background noise", xlabel="$e$ (%)", ylabel="Four-stream raw-pass probability", ylim=(-.03,1.03))
axs[0,1].set(title="(b) WORK-$X$ report flips", xlabel="$a$ (%)", ylabel="Four-stream raw-pass probability", ylim=(-.03,1.03))
axs[1,0].set(title="(c) Rare-event view", xlabel="$a$ (%)", ylabel="Four-stream raw-pass probability")
axs[1,0].set_yscale("log")
axs[1,0].set_ylim(1e-14,1.2)
axs[1,0].axhline(upper95, linestyle=":", label="0 hits / 100,000: 95% upper bound")
axs[1,1].plot(k, pmf, linewidth=1.7)
axs[1,1].fill_between(k[k <= 800], pmf[k <= 800], alpha=.45, label=r"$K_X\leq800$")
axs[1,1].fill_between(k[(k > 800) & (k <= t)], pmf[(k > 800) & (k <= t)], alpha=.35, label=r"$800<K_X\leq1000$")
axs[1,1].fill_between(k[k > t], pmf[k > t], alpha=.3, label=r"$K_X>1000$")
for x in [800,900,1000]:
    axs[1,1].axvline(x, linestyle="--", linewidth=1)
axs[1,1].set(title="(d) One-stream residual count", xlabel="Residual count $K_X$", ylabel="Probability mass")
for a in axs.flat:
    a.grid(True, alpha=.22)
axs[0,0].legend(fontsize=8)
axs[0,1].legend(fontsize=8)
axs[1,0].legend(fontsize=7)
axs[1,1].legend(fontsize=8)
fig.suptitle("GHZ contract protocol: working-residual reference benchmark", fontsize=14)
fig.savefig(FIG/"qcs_residual_overview.png", dpi=240)
fig.savefig(FIG/"qcs_residual_overview.pdf")
plt.close(fig)

# Save residual data.
with open(DATA/"residual_mc_background.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["e","a","r","theory_pass_probability","mc_pass_frequency","repetitions"]); w.writerows(res_a_rows)
with open(DATA/"residual_mc_work_x.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["e","a","r","theory_pass_probability","mc_pass_frequency","repetitions"]); w.writerows(res_b_rows)
key_res=[]
for a in [0.02,0.04,0.05,0.06]:
    qx=e0+(1-2*e0)*a
    for r in raw_thresholds:
        key_res.append({"e":e0,"a":a,"qX":qx,"qZ":e0,"n":n,"t":t,"r":r,
                        "P_raw":residual_pass_prob(e0,a,r)})
with open(DATA/"residual_key_results.json","w") as f:
    json.dump(key_res,f,indent=2)

# ----------------------- Mermin A/B simulation ------------------------------
epsM = 1e-10
Mmin = 3.2
NM_ref = 4606
delta_ref = 4*math.sqrt(2*math.log(1/epsM)/NM_ref)

def kmin_for_N(N):
    return int(math.ceil(0.9*N + math.sqrt(N*math.log(1/epsM)/2)))

def cert_prob(N, pwin):
    kmin = kmin_for_N(int(N))
    if kmin > N:
        return 0.0
    return float(binom.sf(kmin-1, int(N), float(pwin)))

# A: one Born-sampled data set for each v, with theoretical line and lower bound.
v_grid = np.linspace(0.40, 1.00, 16)
A_rows=[]
fig, ax = plt.subplots(figsize=(8.0,5.2), constrained_layout=True)
qline, = ax.plot(v_grid, 4*v_grid, linewidth=2, label=r"Quantum mean $M=4v$")
qcolor = qline.get_color()
ax.axhline(2, linestyle="--", linewidth=1.5, label=r"Local-classical bound $M=2$")
ax.axhline(Mmin, linestyle="-.", linewidth=1.5, label=fr"Certification threshold $M_{{\min}}={Mmin}$")
for v in v_grid:
    pwin=(1+v)/2
    wins=int(rng.binomial(NM_ref,pwin))
    Mhat=8*wins/NM_ref-4
    ML=Mhat-delta_ref
    A_rows.append([v,pwin,NM_ref,wins,Mhat,delta_ref,ML,int(ML>=Mmin)])
    ax.scatter(v,Mhat,s=24,marker="o",color=qcolor)
    ax.vlines(v,ML,Mhat,linewidth=1,alpha=.65,color=qcolor)
    ax.scatter(v,ML,s=38,marker="v",color=qcolor)
ax.set(xlabel="GHZ visibility $v$",ylabel="Mermin score",
       title="A. Finite-sample Mermin score and certification")
ax.set_xlim(.38,1.02); ax.set_ylim(0,4.12)
ax.grid(True,alpha=.25); ax.legend(fontsize=8,loc="lower right")
fig.savefig(FIG/"qcs_mermin_A.png",dpi=240)
fig.savefig(FIG/"qcs_mermin_A.pdf")
plt.close(fig)

# B: exact IID binomial certification probabilities vs N; best local pwin=.75.
N_grid=np.arange(1152,30001,24)
v_values=[0.85,0.90,0.96]
B_rows=[]
# Monte Carlo checks at selected points, 100,000 independent batches each.
mc_rows=[]
for v in [0.85,0.90,0.96]:
    pwin=(1+v)/2
    for N in [2200,4606,6200,12000,26000]:
        kmin=kmin_for_N(N)
        if kmin>N:
            hits=0
        else:
            wins=rng.binomial(N,pwin,100000)
            hits=int(np.sum(wins>=kmin))
        freq=hits/100000
        mc_rows.append([v,N,pwin,kmin,cert_prob(N,pwin),freq,hits,100000])
fig, (ax1,ax2)=plt.subplots(2,1,figsize=(8.2,8.4),sharex=True,
                            gridspec_kw={"height_ratios":[2.1,1]},constrained_layout=True)
for v in v_values:
    pwin=(1+v)/2
    probs=np.array([cert_prob(N,pwin) for N in N_grid])
    line, = ax1.plot(N_grid,probs,label=fr"GHZ visibility $v={v:.2f}$")
    line_color = line.get_color()
    for N,pv in zip(N_grid,probs):
        B_rows.append([int(N),v,pwin,pv])
# local classical IID response: max per-round win probability 3/4.
local_probs=np.array([cert_prob(N,0.75) for N in N_grid])
ax1.plot(N_grid,local_probs,linestyle="--",color="black",label="Best local-classical IID strategy")
ax1.axvline(NM_ref,linestyle=":",color="gray",label=fr"Reference $N_M={NM_ref}$")
# Add binomial Monte Carlo points and exact two-sided 95% Clopper-Pearson intervals.
color_map = {0.85:"C0", 0.90:"C1", 0.96:"C2"}
for v in [0.85,0.90,0.96]:
    rows=[row for row in mc_rows if row[0]==v]
    xs=[]; ys=[]; lo=[]; hi=[]; point_color=None
    for row in rows:
        _,N,_,_,_,freq,hits,R=row
        xs.append(N); ys.append(freq)
        low=0.0 if hits==0 else beta.ppf(0.025,hits,R-hits+1)
        high=1.0 if hits==R else beta.ppf(0.975,hits+1,R-hits)
        lo.append(freq-low); hi.append(high-freq)
    ax1.errorbar(xs,ys,yerr=[lo,hi],fmt="o",color=color_map[v],markersize=3,capsize=2,alpha=.8)
ax1.set(ylabel=r"$P_{\rm cert}=\Pr(M_L\geq3.2)$",
        title="B. Certification probability versus Mermin sample size",
        ylim=(-.03,1.03))
ax1.grid(True,alpha=.25); ax1.legend(fontsize=8,loc="lower right")
# Log10 classical probability in the lower panel; use logsf for stability.
local_log10=[]
for N in N_grid:
    kmin=kmin_for_N(int(N))
    if kmin>N:
        local_log10.append(np.nan)
    else:
        lp=binom.logsf(kmin-1,int(N),0.75)
        local_log10.append(lp/math.log(10))
ax2.plot(N_grid,local_log10,color="black",linestyle="--")
ax2.axvline(NM_ref,linestyle=":",color="gray")
ax2.set(xlabel="Number of Mermin test trials $N_M$",
        ylabel=r"$\log_{10} P_{\rm cert}^{\rm local}$")
ax2.grid(True,alpha=.25)
fig.savefig(FIG/"qcs_mermin_B.png",dpi=240)
fig.savefig(FIG/"qcs_mermin_B.pdf")
plt.close(fig)

# Save A/B simulation data.
with open(DATA/"mermin_A_samples.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["v","p_win","N_M","wins","Mhat","delta","M_L","pass"]); w.writerows(A_rows)
with open(DATA/"mermin_B_theory.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["N_M","v","p_win","P_cert"]); w.writerows(B_rows)
with open(DATA/"mermin_B_mc.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["v","N_M","p_win","k_min","P_cert_theory","P_cert_MC","hits","repetitions"]); w.writerows(mc_rows)

# Combined A/B overview.
from PIL import Image
imgA=Image.open(FIG/"qcs_mermin_A.png").convert("RGB")
imgB=Image.open(FIG/"qcs_mermin_B.png").convert("RGB")
# Save separately; a side-by-side montage for convenient manuscript inclusion.
fig, axs=plt.subplots(1,2,figsize=(15,5.8),constrained_layout=True)
for ax, im in [(axs[0],imgA),(axs[1],imgB)]:
    ax.imshow(im); ax.axis("off")
fig.savefig(FIG/"qcs_mermin_AB_overview.png",dpi=220)
fig.savefig(FIG/"qcs_mermin_AB_overview.pdf")
plt.close(fig)

# Summary with exact reference-point probabilities.
ref = {
    "n":n,"t":t,"raw_thresholds":raw_thresholds,"e0":e0,"a0":a0,
    "qX_at_e02_a04":e0+(1-2*e0)*a0,"qZ":e0,
    "Mermin":{"N_M":NM_ref,"epsilon_M":epsM,"M_min":Mmin,
              "delta_N":delta_ref,"k_min_at_4606":kmin_for_N(NM_ref),
               "P_cert_at_4606":{str(v):cert_prob(NM_ref,(1+v)/2) for v in [0.85,0.90,0.96]},
              "P_fail_at_4606":{str(v):float(binom.cdf(kmin_for_N(NM_ref)-1,NM_ref,(1+v)/2)) for v in [0.85,0.90,0.96]},
              "P_cert_local_IID_at_4606":cert_prob(NM_ref,0.75)},
    "note":"Residual model assumes independent Bernoulli errors in four streams. Mermin model assumes IID trials and independent uniform settings. Neither simulation is a complete adversarial protocol or full adjudicator simulation."
}
with open(DATA/"summary.json","w") as f: json.dump(ref,f,indent=2)

print(json.dumps(ref,indent=2))
