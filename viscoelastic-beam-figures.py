"""
Combined publication figure — growing viscoelastic beam (Fig. 1 of the paper).

Layout  (2 rows × 3 cols, two-column wide) — row-major, matching the paper's Fig. 1:
  col 0                  col 1                   col 2
  ──────────────────────────────────────────────────────────────
  (a) cartoon            (b) θ(L0,t) vs L/L0     (c) beam snap
  (d) N_snaps vs gτ      (e) 𝓔(t) vs t (3 taus)  (f) 𝓔(T) vs gτ

Solves, in the rescaled-load convention Ftilde = e^{gt} F:
    dq/dt      = -(Ftilde/tau) sin(phi)
    I e^{-gt} phi'' - Ftilde sin(phi) + q = 0
with q(s,t) = d/ds( T - I e^{-gt} phi' ) and Ftilde = k(L0 - e^{gt} int cos(phi) ds).
The code variable `h` is -q, so its update carries the opposite sign.
"""

import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import time as timer

# ============================================================
# Publication style
# ============================================================
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
})

# ============================================================
# Physics parameters
# ============================================================
g, L, E, a, k = 1.0, 1.0, 1.0, 1.0, 500.0
N = 80; ds = L / N; s = np.linspace(0, L, N + 1)
trap_w = np.ones(N + 1) * ds; trap_w[0] = ds / 2; trap_w[-1] = ds / 2

eps = 0.05
phi0 = eps * np.sin(np.pi * s / (2 * L)); phi0[0] = 0.0
F_euler = -E * np.pi**2 / (4 * L**2)
L0 = np.dot(np.cos(phi0), trap_w) + F_euler / k

def compute_F(phi_arr, t_phys):
    return k * (L0 - np.dot(np.cos(phi_arr) * np.exp(g * t_phys), trap_w))

def compute_d2(phi_int):
    pf = np.zeros(N + 2); pf[1:N+1] = phi_int; pf[N+1] = pf[N-1]
    return (pf[2:N+2] - 2 * pf[1:N+1] + pf[0:N]) / ds**2

def compute_kappa(phi_full):
    kap = np.zeros(N + 1)
    kap[0] = (phi_full[1] - phi_full[0]) / ds
    kap[1:N] = (phi_full[2:N+1] - phi_full[0:N-1]) / (2 * ds)
    kap[N] = (phi_full[N] - phi_full[N-1]) / ds
    return kap

def bvp_residual(phi_int, h_nodes, t_phys):
    stiff = E * np.exp(-a * t_phys)
    pf = np.zeros(N + 1); pf[1:N+1] = phi_int
    F = compute_F(pf, t_phys)
    return stiff * compute_d2(phi_int) - h_nodes[1:N+1] - F * np.sin(pf[1:N+1])

def bvp_jacobian(phi_int, h_nodes, t_phys):
    stiff = E * np.exp(-a * t_phys); egt = np.exp(g * t_phys)
    pf = np.zeros(N + 1); pf[1:N+1] = phi_int; F = compute_F(pf, t_phys)
    c = stiff / ds**2
    J = np.diag(np.full(N, -2*c)) + np.diag(np.full(N-1, c), 1) + np.diag(np.full(N-1, c), -1)
    J[-1, -2] += c
    sin_p = np.sin(pf[1:N+1]); cos_p = np.cos(pf[1:N+1])
    J[np.arange(N), np.arange(N)] -= F * cos_p
    J -= np.outer(sin_p, k * egt * sin_p * trap_w[1:N+1])
    return J

def try_newton(phi_guess, h_nodes, t_phys, tol=1e-6):
    sol = root(bvp_residual, phi_guess[1:N+1], args=(h_nodes, t_phys),
               jac=bvp_jacobian, method='hybr', options={'maxfev': 200})
    phi_out = np.zeros(N + 1); phi_out[1:N+1] = sol.x
    res = np.max(np.abs(sol.fun))
    return phi_out, (sol.success and res < tol), res

def relax_solve(phi_guess, h_nodes, t_phys, mu_init=0.05, max_fict_time=10.0, tol=1e-5):
    stiff = E * np.exp(-a * t_phys); mu = mu_init; theta = phi_guess[1:N+1].copy()
    def make_rhs_jac(mu_val):
        def rhs(t_f, th):
            pf = np.zeros(N + 1); pf[1:N+1] = th; F = compute_F(pf, t_phys)
            return (stiff * compute_d2(th) - h_nodes[1:N+1] - F * np.sin(pf[1:N+1])) / mu_val
        def jac(t_f, th):
            pf = np.zeros(N + 1); pf[1:N+1] = th; F = compute_F(pf, t_phys); egt = np.exp(g * t_phys)
            c = stiff / (mu_val * ds**2); sin_p = np.sin(pf[1:N+1]); cos_p = np.cos(pf[1:N+1])
            J = np.diag(np.full(N, -2*c) - F * cos_p / mu_val)
            J += np.diag(np.full(N-1, c), 1) + np.diag(np.full(N-1, c), -1); J[-1, -2] += c
            J -= np.outer(sin_p, k * egt * sin_p * trap_w[1:N+1]) / mu_val
            return J
        return rhs, jac
    rhs, jac = make_rhs_jac(mu); r0 = np.max(np.abs(rhs(0, theta)))
    sol_p = solve_ivp(rhs, [0, 0.02], theta, method='BDF', jac=jac,
                      rtol=1e-8, atol=1e-10, max_step=0.005)
    if sol_p.status == 0:
        r1 = np.max(np.abs(rhs(0.02, sol_p.y[:, -1])))
        if r1 > 2 * r0: mu = -mu; rhs, jac = make_rhs_jac(mu)
        else:
            theta = sol_p.y[:, -1]
            if np.max(np.abs(bvp_residual(theta, h_nodes, t_phys))) < tol:
                phi_out = np.zeros(N + 1); phi_out[1:N+1] = theta; return phi_out, True
    else: mu = -mu; rhs, jac = make_rhs_jac(mu)
    remaining = max_fict_time; chunk = 0.5
    while remaining > 0:
        dt_c = min(chunk, remaining)
        sol = solve_ivp(rhs, [0, dt_c], theta, method='BDF', jac=jac,
                        rtol=1e-8, atol=1e-10, max_step=0.02)
        if sol.status != 0: break
        theta = sol.y[:, -1]
        if np.max(np.abs(bvp_residual(theta, h_nodes, t_phys))) < tol: break
        remaining -= dt_c
    phi_out = np.zeros(N + 1); phi_out[1:N+1] = theta
    return phi_out, np.max(np.abs(bvp_residual(theta, h_nodes, t_phys))) < tol * 10


# ============================================================
# PART 1: tau=1 run — capture first jump with |Δφ(L)| > π/2
# ============================================================
tau_snap = 1.0
T_final  = 5.0; dt = 1e-3; n_steps = int(T_final / dt)

phi = phi0.copy(); h = np.zeros(N + 1); F_now = compute_F(phi, 0.0)
first_big_jump = None

print("Running gτ=1 for beam snapshot (first |Δφ|>π/2)...")
w0 = timer.time()
for step in range(1, n_steps + 1):
    t_new = step * dt
    h_new = h + dt * F_now * np.sin(phi) / tau_snap
    phi_old = phi.copy()
    phi_new, ok, res = try_newton(phi, h_new, t_new)
    if not ok:
        phi_new, rok = relax_solve(phi, h_new, t_new)
        delta = (phi_new[-1] - phi_old[-1]) if rok else 0.0
        if rok and first_big_jump is None and abs(delta) > np.pi / 2:
            first_big_jump = {
                't_phys':     t_new,
                'phi_before': phi_old.copy(),
                'phi_after':  phi_new.copy(),
            }
            print(f"  Found at t={t_new:.4f},  "
                  f"φ_tip: {phi_old[-1]:+.3f} → {phi_new[-1]:+.3f}  "
                  f"(Δ = {delta:+.3f} rad = {delta/np.pi:+.2f}π)")
            break
    F_now = compute_F(phi_new, t_new); phi = phi_new; h = h_new

print(f"  Done in {timer.time()-w0:.1f}s")


# ============================================================
# PART 2: Sweep — store κ history for ALL taus
# ============================================================
tau_values_plot  = [0.1, 1.0, 100.0]
tau_values_dense = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0,
                    3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 50.0, 100.0]

hist_skip  = 5
n_hist     = n_steps // hist_skip + 1
save_every = max(1, int(0.01 / dt))
data       = {}

print("\nRunning tau sweep...")
wall_total = timer.time()

for tau in tau_values_dense:
    phi = phi0.copy(); h = np.zeros(N + 1); F_now = compute_F(phi, 0.0)
    t_save = [0.0]; phiL_save = [phi[-1]]; jumps = []
    kappa_hist = np.zeros((n_hist, N + 1))
    kappa_hist[0] = compute_kappa(phi)
    t_hist = np.zeros(n_hist); hist_idx = 1

    w0 = timer.time()
    for step in range(1, n_steps + 1):
        t_new = step * dt
        h_new = h + dt * F_now * np.sin(phi) / tau
        phi_old_tip = phi[-1]
        phi_new, ok, res = try_newton(phi, h_new, t_new)
        if not ok: phi_new, rok = relax_solve(phi, h_new, t_new)
        delta = phi_new[-1] - phi_old_tip
        if abs(delta) > np.pi / 2: jumps.append((t_new, delta))
        F_now = compute_F(phi_new, t_new); phi = phi_new; h = h_new
        if step % hist_skip == 0 and hist_idx < n_hist:
            kappa_hist[hist_idx] = compute_kappa(phi); t_hist[hist_idx] = t_new; hist_idx += 1
        if step % save_every == 0:
            t_save.append(t_new); phiL_save.append(phi[-1])

    data[tau] = {
        't':          np.array(t_save),
        'phiL':       np.array(phiL_save),
        'jumps':      jumps,
        'final_phiL': phi[-1],
        'kappa_hist': kappa_hist[:hist_idx],
        't_hist':     t_hist[:hist_idx],
    }
    print(f"  gτ = {g*tau:5.1f}  |  φ(L) = {phi[-1]:+8.3f}  "
          f"|  jumps: {len(jumps):2d}  |  {timer.time()-w0:.1f}s")

print(f"Total sweep: {timer.time()-wall_total:.0f}s")


# ============================================================
# PART 3: Compute the effective energy 𝓔(t) for all taus
# ============================================================
# Paper's functional (SI), with E playing the role of the bending modulus I:
#
#   𝓔 = I ∫_0^{L0} ds (1/2τ) ∫_0^t dt' e^{-(t-t')/τ} e^{-gt} /(1 - e^{-t/τ})
#                            ( θ'(s,t) - (1 - e^{-t/τ}) e^{g(t-t')} θ'(s,t') )²
#
# The paper's s runs over the WHOLE beam [0, L0] (both ends torque-free,
# mode cos(πns/L0)), while this code simulates only the half [0, L] with
# θ(0)=0 at the midpoint, i.e. L = L0/2.  θ is odd about the midpoint, so
# κ = θ' is even and the integrand is symmetric: the full-beam integral is
# exactly twice the half-beam one.  Hence the factor 2 below, which cancels
# the 1/2 in the prefactor — with E = 1 the numbers are unchanged, but the
# expression is now the paper's, and stays correct if E is ever changed.
print("\nComputing 𝓔(t) for all tau values...")
for tau in tau_values_dense:
    d = data[tau]
    kh = d['kappa_hist']; th = d['t_hist']
    dt_h = th[1] - th[0] if len(th) > 1 else dt * hist_skip
    t_eval = d['t']
    H_vals = np.zeros(len(t_eval))
    for ei, t_e in enumerate(t_eval):
        if t_e < 2 * dt_h: continue
        idx_t = np.searchsorted(th, t_e, side='right') - 1
        if idx_t < 1: continue
        kappa_now     = kh[idx_t]
        one_minus_exp = 1.0 - np.exp(-t_e / tau)
        if one_minus_exp < 1e-15: continue
        prefactor = E * np.exp(-a * t_e) / (2.0 * tau * one_minus_exp)
        integ_s = np.zeros(N + 1)
        for j in range(idx_t + 1):
            delta_t = t_e - th[j]
            diff    = kappa_now - one_minus_exp * np.exp(a * delta_t) * kh[j]
            integ_s += np.exp(-delta_t / tau) * diff**2 * dt_h
        # factor 2: half-beam integral -> full-beam integral (see note above)
        H_vals[ei] = 2.0 * prefactor * np.dot(integ_s, trap_w)
    d['H'] = H_vals
    print(f"  gτ = {g*tau:.1f}: 𝓔(T) = {H_vals[-1]:.4f}")


# ============================================================
# PART 4: Build 2×3 combined figure
# ============================================================
colors_3 = ['#1b9e77', '#d95f02', '#7570b3']   # gτ = 0.1, 1, 10

fig = plt.figure(figsize=(7.2, 4.8))
gs = gridspec.GridSpec(
    2, 3, figure=fig,
    left=0.07, right=0.97,
    top=0.94, bottom=0.10,
    wspace=0.50, hspace=0.48,
)

# Row-major, matching the panel labels used in the paper's Fig. 1 caption.
ax_a = fig.add_subplot(gs[0, 0])   # (a) cartoon
ax_b = fig.add_subplot(gs[0, 1])   # (b) θ(L0,t) vs L/L0
ax_c = fig.add_subplot(gs[0, 2])   # (c) beam snap
ax_d = fig.add_subplot(gs[1, 0])   # (d) N_snaps vs gτ
ax_e = fig.add_subplot(gs[1, 1])   # (e) 𝓔(t) vs t
ax_f = fig.add_subplot(gs[1, 2])   # (f) 𝓔(T) vs gτ


# ──────────────────────────────────────────────────────────────
# (a) Cartoon: beam fragment bent into circular arc
#     Origin at midpoint of centerline (top of arc).
#     X along arc (tangent at origin), Y perpendicular outward.
#     Center of curvature is BELOW the origin.
# ──────────────────────────────────────────────────────────────
ax = ax_a
ax.set_aspect('equal')
ax.axis('off')

R_cart   = 3.0       # radius of curvature (cartoon units)
h_beam   = 0.80      # beam thickness (cartoon units)
th_max   = 0.72      # half-arc angle [rad] (~41°)
theta_c  = np.linspace(-th_max, th_max, 300)

# Geometry: center of curvature at (0, -R_cart) in plot coords.
# A point at radial distance r from center, at angle θ from upward vertical:
#   x = r sin θ,   y = r cos θ − R_cart   →  at (r,θ)=(R,0): (0, 0) ✓
# Outward normal (away from center) at angle θ: n̂ = (sin θ, cos θ).
# Beam boundary at ±h/2 from centerline:
#   outer (Y = +h/2):  r = R + h/2
#   inner (Y = −h/2):  r = R − h/2

def arc_xy(r, theta):
    return r * np.sin(theta), r * np.cos(theta) - R_cart

xc, yc = arc_xy(R_cart,            theta_c)          # centerline
xo, yo = arc_xy(R_cart + h_beam/2, theta_c)          # outer boundary
xi, yi = arc_xy(R_cart - h_beam/2, theta_c)          # inner boundary

# End caps (straight segments connecting outer → inner at each end)
def end_cap(theta_e):
    pts_x = [(R_cart + h_beam/2) * np.sin(theta_e),
             (R_cart - h_beam/2) * np.sin(theta_e)]
    pts_y = [(R_cart + h_beam/2) * np.cos(theta_e) - R_cart,
             (R_cart - h_beam/2) * np.cos(theta_e) - R_cart]
    return pts_x, pts_y

# Filled beam body
fill_x = np.concatenate([xo, xi[::-1]])
fill_y = np.concatenate([yo, yi[::-1]])
ax.fill(fill_x, fill_y, color='#d0e8f8', alpha=0.85, zorder=1)

# Outer and inner solid boundaries
ax.plot(xo, yo, 'k-', lw=1.4, zorder=3)
ax.plot(xi, yi, 'k-', lw=1.4, zorder=3)

# End caps
for sgn in [-1, 1]:
    cx, cy = end_cap(sgn * th_max)
    ax.plot(cx, cy, 'k-', lw=1.4, zorder=3)

# Centerline (dashed)
ax.plot(xc, yc, 'k--', lw=1.0, dashes=(5, 3), zorder=4)

# ── Coordinate axes at origin (0, 0) ──
ax_len   = 1.05 * (R_cart + h_beam/2) * np.sin(th_max)   # ~x extent
ax_len_x = ax_len * 0.55
ax_len_y = ax_len * 0.55

arrow_kw = dict(arrowstyle='->', color='k',
                mutation_scale=8, lw=1.1, zorder=6)
ax.annotate('', xy=( ax_len_x, 0), xytext=(0, 0),
            arrowprops=arrow_kw)
ax.annotate('', xy=(0,  ax_len_y), xytext=(0, 0),
            arrowprops=arrow_kw)

ax.text(ax_len_x + 0.07, -0.02, r'$x$',
        ha='left', va='top', fontsize=9, style='italic')
# label to the LEFT of the arrow tip, not above it, so it does not
# collide with the panel title
ax.text(-0.10, ax_len_y, r'$y$',
        ha='right', va='center', fontsize=9, style='italic')

# Origin dot
ax.plot(0, 0, 'ko', ms=2.8, zorder=7)

# ── Radius: full line from center of curvature to origin + label R ──
y_center = -R_cart
# dot at center of curvature
ax.plot(0, y_center, 'o', color='gray', ms=3.5, zorder=5)
# solid line: center of curvature → origin
ax.plot([0, 0], [y_center, 0], color='gray', lw=0.9,
        ls=(0, (4, 2)), zorder=2)
# arrowhead pointing from center toward origin
ax.annotate('', xy=(0, -0.05), xytext=(0, y_center * 0.55),
            arrowprops=dict(arrowstyle='->', color='gray',
                            mutation_scale=7, lw=0.9))
# label R at midpoint, offset to right
ax.text(0.12, y_center * 0.50,
        r'$R$', ha='left', va='center',
        fontsize=9, color='gray')

# ── Arc-length arrow along centerline ──
# Small double-headed arrow along the top surface to suggest arc length
s_ann = np.linspace(-th_max * 0.55, -th_max * 0.10, 40)
xs_ann, ys_ann = arc_xy(R_cart + h_beam/2 + 0.25, s_ann)
ax.annotate('', xy=(xs_ann[-1], ys_ann[-1]), xytext=(xs_ann[0], ys_ann[0]),
            arrowprops=dict(arrowstyle='<->', color='#555555',
                            mutation_scale=7, lw=0.9))
ax.text(np.mean(xs_ann), np.mean(ys_ann) + 0.18,
        r'$s$', ha='center', va='bottom', fontsize=8, color='#555555')

ax.set_title('(a)', pad=3)

# set equal limits with a small margin
xpad, ypad = 0.3, 0.3
ax.set_xlim(xo.min() - xpad, xo.max() + xpad + 0.5)
ax.set_ylim(y_center - ypad * 0.6, yo.max() + ypad + 0.6)


# ──────────────────────────────────────────────────────────────
# (c) Beam shape — first |Δθ|>π/2 snap at gτ=1
# ──────────────────────────────────────────────────────────────
ax = ax_c
if first_big_jump is not None:
    t_phys = first_big_jump['t_phys']
    growth = np.exp(g * t_phys)
    for phi_plot, col, lab in [
        (first_big_jump['phi_before'], '#1f77b4', 'before'),
        (first_big_jump['phi_after'],  '#d62728', 'after'),
    ]:
        s_sym   = np.concatenate([-s[::-1][:-1], s])
        phi_sym = np.concatenate([-phi_plot[::-1][:-1], phi_plot])
        n_f = len(s_sym)
        X = np.zeros(n_f); Y = np.zeros(n_f)
        for i in range(1, n_f):
            d_s = s_sym[i] - s_sym[i-1]
            X[i] = X[i-1] + 0.5*(np.cos(phi_sym[i-1])+np.cos(phi_sym[i]))*growth*d_s
            Y[i] = Y[i-1] + 0.5*(np.sin(phi_sym[i-1])+np.sin(phi_sym[i]))*growth*d_s
        ax.plot(X, Y, color=col, lw=1.7, label=lab)

ax.set_title('(c)', pad=3)
ax.set_aspect('equal', adjustable='datalim')
ax.axis('off')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2,
          framealpha=0.0, edgecolor='none', handlelength=1.6,
          columnspacing=1.4)


# ──────────────────────────────────────────────────────────────
# (b) θ(L0,t) vs L/L0
# ──────────────────────────────────────────────────────────────
ax = ax_b
for i, tau in enumerate(tau_values_plot):
    d = data[tau]
    length = np.exp(g * d['t'])
    lbl = (f'$g\\tau = {g*tau:.0f}$' if g*tau == int(g*tau)
           else f'$g\\tau = {g*tau}$')
    ax.plot(length, d['phiL'], color=colors_3[i], label=lbl)
    for j_t, _ in d['jumps']:
        ax.axvline(np.exp(g*j_t), color=colors_3[i], lw=0.4, alpha=0.4)

for n_pi in range(1, 8):
    val = n_pi * np.pi
    if val < 22:
        ax.axhline(val, color='gray', ls=':', lw=0.35, alpha=0.4)

ax.set_xlabel(r'$L / L_0$')
ax.set_ylabel(r'$\theta(L_0,\, t)$')
ax.set_title('(b)', pad=3)
ax.legend(loc='upper left', framealpha=0.9, edgecolor='none')
ax.set_xlim(1, np.exp(g * T_final) + 3)
ax.set_ylim(0, None)
ax.grid(True, alpha=0.15, lw=0.4)


# ──────────────────────────────────────────────────────────────
# (e) 𝓔(t) vs physical time t  (faint jump verticals)
# ──────────────────────────────────────────────────────────────
ax = ax_e
for i, tau in enumerate(tau_values_plot):
    d = data[tau]
    lbl = (f'$g\\tau = {g*tau:.0f}$' if g*tau == int(g*tau)
           else f'$g\\tau = {g*tau}$')
    ax.plot(d['t'], d['H'], color=colors_3[i], label=lbl)
    for j_t, _ in d['jumps']:
        ax.axvline(j_t, color=colors_3[i], lw=0.4, alpha=0.4)

ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\mathcal{E}(t)$')
ax.set_title('(e)', pad=3)
ax.legend(loc='upper left', framealpha=0.9, edgecolor='none')
ax.set_xlim(0, T_final)
ax.grid(True, alpha=0.15, lw=0.4)


# ──────────────────────────────────────────────────────────────
# (d) number of snaps vs gτ  [log x]
#     The paper's caption describes this panel as N_snaps alone, so no
#     twin φ(L0,T) axis here.
# ──────────────────────────────────────────────────────────────
ax = ax_d
tau_dense    = np.array(sorted(data.keys()))
gtau_dense   = g * tau_dense
n_jumps_arr  = np.array([len(data[t]['jumps'])  for t in tau_dense])

ax.plot(gtau_dense, n_jumps_arr, 's-',
        color='#d62728', ms=4, lw=1.4,
        mfc='white', mew=1.0)

ax.set_xlabel(r'$g\tau$')
ax.set_ylabel(r'$N_{\mathrm{snaps}}$')
ax.set_xscale('log')
ax.set_ylim(-0.3, None)
ax.set_title('(d)', pad=3)
ax.grid(True, alpha=0.15, which='both', lw=0.4)


# ──────────────────────────────────────────────────────────────
# (f) 𝓔(T) vs gτ  [log x]
# ──────────────────────────────────────────────────────────────
ax = ax_f
H_finals = np.array([data[tau]['H'][-1] for tau in tau_dense])

ax.plot(gtau_dense, H_finals, 'o-',
        color='#9467bd', ms=4.5, lw=1.4,
        mfc='white', mew=1.1)
ax.set_xlabel(r'$g\tau$')
ax.set_ylabel(r'$\mathcal{E}(T)$')
ax.set_title('(f)', pad=3)
ax.set_xscale('log')
ax.grid(True, alpha=0.15, which='both', lw=0.4)



# ──────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────
out_png = 'combined_figure.png'
out_pdf = 'combined_figure.pdf'
plt.savefig(out_png, dpi=300)
plt.savefig(out_pdf)
plt.close()
print(f"\n\u2713 Saved {out_png}")
print(f"\u2713 Saved {out_pdf}")
