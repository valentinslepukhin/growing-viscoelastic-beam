# Growth-Induced Transitions in Viscoelastic Matter — beam simulation

Simulation code for the growing viscoelastic beam of *Growth-Induced Transitions in
Viscoelastic Matter* (V. Slepukhin and O. Hallatschek). The script reproduces Fig. 1 of
the paper: buckling, coiling and snapping of a unidirectionally growing viscoelastic beam
as a function of the single dimensionless control parameter `gτ`.

## What is solved

A slender beam grows unidirectionally at rate `g` with its endpoints pinned, so it must
buckle to accommodate its increasing arclength. The material is Maxwell viscoelastic with
relaxation time `τ`. Writing `θ(s,t)` for the angle between the beam tangent and the
initial beam axis, with `s` the arclength in the *initial* (undeformed) frame, the torque
carries an exponentially fading memory of the curvature,

```
T(s,t) = I ∫₀ᵗ e^{-(t-t')/τ} d( e^{-g t'} θ'(s,t') )
```

Growth does not appear explicitly in the torque; it enters through the compressive load,
which is fixed by the constraint that the end-to-end distance stays constant while the
arclength grows as `L(t) = L₀ e^{gt}`.

The code integrates the equivalent differential system, written in terms of the deviation
from pure elasticity `q(s,t) = d/ds( T - I e^{-gt} θ' )` and the **rescaled** load
`F̃ = e^{gt} F`:

```
dq/dt = -(F̃/τ) sin θ                        (evolution, explicit Euler)
I e^{-gt} θ'' - F̃ sin θ + q = 0             (boundary-value problem at each t)
F̃ = k ( L₀ - e^{gt} ∫ cos θ ds )            (stiff spring enforcing the end-to-end constraint)
```

Note `F̃` is not the physical load; the physical one is `F = e^{-gt} F̃`. In the code the
variable `h` is `-q`, so its update carries the opposite sign to the equation above.

### Geometry and boundary conditions

Only half the beam is simulated, `s ∈ [0, L]`, exploiting the symmetry `θ(-s) = -θ(s)`:

- `θ(0) = 0` — midpoint of the full beam,
- `θ'(L) = 0` — zero torque at the pinned end, imposed by the reflection `pf[N+1] = pf[N-1]`.

The full beam therefore has length `2L`, and `F_euler = -E π²/(2L)²` is its Euler buckling
load. The seed `θ₀ = ε sin(πs/2L)` is the first buckling mode.

### Snaps

At each step the boundary-value problem is solved by Newton's method (`scipy.optimize.root`,
`hybr`) starting from the previous configuration. When the beam snaps, no solution exists
near the previous one — Newton fails, and the code falls back to relaxation in a fictitious
time `t̃`,

```
γ dθ/dt̃ = I e^{-gt} θ'' - F̃ sin θ + q
```

integrated with BDF until a new stationary state is reached. A snap is recorded when the
tip angle jumps by more than `π/2`.

## Running

```bash
pip install numpy scipy matplotlib
python viscoelastic-beam-figures.py
```

Writes `combined_figure.png` and `combined_figure.pdf` (Fig. 1) to the working directory.
The whole run takes about a minute and a half on a laptop, dominated by the 16-value `τ`
sweep at `dt = 1e-3` (~70 s).

## Figure panels

Panels are laid out row-major, matching the paper's Fig. 1 caption:

| panel | content |
|---|---|
| (a) | schematic of the bending deformation |
| (b) | tip angle `θ(L₀,t)` vs. `L/L₀` for `gτ = 0.1, 1, 100` |
| (c) | beam profile immediately before and after a snap, at `gτ = 1` |
| (d) | number of snaps accumulated up to `T = 5/g` vs. `gτ` |
| (e) | effective energy `𝓔(t)` vs. `t` for the same three values of `gτ` |
| (f) | effective energy `𝓔(T)` at `T = 5/g` vs. `gτ` |

The effective energy is the functional whose variation with respect to `θ'` reproduces the
torque, evaluated at each instant with `t` treated as a parameter (see the paper's SI):

```
𝓔 = I ∫₀^{L₀} ds (1/2τ) ∫₀ᵗ dt' e^{-(t-t')/τ} e^{-gt} / (1 - e^{-t/τ})
                    ( θ'(s,t) - (1 - e^{-t/τ}) e^{g(t-t')} θ'(s,t') )²
```

The paper's `s` runs over the whole beam `[0, L₀]`, while this code simulates only the
half `[0, L]` with `θ(0) = 0` at the midpoint, i.e. `L = L₀/2`. Since `θ` is odd about the
midpoint, `κ = θ'` is even and the integrand is symmetric, so the full-beam integral is
exactly twice the half-beam one — hence the factor 2 in the code, which cancels the `1/2`
in the prefactor.

## Parameters

Set at the top of the script:

| name | value | meaning |
|---|---|---|
| `g` | `1.0` | growth rate — held fixed; all `gτ` variation comes from `τ` |
| `a` | `1.0` | growth exponent in the stiffness `E e^{-a t}` and in the energy functional; equals `g` |
| `E` | `1.0` | bending modulus — the paper's `I = 2μ∫y²dy`, with the elastic modulus absorbed |
| `L` | `1.0` | half-length of the beam |
| `k` | `500.0` | spring constant enforcing the end-to-end constraint |
| `N` | `80` | spatial grid points |
| `dt` | `1e-3` | time step |
| `T_final` | `5.0` | final time, i.e. `T = 5/g` |
| `eps` | `0.05` | amplitude of the initial buckling seed |

**On `dt`:** `θ ≡ 0, q = 0` is an exact fixed point of the dynamics, and the buckling
amplitude grows as `A ≈ 2√(gt)` — near-vertical at `t = 0`. If the first step is too large,
Newton converges onto the trivial branch and the beam never buckles (`θ(L) ≡ 0` for every
`τ`). `dt = 1e-3` is comfortably inside the working range; `dt = 5e-3` already fails.

## Small-angle check

While the amplitude is small the shape is fixed entirely by the current length and is
independent of `gτ`:

```
θ(s,t) = A(t) cos(π s / L₀),    A(t) = 2 √( 1 - L₀/L(t) ) = 2 √( 1 - e^{-gt} )
```

so the centerline is a sine wave. At `t = 0.5` with `g = 1` this predicts `A = 1.254`
against a simulated tip angle of `1.318` — agreeing to a few percent at an amplitude
already outside the strict range of validity. In the elastic limit the same calculation
gives `F = -I π²/L²(t)`, the Euler load at the instantaneous length.

## Files

- `viscoelastic-beam-figures.py` — the simulation and figure script
- `combined_figure.png` / `.pdf` — Fig. 1 of the paper

---

## Changelog

### 2026-08-03

**Panels reordered to match the paper.** The script laid the figure out column-major
(`(a),(c),(e)` on the top row, `(b),(d),(f)` on the bottom), so its panel labels did not
correspond to the panels described in the Fig. 1 caption. The layout is now row-major and
agrees with the caption; see the panel table above. The published `combined_figure.png` was
already in the correct order — the script was the file out of sync.

**Twin axis removed.** The snaps panel carried a second `φ(L₀,T)` axis that the caption does
not mention. Panel (d) is now `N_snaps` alone.

**Energy expression matched to the SI.** The code previously computed the functional without
the `I/2` prefactor, and integrated over the simulated half-beam only. Those two omissions
cancel exactly, so the plotted curves were always correct, but the code now writes both
factors explicitly and stays correct if `E` is changed from 1. See the note under
*Figure panels* above.

**Relabelled** `H → 𝓔` and `φ → θ` to match the paper's symbols, and moved the `y`-axis label
in the cartoon so it no longer collides with the panel title.

**Runtime corrected** in this README: the run takes about 90 s, not the hour previously
claimed.

Verified after the re-run — the regenerated figure reproduces the published numbers exactly:
`N_snaps` = 0, 0, 1, 2, 4, 5, 5, 5, 4, 3, 2, 1, 1, 0, 0, 0 across the `gτ` sweep;
`𝓔(T)` peaks at 5.81 at `gτ = 3`; tip angle 18.44 at `gτ = 1`.

### Known difference from the published figure

Panel (a) of the published `combined_figure.png` is a **different drawing** from the one this
script produces: the published version shows a bent line with a dashed centerline, a red
tangent arrow and a blue `θ(s)` label, while the script draws a filled arc annotated with `R`
and `s`. Both are consistent with the Fig. 1 caption. Only the panel *ordering* was
reconciled; the artwork was left alone. Decide before submission whether to redraw the
script's cartoon or to keep the hand-made panel (a).

### Repository status

The GitHub repository `valentinslepukhin/growing-viscoelastic-beam` is still **private**, and
its remote copy is still at the original commit `4c6b9b5` — it does **not** yet contain any of
the 2026-08-03 changes above. The Data Availability URL in the paper will 404 until both the
push and

```bash
gh repo edit valentinslepukhin/growing-viscoelastic-beam --visibility public
```

have been done. Push first: publishing the current remote would expose a version of the
script whose panel order contradicts Fig. 1.
