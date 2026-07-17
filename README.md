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
The full sweep takes on the order of an hour on a laptop; runtime is dominated by the
16-value `τ` sweep at `dt = 1e-3`.

## Figure panels

| panel | content |
|---|---|
| (a) | schematic of the bending deformation |
| (b) | beam profile immediately before and after a snap, at `gτ = 1` |
| (c) | tip angle `φ(L₀,t)` vs. `L/L₀` for `gτ = 0.1, 1, 100` |
| (d) | effective energy `H(t)` vs. `t` for the same three values |
| (e) | number of snaps and tip angle at `T = 5/g` vs. `gτ` |
| (f) | effective energy `H(T)` at `T = 5/g` vs. `gτ` |

The effective energy is the functional whose variation with respect to `θ'` reproduces the
torque, evaluated at each instant with `t` treated as a parameter (see the paper's SI).

## Parameters

Set at the top of the script:

| name | value | meaning |
|---|---|---|
| `g` | `1.0` | growth rate — held fixed; all `gτ` variation comes from `τ` |
| `a` | `1.0` | growth exponent in the stiffness `E e^{-a t}` and in the energy functional; equals `g` |
| `E` | `1.0` | bending modulus |
| `L` | `1.0` | half-length of the beam |
| `k` | `500.0` | spring constant enforcing the end-to-end constraint |
| `N` | `80` | spatial grid points |
| `dt` | `1e-3` | time step |
| `T_final` | `5.0` | final time, i.e. `T = 5/g` |
| `eps` | `0.05` | amplitude of the initial buckling seed |

**On `dt`:** `θ ≡ 0, q = 0` is an exact fixed point of the dynamics, and the buckling
amplitude grows as `A ≈ 2√(gt)` — near-vertical at `t = 0`. If the first step is too large,
Newton converges onto the trivial branch and the beam never buckles (`φ(L) ≡ 0` for every
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
