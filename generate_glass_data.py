import jax
# Enable x64 (Double Precision) - REQUIRED for Physics
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax_md import space, energy, simulate, minimize
import numpy as np
import pickle
from tqdm import tqdm

print("Initializing 'Quench & Crush' Glass Generator...")

# 1. SETUP
N = 1000   
density = 1.2
box_size = float((N / density) ** (1/3))
dt = 0.002
target_temp = 0.44

# 2. SYSTEM DEFINITION
displacement, shift = space.periodic(box_size)
sigma = jnp.array([[1.0, 0.8], [0.8, 0.88]], dtype=jnp.float64)
epsilon = jnp.array([[1.0, 1.5], [1.5, 0.5]], dtype=jnp.float64)
species = jnp.where(jnp.arange(N) < 800, 0, 1)

# Real Physics
energy_fn = energy.lennard_jones_pair(
    displacement, species=species, sigma=sigma, epsilon=epsilon, r_cutoff=2.5
)
# Soft Physics (Marshmallows)
soft_energy_fn = energy.soft_sphere_pair(
    displacement, species=species, sigma=sigma, epsilon=epsilon, alpha=jnp.array(2.0)
)

# 3. INITIALIZATION
key = jax.random.PRNGKey(42)
R_init = jax.random.uniform(key, (N, 3), minval=0.0, maxval=box_size)

print("[1/6] UNTANGLING (Soft Gradient Descent)...")
# Gently push atoms apart using Soft Spheres
gd_init, gd_apply = minimize.gradient_descent(soft_energy_fn, shift, step_size=1e-3)
state = gd_init(R_init) 
state = jax.lax.fori_loop(0, 5000, lambda i, s: gd_apply(s), state)
R_safe = state 

print("[2/6] PRE-HARDENING (Hard Gradient Descent)...")
# Switch to Real Physics
gd_init_real, gd_apply_real = minimize.gradient_descent(energy_fn, shift, step_size=1e-4)
state = gd_init_real(R_safe)
state = jax.lax.fori_loop(0, 2000, lambda i, s: gd_apply_real(s), state)
R_ready = state 

current_energy = energy_fn(R_ready)
print(f"  > Pre-Thermal Energy: {current_energy:.4f}")

# 4. THERMALIZATION
print("[3/6] HEATING (Brownian Dynamics T=0.44)...")
init_fn, apply_fn = simulate.brownian(energy_fn, shift, dt=dt, kT=target_temp)
state = init_fn(key, R_ready)

# Run 10,000 steps to melt the glass
steps = 10000
@jax.jit
def run_dynamics(state):
    return jax.lax.fori_loop(0, steps, lambda i, s: apply_fn(s, t=i*dt), state)

state = run_dynamics(state)
R_hot = state.position 

# --- CRITICAL NEW STEP: QUENCHING ---
print("[4/6] QUENCHING (Brownian Dynamics T=0.0)...")
# We run the simulation with ZERO temperature. 
# This acts as a "physics-compliant" minimizer that drains heat 
# without teleporting atoms (which causes explosions).
quench_fn, quench_apply = simulate.brownian(energy_fn, shift, dt=dt, kT=0.0)
state_quench = quench_fn(key, R_hot)

# Run 5,000 steps to cool down rapidly but safely
@jax.jit
def run_quench(state):
    return jax.lax.fori_loop(0, 5000, lambda i, s: quench_apply(s, t=i*dt), state)

state_quench = run_quench(state_quench)
R_cold = state_quench.position
print("  > System Quenched. Atoms are now cold and stable.")

# 5. INHERENT STRUCTURE (Final Minimization)
print("[5/6] FINAL CRUSH (FIRE Minimization)...")

# Now that atoms are cold, we can use FIRE (Fast Inertial Relaxation)
# We use small initial steps to be extra safe.
fire_init, fire_apply = minimize.fire_descent(energy_fn, shift, dt_start=1e-5, dt_max=1e-3)
fire_state = fire_init(R_cold)

# Run deep minimization
fire_state = jax.lax.fori_loop(0, 5000, lambda i, s: fire_apply(s), fire_state)
R_inherent = fire_state.position 

final_energy = energy_fn(R_inherent)
print(f"  > Final Energy: {final_energy:.4f}")

if final_energy > -6000:
    print("CRITICAL FAILURE: Energy is not low enough.")
    exit()
else:
    print("  > SUCCESS: Energy is Negative and Stable.")

# 6. HESSIAN
print("[6/6] Computing Hessian...")
def total_energy(R_flat):
    R = R_flat.reshape(N, 3)
    return energy_fn(R)

grad_fn = jax.grad(total_energy)

def hvp(v, R_flat):
    return jax.jvp(grad_fn, (R_flat,), (v,))[1]

R_flat = R_inherent.reshape(-1)
identity = jnp.eye(3 * N)
hessian_columns = jax.lax.map(lambda v: hvp(v, R_flat), identity)
H = hessian_columns.T 

print("  > Diagonalizing...")
evals, evecs = jnp.linalg.eigh(H)
soft_modes = evecs[:, 3:13].reshape(N, 3, 10)
softness_score = jnp.sum(jnp.square(soft_modes), axis=(1, 2))

# 7. SAVE
data = {
    "positions": np.array(R_inherent),
    "species": np.array(species),
    "box_size": box_size,
    "hessian_eigenvectors": np.array(soft_modes),
    "softness_score": np.array(softness_score), 
    "energy": float(final_energy)
}

with open("glass_data_1k.pkl", "wb") as f:
    pickle.dump(data, f)

print("Success! Dataset generated.")