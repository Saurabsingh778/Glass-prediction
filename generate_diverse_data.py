import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax_md import space, energy, simulate, minimize
import numpy as np
import pickle
from tqdm import tqdm
import os

print("=" * 70)
print("DIVERSE GLASS DATASET GENERATOR")
print("=" * 70)

# ==========================================
# CONFIGURATION
# ==========================================
NUM_SNAPSHOTS = 120  # Generate 120 diverse base snapshots
N = 1000
BASE_DENSITY = 1.2
BASE_TEMP = 0.44

# Variation ranges for diversity
DENSITY_RANGE = (1.1, 1.3)
TEMP_RANGE = (0.40, 0.48)
QUENCH_STEPS_RANGE = (3000, 7000)
SPECIES_RATIO_RANGE = (0.75, 0.85)  # Fraction of species 0

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def generate_varied_parameters(snapshot_id):
    """Generate physics parameters for diversity"""
    np.random.seed(snapshot_id * 42)
    
    density = np.random.uniform(*DENSITY_RANGE)
    target_temp = np.random.uniform(*TEMP_RANGE)
    quench_steps = int(np.random.uniform(*QUENCH_STEPS_RANGE))
    species_ratio = np.random.uniform(*SPECIES_RATIO_RANGE)
    
    # Vary interaction parameters slightly
    sigma_scale = np.random.uniform(0.95, 1.05)
    epsilon_scale = np.random.uniform(0.90, 1.10)
    
    return {
        'density': density,
        'target_temp': target_temp,
        'quench_steps': quench_steps,
        'species_ratio': species_ratio,
        'sigma_scale': sigma_scale,
        'epsilon_scale': epsilon_scale
    }

def compute_hessian_softness(R_inherent, energy_fn, N):
    """Compute softness from Hessian eigenmodes"""
    R_flat = R_inherent.reshape(-1)
    
    def total_energy(R_flat):
        return energy_fn(R_flat.reshape(N, 3))
    
    grad_fn = jax.grad(total_energy)
    
    def hvp(v, R_flat):
        return jax.jvp(grad_fn, (R_flat,), (v,))[1]
    
    # Compute Hessian column by column (memory efficient)
    identity = jnp.eye(3 * N)
    H = jax.lax.map(lambda v: hvp(v, R_flat), identity).T
    
    # Eigendecomposition
    evals, evecs = jnp.linalg.eigh(H)
    
    # Softness: participation in low-frequency modes (skip 3 rigid translations)
    soft_modes = evecs[:, 3:13].reshape(N, 3, 10)
    softness = jnp.sum(jnp.square(soft_modes), axis=(1, 2))
    
    return softness, evals

def generate_single_snapshot(snapshot_id, params):
    """Generate one glass snapshot with given parameters"""
    
    # Unpack parameters
    density = params['density']
    target_temp = params['target_temp']
    quench_steps = params['quench_steps']
    species_ratio = params['species_ratio']
    sigma_scale = params['sigma_scale']
    epsilon_scale = params['epsilon_scale']
    
    box_size = float((N / density) ** (1/3))
    dt = 0.002
    
    # System Definition
    displacement, shift = space.periodic(box_size)
    
    # Varied interaction parameters
    sigma = jnp.array([
        [1.0 * sigma_scale, 0.8 * sigma_scale],
        [0.8 * sigma_scale, 0.88 * sigma_scale]
    ], dtype=jnp.float64)
    
    epsilon = jnp.array([
        [1.0 * epsilon_scale, 1.5 * epsilon_scale],
        [1.5 * epsilon_scale, 0.5 * epsilon_scale]
    ], dtype=jnp.float64)
    
    # Varied species composition
    n_species_0 = int(N * species_ratio)
    species = jnp.where(jnp.arange(N) < n_species_0, 0, 1)
    
    # Energy functions
    energy_fn = energy.lennard_jones_pair(
        displacement, species=species, sigma=sigma, epsilon=epsilon, r_cutoff=2.5
    )
    soft_energy_fn = energy.soft_sphere_pair(
        displacement, species=species, sigma=sigma, epsilon=epsilon, alpha=jnp.array(2.0)
    )
    
    # Initialize with varied seed
    key = jax.random.PRNGKey(snapshot_id)
    R_init = jax.random.uniform(key, (N, 3), minval=0.0, maxval=box_size)
    
    # STEP 1: Untangle (Soft)
    gd_init, gd_apply = minimize.gradient_descent(soft_energy_fn, shift, step_size=1e-3)
    state = gd_init(R_init)
    state = jax.lax.fori_loop(0, 5000, lambda i, s: gd_apply(s), state)
    R_safe = state
    
    # STEP 2: Pre-harden (Hard)
    gd_init_real, gd_apply_real = minimize.gradient_descent(energy_fn, shift, step_size=1e-4)
    state = gd_init_real(R_safe)
    state = jax.lax.fori_loop(0, 2000, lambda i, s: gd_apply_real(s), state)
    R_ready = state
    
    # STEP 3: Thermalize (Heat with VARIED temperature)
    init_fn, apply_fn = simulate.brownian(energy_fn, shift, dt=dt, kT=target_temp)
    state = init_fn(key, R_ready)
    state = jax.lax.fori_loop(0, 10000, lambda i, s: apply_fn(s, t=i*dt), state)
    R_hot = state.position
    
    # STEP 4: Quench (VARIED cooling rate)
    quench_fn, quench_apply = simulate.brownian(energy_fn, shift, dt=dt, kT=0.0)
    state_quench = quench_fn(key, R_hot)
    state_quench = jax.lax.fori_loop(0, quench_steps, lambda i, s: quench_apply(s, t=i*dt), state_quench)
    R_cold = state_quench.position
    
    # STEP 5: Minimize (Inherent Structure)
    fire_init, fire_apply = minimize.fire_descent(energy_fn, shift, dt_start=1e-5, dt_max=1e-3)
    fire_state = fire_init(R_cold)
    fire_state = jax.lax.fori_loop(0, 5000, lambda i, s: fire_apply(s), fire_state)
    R_inherent = fire_state.position
    
    final_energy = energy_fn(R_inherent)
    
    # STEP 6: Compute Softness
    softness, eigenvalues = compute_hessian_softness(R_inherent, energy_fn, N)
    
    # Package data
    data = {
        "positions": np.array(R_inherent),
        "species": np.array(species),
        "box_size": box_size,
        "softness_score": np.array(softness),
        "energy": float(final_energy),
        "eigenvalues": np.array(eigenvalues[:20]),  # Save lowest 20 eigenvalues
        "parameters": params  # Store generation params for analysis
    }
    
    return data, final_energy

# ==========================================
# MAIN GENERATION LOOP
# ==========================================
output_dir = "diverse_glass_dataset"
os.makedirs(output_dir, exist_ok=True)

print(f"\nGenerating {NUM_SNAPSHOTS} diverse glass snapshots...")
print(f"Output directory: {output_dir}/")
print("-" * 70)

successful = 0
failed = 0

for i in tqdm(range(NUM_SNAPSHOTS), desc="Generating"):
    try:
        # Generate varied parameters
        params = generate_varied_parameters(i)
        
        # Generate snapshot
        data, final_energy = generate_single_snapshot(i, params)
        
        # Quality check
        if final_energy > -5000:
            print(f"\n⚠ Snapshot {i}: Energy too high ({final_energy:.1f}), skipping...")
            failed += 1
            continue
        
        # Save
        filename = f"{output_dir}/glass_data_{i:03d}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        
        successful += 1
        
        # Progress report every 20 snapshots
        if (i + 1) % 20 == 0:
            print(f"\n✓ Progress: {successful} successful, {failed} failed")
            
    except Exception as e:
        print(f"\n✗ Snapshot {i} failed: {str(e)}")
        failed += 1
        continue

print("\n" + "=" * 70)
print(f"GENERATION COMPLETE")
print(f"  Successful: {successful}")
print(f"  Failed: {failed}")
print(f"  Success Rate: {100*successful/(successful+failed):.1f}%")
print("=" * 70)

# Save generation metadata
metadata = {
    "num_snapshots": successful,
    "n_atoms": N,
    "density_range": DENSITY_RANGE,
    "temp_range": TEMP_RANGE,
    "quench_steps_range": QUENCH_STEPS_RANGE,
    "species_ratio_range": SPECIES_RATIO_RANGE
}

with open(f"{output_dir}/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("\nDataset ready for training!")