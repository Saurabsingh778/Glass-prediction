import torch
import numpy as np
import pickle
import glob
import os
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

print("="*70)
print("PHASE 1: PHYSICS-INFORMED DATA AUGMENTATION")
print("="*70)

# ==========================================
# 1. PHYSICS ENGINE (Hessian Calculation)
# ==========================================
def compute_cage_eigenvalues(pos, species, box_size):
    """
    Computes the 3 Eigenvalues of the Local Hessian (Cage Stiffness)
    for every atom. Returns tensor of shape [N, 3].
    """
    # Constants
    base_sigma = torch.tensor([[1.0, 0.8], [0.8, 0.88]])
    base_epsilon = torch.tensor([[1.0, 1.5], [1.5, 0.5]])
    cutoff = 2.5
    
    pos = torch.tensor(pos, dtype=torch.float32)
    species = torch.tensor(species, dtype=torch.long)
    
    # Topology
    pos_np = pos.numpy()
    diff = pos_np[:, None, :] - pos_np[None, :, :]
    diff = diff - box_size * np.round(diff / box_size)
    dist_sq = np.sum(diff**2, axis=-1)
    dist = np.sqrt(dist_sq)
    
    mask = (dist < cutoff) & (dist > 1e-3)
    row, col = np.where(mask)
    
    vecs = torch.tensor(diff[mask], dtype=torch.float32)
    r = torch.tensor(dist[mask], dtype=torch.float32)
    
    # Physics parameters
    s_row, s_col = species[row], species[col]
    sig = base_sigma[s_row, s_col]
    eps = base_epsilon[s_row, s_col]
    
    # Hessian Math
    inv_r2 = 1.0 / (r**2)
    s6 = (sig / r) ** 6
    s12 = s6 ** 2
    
    k_rad = (24 * eps * inv_r2) * (26 * s12 - 7 * s6)
    k_trans = -(24 * eps * inv_r2) * (2 * s12 - s6)
    
    n = vecs / r.unsqueeze(-1)
    outer = torch.einsum('bi,bj->bij', n, n)
    eye = torch.eye(3).unsqueeze(0).expand(len(vecs), 3, 3)
    
    # Edge Matrix
    H_edge = (k_rad - k_trans).view(-1, 1, 1) * outer + k_trans.view(-1, 1, 1) * eye
    
    # Aggregate to Nodes
    D_flat = torch.zeros(len(pos), 3, 3)
    # Loop is fine here, we run this offline once
    for i, source in enumerate(row):
        D_flat[source] += H_edge[i]
        
    # Eigendecomposition
    evals, _ = torch.linalg.eigh(D_flat)
    
    # Log-Scale for Neural Network stability
    evals = torch.sign(evals) * torch.log1p(torch.abs(evals))
    
    return evals.numpy()

# ==========================================
# 2. AUGMENTATION LOOP
# ==========================================
def augment_dataset(input_pattern="glass_data_*.pkl", output_dir="augmented_data", factor=10):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(input_pattern))
    
    print(f"Found {len(files)} snapshots. Augmenting x{factor}...")
    
    count = 0
    for file_path in tqdm(files):
        # Load Raw
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            
        pos = data['positions']
        spec = data['species']
        box = data['box_size']
        soft = data['softness_score']
        
        # 1. CALCULATE PHYSICS (ONCE)
        # Eigenvalues are rotationally invariant! We compute them once
        # and attach them to all rotated copies.
        eigenvalues = compute_cage_eigenvalues(pos, spec, box)
        
        # 2. GENERATE ROTATIONS
        for i in range(factor):
            # Rotate Positions
            rot_matrix = R.random().as_matrix()
            center = box / 2.0
            pos_centered = pos - center
            pos_rotated = np.dot(pos_centered, rot_matrix)
            pos_final = (pos_rotated + center) % box
            
            new_data = {
                "positions": pos_final,
                "species": spec,
                "box_size": box,
                "softness_score": soft,
                "eigenvalues": eigenvalues # <--- THE PHYSICS GRAB
            }
            
            save_name = os.path.join(output_dir, f"aug_{count:05d}.pkl")
            with open(save_name, "wb") as f:
                pickle.dump(new_data, f)
            count += 1
            
    print(f"Done! {count} files saved to {output_dir}/")

# RUN AUGMENTATION
if __name__ == "__main__":
    # Ensure we use the raw data from your generation script
    # Assuming raw files are in current dir or specific path
    augment_dataset(input_pattern="/kaggle/input/physics-crystal/glass_data_*.pkl", output_dir="augmented_dataset", factor=10)