import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
import numpy as np
import pickle
import glob
import math
import gc
from scipy.stats import pearsonr
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm # Progress bar

print("="*70)
print("PHASE 2: TRAINING HYBRID PHYSICS-TRANSFORMER (RAM CACHED)")
print("="*70)

# ==========================================
# 1. MATH & BASIS
# ==========================================
class CosineCutoff(nn.Module):
    def __init__(self, cutoff=2.5):
        super().__init__()
        self.cutoff = cutoff
    def forward(self, dist):
        cutoffs = 0.5 * (torch.cos(dist * math.pi / self.cutoff) + 1.0)
        return cutoffs * (dist < self.cutoff).float()

class BesselBasis(nn.Module):
    def __init__(self, num_rbf=32, cutoff=2.5):
        super().__init__()
        self.cutoff = cutoff
        self.register_buffer("freq", torch.arange(1, num_rbf + 1).float() * np.pi)
    def forward(self, dist):
        d_scaled = dist / self.cutoff
        return torch.sin(self.freq * d_scaled.unsqueeze(-1)) / (d_scaled.unsqueeze(-1) + 1e-6)

# ==========================================
# 2. DATASET (RAM CACHED - FIXES HANGING)
# ==========================================
class AugmentedPhysicsDataset(Dataset):
    def __init__(self, file_dir="/kaggle/working/augmented_dataset", cutoff=2.5):
        super().__init__()
        self.file_list = glob.glob(f"{file_dir}/*.pkl")
        self.cutoff = cutoff
        self.data_list = [] # CACHE IN RAM
        
        print(f"Loading {len(self.file_list)} files into RAM to prevent IO bottlenecks...")
        
        # Load everything once with a progress bar
        for path in tqdm(self.file_list, desc="Caching Dataset"):
            with open(path, "rb") as f:
                raw = pickle.load(f)
            
            pos = torch.tensor(raw['positions'], dtype=torch.float)
            species = torch.tensor(raw['species'], dtype=torch.long)
            softness = torch.tensor(raw['softness_score'], dtype=torch.float)
            evals = torch.tensor(raw['eigenvalues'], dtype=torch.float)
            box = raw['box_size']
            
            pos_np = pos.numpy()
            diff = pos_np[:, None, :] - pos_np[None, :, :]
            diff = diff - box * np.round(diff / box)
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            
            mask = (dist < self.cutoff) & (dist > 1e-3)
            row, col = np.where(mask)
            edge_index = torch.tensor(np.vstack((row, col)), dtype=torch.long)
            edge_dist = torch.tensor(dist[mask], dtype=torch.float)
            
            x = torch.cat([F.one_hot(species, 2).float(), evals], dim=1)
            y = (softness - 0.0) / 0.02 
            
            self.data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_dist, y=y.unsqueeze(1)))
            
        print("Dataset successfully cached in RAM.")
        
    def len(self): return len(self.data_list)
    def get(self, idx): return self.data_list[idx]

# ==========================================
# 3. HYBRID ARCHITECTURE
# ==========================================
class HybridBlock(nn.Module):
    def __init__(self, hidden_dim, num_rbf, num_heads=4):
        super().__init__()
        self.head_dim = hidden_dim // num_heads
        self.num_heads = num_heads
        
        self.rbf_mlp = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x, edge_index, rbf, cutoff):
        row, col = edge_index
        x_norm = self.norm(x)
        
        # Physics
        w_phys = self.rbf_mlp(rbf) * cutoff.unsqueeze(-1)
        msg_phys = x_norm[col] * w_phys
        
        # Attention
        Q = self.q(x_norm).view(-1, self.num_heads, self.head_dim)
        K = self.k(x_norm).view(-1, self.num_heads, self.head_dim)
        V = self.v(x_norm).view(-1, self.num_heads, self.head_dim)
        
        q_edge = Q[row]
        k_edge = K[col]
        v_edge = V[col]
        
        scores = (q_edge * k_edge).sum(dim=-1) / math.sqrt(self.head_dim)
        scores = scores * cutoff.unsqueeze(-1)
        attn = F.softmax(scores, dim=1)
        
        msg_attn = (attn.unsqueeze(-1) * v_edge).reshape(-1, self.num_heads * self.head_dim)
        
        combined = msg_phys + self.gate * msg_attn
        
        aggr = torch.zeros_like(x, dtype=combined.dtype)
        aggr.index_add_(0, row, combined)
        
        return x + self.out(aggr)

class PhysicsTransformer(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=5):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rbf = BesselBasis(num_rbf=32, cutoff=2.5)
        self.cut = CosineCutoff(cutoff=2.5)
        self.layers = nn.ModuleList([
            HybridBlock(hidden_dim, num_rbf=32) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        rbf = self.rbf(edge_attr)
        cut = self.cut(edge_attr)
        h = self.embed(x)
        
        for layer in self.layers:
            h = checkpoint(layer, h, edge_index, rbf, cut, use_reentrant=False)
            
        return self.head(h)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train_hybrid_physics():
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. LOAD DATASET
    # We assume filenames are formatted like "aug_XXXXX.pkl"
    # where XXXXX corresponds to the augmentation index
    # We need to map them back to the ORIGINAL 120 snapshots.
    
    file_dir = "/kaggle/working/augmented_dataset"
    all_files = sorted(glob.glob(f"{file_dir}/*.pkl"))
    
    # We know we have 10 augmentations per original file.
    # Total files = 1200. Original snapshots = 120.
    # Files 0-9 belong to Snapshot 0. Files 10-19 belong to Snapshot 1.
    
    total_files = len(all_files)
    aug_factor = 10 
    num_originals = total_files // aug_factor # Should be 120
    
    # 2. CREATE STRICT SPLIT INDICES
    indices = np.arange(num_originals)
    np.random.shuffle(indices) # Shuffle the ORIGINAL IDs (0 to 119)
    
    split = int(0.85 * num_originals) # 102 Train, 18 Test
    train_orig_ids = indices[:split]
    test_orig_ids = indices[split:]
    
    # Expand back to file indices
    train_file_indices = []
    for i in train_orig_ids:
        # Add all 10 rotations for this ID
        start = i * aug_factor
        train_file_indices.extend(range(start, start + aug_factor))
        
    test_file_indices = []
    for i in test_orig_ids:
        start = i * aug_factor
        test_file_indices.extend(range(start, start + aug_factor))
        
    print(f"Original Snapshots: {num_originals}")
    print(f"Train Files: {len(train_file_indices)} (from {len(train_orig_ids)} unique snapshots)")
    print(f"Test Files:  {len(test_file_indices)}  (from {len(test_orig_ids)} unique snapshots)")
    
    # 3. INITIALIZE DATASET
    dataset = AugmentedPhysicsDataset() # Loads everything into RAM
    
    # 4. SUBSET WITH FIXED INDICES
    from torch.utils.data import Subset
    train_set = Subset(dataset, train_file_indices)
    test_set = Subset(dataset, test_file_indices)
    
    # 5. DATALOADERS (Proceed as normal)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=0)
    
    model = PhysicsTransformer(hidden_dim=128, num_layers=5).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')
    
    ACCUM_STEPS = 4 
    best_corr = 0
    
    print("\nStarting Training...")
    
    for epoch in range(1, 201):
        model.train()
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        # Add TQDM to first epoch so user sees it moving
        iter_wrapper = tqdm(train_loader, desc=f"Epoch {epoch}") if epoch == 1 else train_loader
        
        for i, data in enumerate(iter_wrapper):
            data = data.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                pred = model(data)
                loss = F.l1_loss(pred, data.y)
                loss = loss / ACCUM_STEPS 
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * ACCUM_STEPS * data.num_graphs
            
        avg_loss = total_loss / len(train_set)
        
        if epoch % 2 == 0:
            model.eval()
            all_p, all_t = [], []
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device, non_blocking=True)
                    with torch.amp.autocast('cuda'):
                        pred = model(data)
                    all_p.append(pred.float().cpu().numpy().flatten())
                    all_t.append(data.y.float().cpu().numpy().flatten())
            
            p = np.concatenate(all_p)
            t = np.concatenate(all_t)
            
            r = 0.0
            if np.std(p) > 1e-6:
                r, _ = pearsonr(p, t)
                
            scheduler.step(avg_loss)
            
            status = ""
            if r > best_corr:
                best_corr = r
                torch.save(model.state_dict(), "hybrid_physics_best.pth")
                status = "★ NEW BEST"
                
            print(f"Epoch {epoch:<3} | Loss: {avg_loss:.4f} | R: {r:.4f} | {status}")
            
    print(f"Final Best Correlation: {best_corr:.4f}")

if __name__ == "__main__":
    train_hybrid_physics()