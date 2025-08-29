import argparse
import os
import h5py
import numpy as np
from collections import defaultdict, deque
from tqdm import tqdm

# ---------------------------
# Helpers
# ---------------------------
def quantize_index(val: np.ndarray, grid: float) -> np.ndarray:
    # Round-half-up quantizer to integer bin index.
    return np.floor(val / grid + 0.5).astype(np.int64)

def energy_weighted_center(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    # Calculates energy-weighted centroid; falls back to mean if weights are zero.
    ws = float(np.sum(w))
    return (np.average(X, weights=w, axis=0) if ws > 0 else np.mean(X, axis=0))

def pca_axis(P: np.ndarray, E: np.ndarray):
    # Energy-weighted PCA principal axis of a shower: returns (origin, unit_axis).
    O = energy_weighted_center(P, E)
    Y = P - O
    if P.shape[0] < 2:
        return O, np.array([0.0, 0.0, 1.0], dtype=np.float64)
    w = np.asarray(E, dtype=np.float64)
    ws = float(np.sum(w)) if np.sum(w) > 0 else float(len(w))
    Yw = Y * np.sqrt((w / ws)[:, None])
    cov = Yw.T @ Yw
    vals, vecs = np.linalg.eigh(cov)
    a = vecs[:, np.argmax(vals)]
    a = a / (np.linalg.norm(a) + 1e-12)
    return O, a

def orthonormal_basis_from_axis(a: np.ndarray):
    # Return two transverse unit vectors orthogonal to axis a.
    a = a / (np.linalg.norm(a) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, tmp); e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(a, e1);  e2 /= (np.linalg.norm(e2) + 1e-12)
    return e1, e2

def freedman_diaconis_bin_width(x: np.ndarray) -> float:
    # Freedman–Diaconis to find bin width(layer width).
    x = np.asarray(x)
    n = x.size
    if n <= 1:
        return np.inf
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        iqr = np.std(x)
    if iqr <= 0:
        return np.inf
    return 2.0 * iqr * (n ** (-1.0 / 3.0))

# ---------------------------
# I/O
# ---------------------------
def load_first_n_events(input_file: str, n: int = 100, ecal_only: bool = True):
    # Load first N events from hdf5 file (only from the ecal barrel if selected).
    with h5py.File(input_file, "r") as f:
        steps = f["steps"]
        ev = steps["event_id"][:]
        keep = np.unique(ev)[:n]
        mask = np.isin(ev, keep)
        if ecal_only and "subdetector" in steps:
            mask &= (steps["subdetector"][:] == 0)  # ECAL barrel = 0
        data = {
            "positions": steps["position"][:][mask],
            "energies":  steps["energy"][:][mask],
            "times":     steps["time"][:][mask],
            "event_ids": steps["event_id"][:][mask],
        }
        meta = {}
        if "metadata" in f and "subdetector_names" in f["metadata"]:
            meta["subdetector_names"] = f["metadata"]["subdetector_names"][:]
    return data, meta

# ---------------------------
# Energy-aware 2D CCL on (u,v) bins (optional merge stage)
# ---------------------------
def connected_components_groups_energy(
    cell_map: dict,
    bin_energy: dict,
    bin_hits: dict,
    neighborhood: int = 4,
    bridge_energy_factor: float = 0.60,
    bridge_min_hits: int | None = None,
    similarity_ratio: float = 0.50,
):
    """
    2D connected components on occupied (u,v) bins with local energy gating.

    Edge (k ↔ n) allowed if:
      E[k] >= alpha * median_3x3(k) AND E[n] >= alpha * median_3x3(n)
      AND min(Ek, En) >= similarity_ratio * max(Ek, En)
    Optionally OR with a hits-based gate if bridge_min_hits is provided.
    """
    keys = list(cell_map.keys())
    if not keys:
        return []

    nbrs = (
        [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)] if neighborhood == 8
        else [(1,0),(-1,0),(0,1),(0,-1)]
    )

    def local_median(k):
        u, v = k
        vals = [bin_energy.get((u+du, v+dv)) for du, dv in (nbrs + [(0,0)])]
        vals = [x for x in vals if x is not None]
        return float(np.median(vals)) if vals else 0.0

    loc_med = {k: local_median(k) for k in keys}
    key_to_id = {k: i for i, k in enumerate(keys)}
    visited = np.zeros(len(keys), dtype=bool)
    groups = []

    for i, k in enumerate(keys):
        if visited[i]:
            continue
        comp_bins = []
        q = deque([i]); visited[i] = True
        while q:
            j = q.popleft()
            kj = keys[j]
            comp_bins.append(kj)
            Ej = bin_energy[kj]; mj = loc_med[kj]

            for du, dv in nbrs:
                kk = (kj[0] + du, kj[1] + dv)
                jj = key_to_id.get(kk)
                if jj is None or visited[jj]:
                    continue
                Ek = bin_energy[kk]; mk = loc_med[kk]

                pass_local = (Ej >= bridge_energy_factor * max(mj, 1e-12)) and \
                             (Ek >= bridge_energy_factor * max(mk, 1e-12))
                sim_ok = (min(Ej, Ek) >= similarity_ratio * max(Ej, Ek))

                hits_ok = False
                if bridge_min_hits is not None:
                    hits_ok = (bin_hits[kj] + bin_hits[kk]) >= int(bridge_min_hits)

                if (pass_local and sim_ok) or hits_ok:
                    visited[jj] = True
                    q.append(jj)

        # flatten to per-hit indices
        hit_idxs = []
        for b in comp_bins:
            hit_idxs.extend(cell_map[b])
        groups.append(np.asarray(hit_idxs, dtype=np.int64))

    return groups

# ---------------------------
# Clustering (axis fit → pseudo-layers → (u,v) grid → optional CCL merge)
# ---------------------------
def ecal_clustering(
    data,
    grid_trans_mm: float = 3.0,
    fd_min_mm: float = 2.0,
    fd_max_mm: float = 10.0,
    energy_quantile_for_axis: float = 0.7,
    second_stage: str = "ccl4",   # 'none' | 'ccl4' | 'ccl8'
    bridge_energy_factor: float = 0.60,
    similarity_ratio: float = 0.50,
    bridge_min_hits: int | None = None,
):
    """
    Grid-based ECAL clustering in an axis-aligned frame.

    Returns a point cloud dict with: positions, energies, times, subdetectors, event_ids.
    """
    out_pos, out_en, out_t, out_sdet, out_eid = [], [], [], [], []

    unique_events = np.unique(data["event_ids"])
    for event_id in tqdm(unique_events, desc="ECAL clustering (grid, axis)"):
        em = (data["event_ids"] == event_id)
        P = data["positions"][em]
        E = data["energies"][em]
        T = data["times"][em]

        if P.size == 0:
            continue

        # Axis on top-q energy subset for stability
        if E.size >= 5:
            thr = np.quantile(E, energy_quantile_for_axis)
            mask_axis = (E >= thr)
            if not np.any(mask_axis):
                mask_axis = np.ones_like(E, dtype=bool)
        else:
            mask_axis = np.ones_like(E, dtype=bool)

        O, a = pca_axis(P[mask_axis], E[mask_axis])
        e1, e2 = orthonormal_basis_from_axis(a)

        # Project all hits into (w, u, v)
        R = P - O
        w = R @ a
        u = R @ e1
        v = R @ e2

        # Pseudo-layers along w (per-event FD, clamped)
        h_fd = freedman_diaconis_bin_width(w)
        w_bin = float(np.clip(h_fd, fd_min_mm, fd_max_mm)) if np.isfinite(h_fd) and h_fd > 0 else float(fd_min_mm)
        L = np.floor(w / w_bin + 0.5).astype(np.int32)

        for lyr in np.unique(L):
            lm = (L == lyr)
            if not np.any(lm):
                continue

            uL, vL = u[lm], v[lm]
            pL, eL, tL = P[lm], E[lm], T[lm]

            # Quantize to 2D grid
            gu = quantize_index(uL, grid_trans_mm)
            gv = quantize_index(vL, grid_trans_mm)

            # Build per-bin index map and stats
            cell_map: dict[tuple[int,int], list[int]] = defaultdict(list)
            for i in range(uL.size):
                cell_map[(int(gu[i]), int(gv[i]))].append(i)

            bin_energy = {k: float(np.sum(eL[idxs])) for k, idxs in cell_map.items()}
            bin_hits   = {k: int(len(idxs)) for k, idxs in cell_map.items()}

            # Optional 2D CCL merge
            if second_stage in ("ccl4", "ccl8"):
                groups = connected_components_groups_energy(
                    cell_map, bin_energy, bin_hits,
                    neighborhood=(8 if second_stage == "ccl8" else 4),
                    bridge_energy_factor=bridge_energy_factor,
                    bridge_min_hits=bridge_min_hits,
                    similarity_ratio=similarity_ratio,
                )
            else:
                groups = [np.asarray(idxs, dtype=np.int64) for idxs in cell_map.values()]

            # Emit one point per group
            for idxs in groups:
                pos = pL[idxs]
                en  = eL[idxs]
                tim = tL[idxs]

                e_sum = float(en.sum())
                centroid = (np.average(pos, weights=en, axis=0).astype(np.float32)
                            if e_sum > 0 else pos.mean(axis=0).astype(np.float32))

                out_pos.append(centroid)
                out_en.append(np.float32(e_sum))
                out_t.append(np.float32(tim.min()))
                out_sdet.append(np.uint8(0))          # ECAL barrel
                out_eid.append(np.int32(event_id))

    return {
        "positions":    np.asarray(out_pos, dtype=np.float32),
        "energies":     np.asarray(out_en,  dtype=np.float32),
        "times":        np.asarray(out_t,   dtype=np.float32),
        "subdetectors": np.asarray(out_sdet, dtype=np.uint8),
        "event_ids":    np.asarray(out_eid, dtype=np.int32),
    }

# ---------------------------
# Save
# ---------------------------
def save_point_cloud(pc: dict, output_file: str, metadata: dict):
    # Build point cloud with
    with h5py.File(output_file, "w") as f:
        g = f.create_group("steps")
        g.create_dataset("position",    data=pc["positions"])
        g.create_dataset("energy",      data=pc["energies"])
        g.create_dataset("time",        data=pc["times"])
        g.create_dataset("subdetector", data=pc["subdetectors"])
        g.create_dataset("event_id",    data=pc["event_ids"])
        if metadata and ("subdetector_names" in metadata):
            f.create_group("metadata").create_dataset("subdetector_names", data=metadata["subdetector_names"])

# ---------------------------
# Pipeline
# ---------------------------
def run_pipeline(
    input_file: str,
    output_file: str,
    n_events: int = 100,
    grid_trans_mm: float = 3.0,
    fd_min_mm: float = 2.0,
    fd_max_mm: float = 10.0,
    energy_quantile_for_axis: float = 0.7,
    second_stage: str = "ccl4",
    bridge_energy_factor: float = 0.60,
    similarity_ratio: float = 0.50,
    bridge_min_hits: int | None = None,
):
    data, meta = load_first_n_events(input_file, n=n_events, ecal_only=True)

    pc = ecal_clustering(
        data,
        grid_trans_mm=grid_trans_mm,
        fd_min_mm=fd_min_mm,
        fd_max_mm=fd_max_mm,
        energy_quantile_for_axis=energy_quantile_for_axis,
        second_stage=second_stage,
        bridge_energy_factor=bridge_energy_factor,
        similarity_ratio=similarity_ratio,
        bridge_min_hits=bridge_min_hits,
    )

    save_point_cloud(pc, output_file, meta)

    # Minimal summary
    orig = len(data["positions"]); fin = len(pc["positions"])
    cr = (orig / fin) if fin > 0 else np.inf
    print(f"Saved: {output_file}")
    print(f"Points: {orig:,} → {fin:,}  |  Compression: {cr:.2f}×")

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="ECAL grid-based clustering: axis fit → pseudo-layers → (u,v) grid → optional 2D CCL merge."
    )
    p.add_argument("input", type=str, help="Input HDF5 (unclustered steps)")
    p.add_argument("--output", type=str, default=None, help="Output HDF5 file")
    p.add_argument("--n_events", type=int, default=100, help="Number of events to process")

    # Transverse grid and longitudinal pseudo-layering
    p.add_argument("--grid_trans_mm", type=float, default=3.0, help="Transverse bin size (u and v) in mm")
    p.add_argument("--fd_min_mm", type=float, default=2.0, help="Min clamp for FD w-bin width (mm)")
    p.add_argument("--fd_max_mm", type=float, default=10.0, help="Max clamp for FD w-bin width (mm)")
    p.add_argument("--energy_quantile_for_axis", type=float, default=0.7,
                   help="Quantile of energy used to fit PCA axis (0–1)")

    # Optional 2D CCL merge
    p.add_argument("--second_stage", choices=["none", "ccl4", "ccl8"], default="ccl4",
                   help="Merge adjacent (u,v) bins within each pseudo-layer")
    p.add_argument("--bridge_energy_factor", type=float, default=0.60,
                   help="Alpha for per-bin local energy gate (>= alpha * local median)")
    p.add_argument("--similarity_ratio", type=float, default=0.50,
                   help="Min/Max energy ratio required to accept an edge (0–1)")
    p.add_argument("--bridge_min_hits", type=int, default=None,
                   help="Optional hits-based OR for edge acceptance (discouraged; leave unset)")

    return p.parse_args()

def main():
    args = parse_args()
    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)

    default_dir = "/eos/user/t/tnandi/fast_sim_work/Clustered-Output/ecal-clustering"
    if args.output:
        out = args.output
    else:
        grid_t_str = f"{args.grid_trans_mm:.2f}".replace(".", "p")
        out = os.path.join(
            default_dir,
            f"ecal_grid_axis_uv_T{grid_t_str}mm_fd[{args.fd_min_mm:.1f}-{args.fd_max_mm:.1f}]"
            f"_n{args.n_events}{'' if args.second_stage=='none' else f'_{args.second_stage}'}"
            f"_alpha{args.bridge_energy_factor:.2f}_rho{args.similarity_ratio:.2f}"
            f"{'' if args.bridge_min_hits is None else f'_hits{args.bridge_min_hits}'}"
            f".h5"
        )
    os.makedirs(os.path.dirname(out), exist_ok=True)

    run_pipeline(
        input_file=args.input,
        output_file=out,
        n_events=args.n_events,
        grid_trans_mm=args.grid_trans_mm,
        fd_min_mm=args.fd_min_mm,
        fd_max_mm=args.fd_max_mm,
        energy_quantile_for_axis=args.energy_quantile_for_axis,
        second_stage=args.second_stage,
        bridge_energy_factor=args.bridge_energy_factor,
        similarity_ratio=args.similarity_ratio,
        bridge_min_hits=args.bridge_min_hits,
    )

if __name__ == "__main__":
    main()
