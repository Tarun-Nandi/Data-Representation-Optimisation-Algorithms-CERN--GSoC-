import os
import argparse
from typing import Dict, Tuple

import h5py
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN


# =============================================================================
# Fixed scales / geometry
# =============================================================================
XY_SCALE_MM: float = 5.0   # divide x,y by this (cells ≈ 5 mm)
T_SCALE_NS: float = 1.0    # divide (t - median_layer) by this (≈ 1 ns)


# =============================================================================
# Utilities
# =============================================================================
def get_layer(cell_id: int) -> int:
    # Extract layer number from cell_id (9 bits starting at bit 19).
    return (int(cell_id) >> 19) & 0x1FF


def load_first_n_events(path: str, n: int = 100) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    # Load the first N events from the HDF5 file.
    with h5py.File(path, "r") as f:
        steps = f["steps"]
        event_ids_all = steps["event_id"][:]
        kept_event_ids = np.unique(event_ids_all)[:n]
        keep_mask = np.isin(event_ids_all, kept_event_ids)

        data = {
            "positions": steps["position"][:][keep_mask].astype("f4"),
            "energies": steps["energy"][:][keep_mask].astype("f4"),
            "times": steps["time"][:][keep_mask].astype("f4"),
            "pdgs": steps["pdg"][:][keep_mask].astype("i4"),
            "subdetectors": steps["subdetector"][:][keep_mask].astype("u1"),
            "event_ids": steps["event_id"][:][keep_mask].astype("i4"),
            "cell_ids": steps["cell_id"][:][keep_mask].astype("u8"),
        }

        meta: Dict[str, np.ndarray] = {}
        if "metadata" in f and "subdetector_names" in f["metadata"]:
            meta["subdetector_names"] = f["metadata"]["subdetector_names"][:]

    return data, meta


# ---- helper for adaptive mode ----
def _layer_center_and_rq(xy_mm: np.ndarray, energies: np.ndarray, q: float = 0.68):
    """
    Energy-weighted transverse centroid and radius R_q for a layer.
    Returns
        center_xy_mm : (2,)  energy-weighted (x,y) centroid in mm
       rq_mm        :       radius (mm) enclosing fraction q of the layer energy
    """
    E = energies.astype("f4")
    wsum = float(E.sum())
    if wsum > 0:
        center_xy = (xy_mm * (E[:, None] / wsum)).sum(axis=0)
    else:
        center_xy = xy_mm.mean(axis=0)

    radii = np.linalg.norm(xy_mm - center_xy[None, :], axis=1)
    if radii.size == 0 or wsum <= 0:
        return center_xy, radii, 0.0

    order = np.argsort(radii)
    radii_sorted, weights_sorted = radii[order], E[order]
    cumw = np.cumsum(weights_sorted)
    cumw /= cumw[-1]
    rq = float(np.interp(q, cumw, radii_sorted))
    return center_xy, radii, rq


# =============================================================================
# Clustering
# =============================================================================
def cluster_all_events(
    data: Dict[str, np.ndarray],
    eps_scaled: float = 1.5,
    min_samples: int = 5,
    adaptive: bool = False,
    r_quantile: float = 0.68,
    k_core: float = 1.8,
    k_tail: float = 0.6,
) -> np.ndarray:
    """
    Cluster hits per (event, subdetector, layer):

    If adaptive=True, use two-zone eps:
      - core: eps = eps_scaled * k_core
      - tail: eps = eps_scaled * k_tail
      tail clusters are optionally merged to nearest core within base eps.

    Returns
    -------
    labels : np.ndarray of shape (N,)
        cluster labels per hit; -1 denotes noise.
    """
    num_hits = len(data["positions"])
    labels = np.full(num_hits, -1, dtype=int)
    total_clusters = 0
    event_ids = np.unique(data["event_ids"])

    for event_id in tqdm(event_ids, desc="Events"):
        event_mask = data["event_ids"] == event_id
        subdet_ids = np.unique(data["subdetectors"][event_mask])

        for subdet in subdet_ids:
            event_subdet_mask = event_mask & (data["subdetectors"] == subdet)

            positions_es = data["positions"][event_subdet_mask]      
            times_es = data["times"][event_subdet_mask]              
            cell_ids_es = data["cell_ids"][event_subdet_mask]        
            if positions_es.shape[0] < min_samples:
                continue

            layers = np.fromiter((get_layer(cid) for cid in cell_ids_es), dtype=np.int32, count=positions_es.shape[0])
            layer_ids = np.unique(layers)

            for layer in layer_ids:
                layer_mask = layers == layer
                if layer_mask.sum() < min_samples:
                    continue

                xy_mm = positions_es[layer_mask, :2].astype("f4")
                t_layer = times_es[layer_mask].astype("f4")
                t_median = np.median(t_layer)

                # scaled features (unitless)
                xy_scaled = (xy_mm / XY_SCALE_MM).astype("f4")
                t_scaled = ((t_layer - t_median) / T_SCALE_NS).reshape(-1, 1)
                features_scaled = np.hstack([xy_scaled, t_scaled]).astype("f4")

                # indices into global 'data' arrays for the selected layer
                global_indices_for_layer = np.where(event_subdet_mask)[0][layer_mask]

                if not adaptive:
                    db = DBSCAN(eps=eps_scaled, min_samples=min_samples, n_jobs=-1)
                    predicted = db.fit_predict(features_scaled)
                    n_new = len(set(predicted)) - (1 if -1 in predicted else 0)
                    is_cluster = predicted >= 0
                    if n_new > 0 and np.any(is_cluster):
                        predicted[is_cluster] += total_clusters
                        total_clusters += n_new
                    labels[global_indices_for_layer] = predicted
                    continue

                # ---- adaptive: split layer into core/tail by energy-weighted Rq in xy ----
                _, radii_mm, rq_mm = _layer_center_and_rq(
                    xy_mm, data["energies"][event_subdet_mask][layer_mask], q=r_quantile
                )
                core_mask = radii_mm <= rq_mm
                tail_mask = ~core_mask

                layer_labels = np.full(layer_mask.sum(), -1, dtype=int)

                # (a) core clustering with larger eps
                core_pred = np.array([], dtype=int)
                if core_mask.sum() >= min_samples:
                    db_core = DBSCAN(eps=eps_scaled * k_core, min_samples=min_samples, n_jobs=-1)
                    core_pred = db_core.fit_predict(features_scaled[core_mask])
                    n_new = len(set(core_pred)) - (1 if -1 in core_pred else 0)
                    is_core_cluster = core_pred >= 0
                    if n_new > 0 and np.any(is_core_cluster):
                        core_pred[is_core_cluster] += total_clusters
                        total_clusters += n_new
                    layer_labels[: core_mask.sum()] = core_pred

                # (b) tail clustering with smaller eps
                tail_pred = np.array([], dtype=int)
                if tail_mask.sum() >= min_samples:
                    db_tail = DBSCAN(eps=eps_scaled * k_tail, min_samples=min_samples, n_jobs=-1)
                    tail_pred = db_tail.fit_predict(features_scaled[tail_mask])

                    # build core centroids (scaled space) for merging
                    core_centroids_scaled, core_cluster_ids = [], []
                    if core_mask.sum() and np.any(core_pred >= 0):
                        for lbl in np.unique(core_pred[core_pred >= 0]):
                            mask_lbl = core_pred == lbl
                            weights = (
                                data["energies"][event_subdet_mask][layer_mask][core_mask][mask_lbl].astype("f4")
                            )
                            feats = features_scaled[core_mask][mask_lbl]
                            wsum = float(weights.sum())
                            centroid_scaled = (feats * (weights[:, None] / (wsum if wsum > 0 else 1.0))).sum(axis=0)
                            core_centroids_scaled.append(centroid_scaled)
                            core_cluster_ids.append(lbl)
                        core_centroids_scaled = (
                            np.vstack(core_centroids_scaled).astype("f4")
                            if core_centroids_scaled
                            else np.empty((0, features_scaled.shape[1]), dtype="f4")
                        )
                    else:
                        core_centroids_scaled = np.empty((0, features_scaled.shape[1]), dtype="f4")

                    # for each tail cluster, merge to nearest core if within base eps; else make new label
                    good_tail = tail_pred >= 0
                    for lbl in np.unique(tail_pred[good_tail]):
                        mask_lbl = tail_pred == lbl
                        weights = (
                            data["energies"][event_subdet_mask][layer_mask][tail_mask][mask_lbl].astype("f4")
                        )
                        feats = features_scaled[tail_mask][mask_lbl]
                        wsum = float(weights.sum())
                        centroid_tail_scaled = (feats * (weights[:, None] / (wsum if wsum > 0 else 1.0))).sum(axis=0)

                        merged = False
                        if core_centroids_scaled.shape[0]:
                            dists = np.linalg.norm(core_centroids_scaled - centroid_tail_scaled[None, :], axis=1)
                            j = int(np.argmin(dists))
                            if dists[j] <= eps_scaled:
                                tail_pred[mask_lbl] = core_cluster_ids[j]
                                merged = True
                        if not merged:
                            tail_pred[mask_lbl] = total_clusters
                            total_clusters += 1

                    layer_labels[core_mask.sum() :] = tail_pred

                labels[global_indices_for_layer] = layer_labels

    print(f"Total clusters: {total_clusters}")
    return labels


# =============================================================================
# Point cloud creation / I/O
# =============================================================================
def create_point_cloud(data: Dict[str, np.ndarray], labels: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Create a clustered "steps" point cloud:
      - position: energy-weighted centroid per cluster
      - energy: sum per cluster 
      - time: min per cluster 
      - pdg: 0 
    """
    positions_out, energies_out, times_out = [], [], []
    pdgs_out, subdets_out, events_out = [], [], []

    unique_labels = set(labels)
    unique_labels.discard(-1)  # ignore noise

    for lbl in unique_labels:
        cluster_mask = labels == lbl
        pos = data["positions"][cluster_mask].astype("f4")
        ene = data["energies"][cluster_mask].astype("f4")
        tim = data["times"][cluster_mask].astype("f4")
        subd = data["subdetectors"][cluster_mask]
        ev = data["event_ids"][cluster_mask]

        # cluster must not span multiple events/subdetectors
        if len(np.unique(ev)) != 1 or len(np.unique(subd)) != 1:
            raise ValueError(f"Cluster spans multiple events/subdetectors (label={lbl})")

        totE = float(ene.sum())
        if totE > 0:
            centroid = (pos * (ene[:, None] / totE)).sum(axis=0)
        else:
            centroid = pos.mean(axis=0)

        positions_out.append(centroid.astype("f4"))
        energies_out.append(np.float32(totE))
        times_out.append(np.float32(tim.min()))
        pdgs_out.append(np.int32(0))          
        subdets_out.append(np.uint8(subd[0]))
        events_out.append(np.int32(ev[0]))

    return {
        "positions": np.asarray(positions_out, dtype="f4"),
        "energies": np.asarray(energies_out, dtype="f4"),
        "times": np.asarray(times_out, dtype="f4"),
        "pdgs": np.asarray(pdgs_out, dtype="i4"),
        "subdetectors": np.asarray(subdets_out, dtype="u1"),
        "event_ids": np.asarray(events_out, dtype="i4"),
    }


def save_point_cloud(pc: Dict[str, np.ndarray], out_path: str, metadata: Dict[str, np.ndarray]) -> None:
    # Write clustered point cloud to HDF5 in a similar format as original
    with h5py.File(out_path, "w") as f:
        g = f.create_group("steps")
        g.create_dataset("position", data=pc["positions"], compression="lzf", shuffle=True, chunks=True)
        g.create_dataset("energy", data=pc["energies"], compression="lzf", shuffle=True, chunks=True)
        g.create_dataset("time", data=pc["times"], compression="lzf", shuffle=True, chunks=True)
        g.create_dataset("pdg", data=pc["pdgs"])
        g.create_dataset("subdetector", data=pc["subdetectors"])
        g.create_dataset("event_id", data=pc["event_ids"])
        if metadata.get("subdetector_names") is not None:
            f.create_group("metadata").create_dataset("subdetector_names", data=metadata["subdetector_names"])
    print(f"\nSaved {len(pc['positions'])} clustered points to {out_path}")


# =============================================================================
# Pipeline
# =============================================================================
def process_first_n_events(
    in_path: str,
    out_path: str,
    eps_scaled: float = 1.5,
    min_samples: int = 5,
    n_events: int = 100,
    adaptive: bool = False,
) -> None:
    print(f"Loading first {n_events} events…")
    data, meta = load_first_n_events(in_path, n=n_events)
    print(f"Loaded {len(data['positions'])} steps from {len(np.unique(data['event_ids']))} events")

    labels = cluster_all_events(
        data, eps_scaled=eps_scaled, min_samples=min_samples, adaptive=adaptive
    )

    noise = int(np.sum(labels == -1))
    total = int(labels.size)
    print(f"Noise points: {noise}/{total} ({100.0 * noise / total:.2f}%)")

    pc = create_point_cloud(data, labels)
    save_point_cloud(pc, out_path, meta)

    orig = len(data["positions"])
    fin = len(pc["positions"])
    cr = (orig / fin) if fin > 0 else np.inf
    print(f"\nCompression ratio: {cr:.2f}x  (reduction {(1 - 1/cr) * 100:.1f}%)")
    print(f"Original: {orig:,}  →  Clustered: {fin:,}")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    DEFAULT_OUTDIR = "/eos/user/t/tnandi/fast_sim_work/Clustered-Output/dbscan-per-layer"

    parser = argparse.ArgumentParser(description="DBSCAN clustering of calorimeter steps with optional adaptive epsilon.")
    parser.add_argument("input", type=str, help="Input HDF5 file (unclustered steps)")
    parser.add_argument("--output", type=str, default=None, help="Output HDF5 path")
    parser.add_argument("--eps_scaled", type=float, default=1.5, help="DBSCAN epsilon in scaled units")
    parser.add_argument("--min_samples", type=int, default=5, help="DBSCAN min_samples")
    parser.add_argument("--n_events", type=int, default=100, help="Number of events to process")
    parser.add_argument("--adaptive", action="store_true", help="Enable two-zone adaptive epsilon per layer")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)

    output_path = args.output or os.path.join(
        DEFAULT_OUTDIR,
        f"dbscan_eps{args.eps_scaled}_min{args.min_samples}_n{args.n_events}{'_adaptive' if args.adaptive else ''}.h5",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    process_first_n_events(
        args.input,
        output_path,
        eps_scaled=args.eps_scaled,
        min_samples=args.min_samples,
        n_events=args.n_events,
        adaptive=args.adaptive,
    )
