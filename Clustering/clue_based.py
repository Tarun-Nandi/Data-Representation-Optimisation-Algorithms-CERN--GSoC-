import os, json, argparse
import numpy as np
import h5py
from scipy.spatial import cKDTree
from tqdm import tqdm

# ---- fixed scales & geometry (match DBSCAN) ----
XY_SCALE_MM = 5.0    # divide x,y by this (cell size)

def get_layer(cell_id):
    # Extract layer number from cell_id (9 bits starting at bit 19).
    return (int(cell_id) >> 19) & 0x1FF

def load_first_n_events(path, n=100):
    with h5py.File(path, 'r') as f:
        steps = f['steps']
        event_ids_all = steps['event_id'][:]
        kept_event_ids = np.unique(event_ids_all)[:n]
        keep_mask = np.isin(event_ids_all, kept_event_ids)
        data = {
            'positions':    steps['position'][:][keep_mask].astype('f4'),
            'energies':     steps['energy'][:][keep_mask].astype('f4'),
            'pdgs':         steps['pdg'][:][keep_mask].astype('i4'),
            'subdetectors': steps['subdetector'][:][keep_mask].astype('u1'),
            'event_ids':    steps['event_id'][:][keep_mask].astype('i4'),
            'cell_ids':     steps['cell_id'][:][keep_mask].astype('u8'),
        }
        meta = {}
        if 'metadata' in f and 'subdetector_names' in f['metadata']:
            meta['subdetector_names'] = f['metadata']['subdetector_names'][:]
    return data, meta

# ---------------------------
# CLUE core (hit-count density)
# ---------------------------
class ClueClusterer:
    def __init__(self, dc_mm=15.3, rhoc=5.0, deltac_mm=None, outlier_factor=None):
        self.dc_scaled     = float(dc_mm) / XY_SCALE_MM
        self.deltac_scaled = (float(deltac_mm) / XY_SCALE_MM) if (deltac_mm is not None) else self.dc_scaled
        self.rhoc          = float(rhoc)
        self.outlier_factor = None if outlier_factor in (None, 0) else float(outlier_factor)

    def _kdtree(self, feats):
        return cKDTree(feats)

    def _rho(self, feats):
        # rho_i = 1 + 0.5 * (#neighbors within dc, excluding self)
        tree = self._kdtree(feats)
        neigh = tree.query_ball_tree(tree, self.dc_scaled)
        rho = np.empty(len(feats), dtype='f8')
        for i, nb in enumerate(neigh):
            rho[i] = 1.0 + 0.5 * max(0, len(nb) - 1)
        return rho

    def _delta_and_nh(self, feats, rho):
        n = len(feats)
        delta = np.full(n, np.inf, dtype='f8')
        nh    = np.full(n, -1, dtype='i8')

        order = np.argsort(-rho)
        rank  = np.empty(n, dtype='i8'); rank[order] = np.arange(n)

        tree = self._kdtree(feats)
        max_search = (self.outlier_factor * self.dc_scaled) if (self.outlier_factor is not None) else (5 * self.dc_scaled)

        for idx, i in enumerate(order):
            if idx == 0:  # highest-density point
                continue
            # bounded neighbor search; cap k for speed
            k = min(idx + 1, 128)
            dists, inds = tree.query(feats[i], k=k, distance_upper_bound=max_search)
            if np.isscalar(dists):
                dists = np.array([dists]); inds = np.array([inds])
            for d, j in zip(dists, inds):
                if j < n and j != i and rank[j] < rank[i]:
                    delta[i] = d; nh[i] = j; break

        # fill inf with max finite or search bound
        inf = ~np.isfinite(delta)
        if np.any(inf):
            finite = delta[np.isfinite(delta)]
            fill = np.max(finite) if finite.size else max_search
            delta[inf] = fill
        return delta, nh

    def _classify(self, rho, delta):
        # 1=seed, 0=follower, -1=outlier (only if outlier_factor set)
        n  = len(rho)
        pt = np.zeros(n, dtype='i1')
        seed = (rho >= self.rhoc) & (delta >= self.deltac_scaled)
        pt[seed] = 1
        if self.outlier_factor is not None:
            out = (rho < self.rhoc) & (delta >= self.outlier_factor * self.dc_scaled)
            pt[out] = -1
        # ensure at least one seed if any non-outlier exists
        if not np.any(seed) and np.any(pt != -1):
            score = rho * delta; score[pt == -1] = -1.0
            best = int(np.argmax(score))
            if score[best] > 0:
                pt[best] = 1
        return pt

    def _assign(self, rho, nh, pt):
        n = len(rho)
        labels = np.full(n, -1, dtype='i8')
        seeds = np.where(pt == 1)[0]
        for cid, s in enumerate(seeds):
            labels[s] = cid
        for i in np.argsort(-rho):
            if pt[i] == -1 or labels[i] >= 0:
                continue
            j = nh[i]; chain = []
            while j != -1 and pt[j] != -1 and labels[j] == -1:
                chain.append(j); j = nh[j]
            if j != -1 and labels[j] >= 0:
                lab = labels[j]
                labels[i] = lab
                for k in chain:
                    labels[k] = lab
        return labels

    def cluster(self, feats):
        if feats.shape[0] == 0:
            return np.array([], dtype='i8')
        rho = self._rho(feats)
        delta, nh = self._delta_and_nh(feats, rho)
        pt = self._classify(rho, delta)
        return self._assign(rho, nh, pt)

# ---------------------------
# CLUE over events (per subdetector & layer) with progress
# ---------------------------
def cluster_all_events(data, dc=15.3, rhoc=5.0, deltac=None, outlier_factor=None, min_points=5):
    N = len(data['positions'])
    labels = np.full(N, -1, dtype='i8')
    total_clusters = 0

    clusterer = ClueClusterer(dc_mm=dc, rhoc=rhoc, deltac_mm=deltac, outlier_factor=outlier_factor)

    events = np.unique(data['event_ids'])
    with tqdm(total=len(events), desc="CLUE: events", unit="ev", dynamic_ncols=True) as pbar:
        for ev in events:
            event_mask = (data['event_ids'] == ev)
            for subdet_id in np.unique(data['subdetectors'][event_mask]):
                event_subdet_mask = event_mask & (data['subdetectors'] == subdet_id)

                positions_es = data['positions'][event_subdet_mask]
                cell_ids_es  = data['cell_ids'][event_subdet_mask]
                if positions_es.shape[0] < min_points:
                    continue

                layers = np.fromiter((get_layer(c) for c in cell_ids_es), dtype=np.int32, count=positions_es.shape[0])
                for layer_id in np.unique(layers):
                    layer_mask = (layers == layer_id)
                    if layer_mask.sum() < min_points:
                        continue

                    xy_mm_layer = positions_es[layer_mask, :2].astype('f4')
                    features_scaled = (xy_mm_layer / XY_SCALE_MM).astype('f4')  # 2D CLUE (unitless)

                    local_labels = clusterer.cluster(features_scaled)

                    is_cluster = (local_labels >= 0)
                    if np.any(is_cluster):
                        # offset to global
                        mapping = {}
                        for sid in np.unique(local_labels[is_cluster]):
                            mapping[int(sid)] = total_clusters
                            total_clusters += 1
                        local_labels[is_cluster] = np.vectorize(mapping.get)(local_labels[is_cluster])

                    global_indices_for_layer = np.where(event_subdet_mask)[0][layer_mask]
                    labels[global_indices_for_layer] = local_labels

            pbar.set_postfix_str(f"clusters={total_clusters}")
            pbar.update(1)

    return labels

# ---------------------------
# Point cloud reducer
# ---------------------------
def create_point_cloud(data, labels):
    positions, energies, pdgs, subdets, events = [], [], [], [], []
    labs = set(labels.tolist()); labs.discard(-1)
    for L in labs:
        m = (labels == L)
        pos = data['positions'][m].astype('f4')
        ene = data['energies'][m].astype('f4')
        sde = data['subdetectors'][m]
        ev  = data['event_ids'][m]
        if len(np.unique(ev)) != 1 or len(np.unique(sde)) != 1:
            continue
        Etot = float(ene.sum())
        wpos = (pos * (ene[:, None] / (Etot if Etot > 0 else 1.0))).sum(axis=0) if Etot > 0 else pos.mean(axis=0)
        positions.append(wpos.astype('f4'))
        energies.append(np.float32(Etot))
        pdgs.append(np.int32(0))  # EM not used downstream
        subdets.append(np.uint8(sde[0]))
        events.append(np.int32(ev[0]))
    return {
        'positions':    np.asarray(positions, dtype='f4'),
        'energies':     np.asarray(energies,  dtype='f4'),
        'pdgs':         np.asarray(pdgs,      dtype='i4'),
        'subdetectors': np.asarray(subdets,   dtype='u1'),
        'event_ids':    np.asarray(events,    dtype='i4'),
    }

def save_point_cloud(point_cloud, out_path, metadata, attrs):
    with h5py.File(out_path, 'w') as f:
        g = f.create_group('steps')
        g.create_dataset('position',    data=point_cloud['positions'],    compression='lzf', shuffle=True, chunks=True)
        g.create_dataset('energy',      data=point_cloud['energies'],     compression='lzf', shuffle=True, chunks=True)
        g.create_dataset('pdg',         data=point_cloud['pdgs'])
        g.create_dataset('subdetector', data=point_cloud['subdetectors'])
        g.create_dataset('event_id',    data=point_cloud['event_ids'])
        if metadata.get('subdetector_names') is not None:
            f.create_group('metadata').create_dataset('subdetector_names', data=metadata['subdetector_names'])
        f.attrs['units'] = np.string_('x,y,z: mm; energy: MeV')
        f.attrs['cluster_params'] = np.string_(json.dumps(attrs))

# ---------------------------
# Pipeline
# ---------------------------
def process_first_n_events(in_path, out_path, dc=15.3, rhoc=5.0, deltac=None,
                           outlier_factor=None, n_events=100, min_points=5):
    data, meta = load_first_n_events(in_path, n=n_events)

    labels = cluster_all_events(
        data, dc=dc, rhoc=rhoc, deltac=deltac,
        outlier_factor=outlier_factor, min_points=min_points
    )
    point_cloud = create_point_cloud(data, labels)

    attrs = {
        'algorithm': 'CLUE',
        'dc_mm': float(dc),
        'deltac_mm': float(deltac) if deltac is not None else None,
        'rhoc': float(rhoc),
        'outlier_factor': None if outlier_factor in (None, 0) else float(outlier_factor),
        'xy_scale_mm': float(XY_SCALE_MM),
        'layer_bits': {'start': 19, 'width': 9},
        'features': '[(x/5mm),(y/5mm)]',
    }
    save_point_cloud(point_cloud, out_path, meta, attrs)

    # minimal summary
    orig = len(data['positions']); fin = len(point_cloud['positions'])
    cr = (orig / fin) if fin > 0 else np.inf
    print(f"Saved {out_path}")
    print(f"Points: {orig:,} → {fin:,}  |  Compression: {cr:.2f}×")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CLUE clustering (unitless; minimal output) with tqdm progress.")
    p.add_argument('input', type=str, help='Input HDF5 file (unclustered steps)')
    p.add_argument('--output', type=str, default=None, help='Output HDF5 path')
    p.add_argument('--dc', type=float, default=15.3, help='Critical distance in mm (unscaled)')
    p.add_argument('--deltac', type=float, default=None, help='Separation threshold in mm (default: == dc)')
    p.add_argument('--rhoc', type=float, default=5.0, help='Seed density threshold (hit-count density)')
    p.add_argument('--outlier_factor', type=float, default=None, help='Enable outlier labelling with factor×dc; omit/0 to disable')
    p.add_argument('--n_events', type=int, default=100, help='Number of events to process')
    p.add_argument('--min_points', type=int, default=5, help='Minimum points per (event,subdet,layer)')
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)

    DEFAULT_OUTDIR = '/eos/user/t/tnandi/fast_sim_work/Clustered-Output/clue-clean'
    output_path = args.output or os.path.join(
        DEFAULT_OUTDIR,
        f"clue_dc{args.dc}_rhoc{args.rhoc}{'_dt' if args.deltac is not None else ''}{('_of'+str(args.outlier_factor)) if args.outlier_factor else ''}_n{args.n_events}.h5"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    process_first_n_events(
        args.input, output_path,
        dc=args.dc, rhoc=args.rhoc, deltac=args.deltac,
        outlier_factor=args.outlier_factor,
        n_events=args.n_events, min_points=args.min_points
    )
