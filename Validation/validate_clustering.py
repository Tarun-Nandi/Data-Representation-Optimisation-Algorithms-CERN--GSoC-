import argparse
import os
import pathlib
from typing import Iterable
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# Constants
# =============================================================================
DEFAULT_LONG_BINS = 30
DEFAULT_RADIAL_BINS = 50
DEFAULT_LOGE_RANGE = (-8.0, 2.0)
DEFAULT_LOGE_BINS = 50


# =============================================================================
# Helpers
# =============================================================================
def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    # Wrap angle(s) to (-π, π].
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def circular_mean(angles: np.ndarray, weights: np.ndarray | None = None) -> float:
    if angles.size == 0:
        return 0.0
    if weights is None:
        weights = np.ones_like(angles, dtype=np.float64)

    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return 0.0

    c = np.sum(np.cos(angles) * weights) / weight_sum
    s = np.sum(np.sin(angles) * weights) / weight_sum
    return float(np.arctan2(s, c))


def load_steps_as_df(filename: str, ecal_only: bool = True) -> pd.DataFrame:
    # Load step-level data from HDF5 into a pandas DataFrame.
    with h5py.File(filename, "r") as f:
        steps = f["steps"]
        df = pd.DataFrame(
            {
                "event_id": steps["event_id"][:],
                "energy": steps["energy"][:],
                "x": steps["position"][:, 0],
                "y": steps["position"][:, 1],
                "z": steps["position"][:, 2],
            }
        )

        if ecal_only:
            if "subdetector" in steps:
                is_ecal = steps["subdetector"][:] == 0
                df = df[is_ecal].copy()
                print(f"[{os.path.basename(filename)}] ECAL-only: kept {len(df)} rows.")
            else:
                print(f"[{os.path.basename(filename)}] Warning: no 'subdetector' dataset; ")
        return df


# =============================================================================
# Axis finding (BEFORE only)
# =============================================================================
def pca_axis_energy_weighted(positions: np.ndarray, energies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Energy-weighted PCA axis (principal direction) and centroid.
    energy_sum = float(np.sum(energies))
    centroid = (np.average(positions, weights=energies, axis=0)if energy_sum > 0 else np.mean(positions, axis=0))
    centered_positions = positions - centroid
    if len(positions) < 2:
        return centroid, np.array([0.0, 0.0, 1.0], dtype=np.float64)

    weights = energies if energy_sum > 0 else np.ones(len(positions))
    total_weight = float(np.sum(weights)) or 1.0
    weighted_positions = centered_positions * np.sqrt((weights / total_weight)[:, None])
    covariance = weighted_positions.T @ weighted_positions
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    # Fix an arbitrary global sign to keep visuals stable
    if np.dot(axis, np.array([1.0, 1.0, 1.0])) < 0.0:
        axis = -axis
    return centroid, axis


def refined_pca_axis_from_before(df_before: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine the PCA axis: compute initial axis, trim long-projection to [10, 99]%
    if enough hits, and recompute. Improves robustness to outliers.
    """
    positions = df_before[["x", "y", "z"]].values
    energies = df_before["energy"].values
    if len(positions) == 0:
        return np.zeros(3), np.array([0.0, 0.0, 1.0])

    centroid0, axis0 = pca_axis_energy_weighted(positions, energies)
    longitudinal0 = (positions - centroid0) @ axis0
    if len(longitudinal0) >= 10:
        low_quantile, high_quantile = np.percentile(longitudinal0, [10, 99])
        in_trim_range = (longitudinal0 >= low_quantile) & (longitudinal0 <= high_quantile)
        if np.count_nonzero(in_trim_range) >= 2:
            return pca_axis_energy_weighted(positions[in_trim_range], energies[in_trim_range])
    return centroid0, axis0


# =============================================================================
# Coordinate projection
# =============================================================================
def project_to_axis(df: pd.DataFrame, centroid: np.ndarray, axis: np.ndarray) -> pd.DataFrame:
    # Project positions to (longitudinal, r, phi) around axis through centroid.
    positions = df[["x", "y", "z"]].values
    energies = df["energy"].values
    rel = positions - centroid
    longitudinal = rel @ axis

    # Transverse basis
    temp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    basis1 = np.cross(axis, temp)
    basis1 /= (np.linalg.norm(basis1) + 1e-12)
    basis2 = np.cross(axis, basis1)
    basis2 /= (np.linalg.norm(basis2) + 1e-12)

    u = rel @ basis1
    v = rel @ basis2
    radial = np.sqrt(u * u + v * v)

    # Reference for phi (only used for centering)
    reference_vector = np.array([1.0, 0.0, 0.0])
    reference_proj = reference_vector - (reference_vector @ axis) * axis
    if np.linalg.norm(reference_proj) > 1e-9:
        reference_proj = reference_proj / np.linalg.norm(reference_proj)

    transverse_vectors = np.outer(u, basis1) + np.outer(v, basis2)  # Nx3
    phi = np.arctan2(np.cross(np.tile(reference_proj, (len(transverse_vectors), 1)), transverse_vectors).dot(axis),transverse_vectors.dot(reference_proj))

    return pd.DataFrame({"long": longitudinal, "r": radial, "phi": phi, "energy": energies})


# =============================================================================
# Cuts and statistics
# =============================================================================
def shared_cut_masks(df_before_cyl: pd.DataFrame,df_after_cyl: pd.DataFrame,q_low: float = 0.01,q_high: float = 0.99):
    # Compute (long, r, phi) quantile cuts from BEFORE and return boolean masks for BEFORE & AFTER.
    
    cut_bounds: dict[str, tuple[float, float]] = {}
    for col in ("long", "r", "phi"):
        low, high = np.quantile(df_before_cyl[col].values, [q_low, q_high])
        cut_bounds[col] = (float(low), float(high))

    def mask_from_cuts(df_cyl: pd.DataFrame) -> np.ndarray:
        mask = np.ones(len(df_cyl), dtype=bool)
        for col, (low, high) in cut_bounds.items():
            values = df_cyl[col].values
            mask &= (values >= low) & (values <= high)
        return mask

    return mask_from_cuts(df_before_cyl), mask_from_cuts(df_after_cyl), cut_bounds


def weighted_quantile(values: np.ndarray,quantiles: np.ndarray | Iterable[float],sample_weight: np.ndarray | None = None) -> np.ndarray:
    # Weighted quantile(s); returns NaNs if empty or total weight ≤ 0.
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    weights = (np.ones_like(values, dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64))

    if values.size == 0 or np.sum(weights) <= 0:
        return np.full_like(quantiles, np.nan, dtype=np.float64)

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cum_weights = np.cumsum(sorted_weights)
    cum_weights /= cum_weights[-1]
    return np.interp(quantiles, cum_weights, sorted_values, left=sorted_values[0], right=sorted_values[-1])


# =============================================================================
# Per-event histograms and moments
# =============================================================================
def event_histograms(df_cyl: pd.DataFrame, long_edges: np.ndarray, radial_edges: np.ndarray, loge_edges: np.ndarray):
    # Energy-weighted long/r profiles and count histogram in log10(E).
    energies = df_cyl["energy"].values
    hist_longitudinal, _ = np.histogram(df_cyl["long"].values, bins=long_edges, weights=energies)
    hist_radial, _ = np.histogram(df_cyl["r"].values, bins=radial_edges, weights=energies)
    hist_log_energy, _ = np.histogram(np.log10(energies + 1e-12), bins=loge_edges)  # counts
    return (hist_longitudinal, long_edges), (hist_radial, radial_edges), (hist_log_energy, loge_edges)


def energy_weighted_moments(df_cyl: pd.DataFrame) -> tuple[float, float, float, float]:
    # Raw first and second moments (⟨L⟩, ⟨L²⟩, ⟨r⟩, ⟨r²⟩), energy-weighted.
    energies = df_cyl["energy"].values
    if energies.size == 0 or np.sum(energies) <= 0:
        return 0.0, 0.0, 0.0, 0.0

    energy_sum = float(np.sum(energies))
    longitudinal = df_cyl["long"].values
    radial = df_cyl["r"].values
    mean_longitudinal = float(np.sum(energies * longitudinal) / energy_sum)
    second_moment_longitudinal = float(np.sum(energies * longitudinal**2) / energy_sum)
    mean_radial = float(np.sum(energies * radial) / energy_sum)
    second_moment_radial = float(np.sum(energies * radial**2) / energy_sum)
    return mean_longitudinal, second_moment_longitudinal, mean_radial, second_moment_radial


# =============================================================================
# Plotting
# =============================================================================
def _avg_profile(event_profiles: list[dict], key: str) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Average per-event histogram (aligned-bin stairs).
    Energy profiles normalized to per-event kept energy; logE normalized by object count.
    """
    if not event_profiles:
        return np.array([]), None

    edges = event_profiles[0][key][1]
    per_event_normalized: list[np.ndarray] = []

    for profile in event_profiles:
        hist = profile[key][0].astype(float)

        if key in ("long_profile", "r_profile"):
            denominator = profile.get("evt_energy_kept")
            if denominator is None or denominator <= 0:
                denominator = np.sum(hist) if np.sum(hist) > 0 else 1.0
            hist = hist / denominator

        if key == "log_energy":
            n_objects = float(profile.get("num_objects", 0)) or 1.0
            hist = hist / n_objects

        per_event_normalized.append(hist)

    return np.mean(np.vstack(per_event_normalized), axis=0), edges


def plot_profiles_stairs(profiles_before: list[dict],profiles_after: list[dict],n_events: int,outpath: str = "comparison.png") -> None:
    # Plot energy-weighted longitudinal/radial profiles and log10(E) counts.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        ("long_profile", "longitudinal (mm)", False),
        ("r_profile", "radial (mm)", True),
        ("log_energy", "log10(E)", False),
    ]

    for ax, (key, xlabel, use_log_y) in zip(axes, panels):
        mean_before, edges_before = _avg_profile(profiles_before, key)
        mean_after, edges_after = _avg_profile(profiles_after, key)

        if edges_before is not None and mean_before.size:
            ax.stairs(mean_before, edges_before, label=f"Before (avg over {n_events} events)")
        if edges_after is not None and mean_after.size:
            ax.stairs(mean_after, edges_after, label=f"After (avg over {n_events} events)")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized energy" if key != "log_energy" else "Fraction of objects")
        if use_log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3) 
        ax.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print("Saved:", os.path.abspath(outpath))
    plt.close(fig)


def make_common_bins(before_values: np.ndarray, after_values: np.ndarray, nbins: int = 40, pad: float = 0.05) -> np.ndarray:
    # Create common bin edges covering both arrays with a small padding.
    combined = np.concatenate([before_values, after_values]) if after_values.size else before_values
    if combined.size == 0:
        return np.linspace(0, 1, nbins + 1)

    min_value, max_value = float(np.min(combined)), float(np.max(combined))
    if min_value == max_value:
        min_value -= 1e-6
        max_value += 1e-6

    value_range = max_value - min_value
    min_value -= pad * value_range
    max_value += pad * value_range
    return np.linspace(min_value, max_value, nbins + 1)


def plot_moment_histograms(moments_before: dict[str, np.ndarray],moments_after: dict[str, np.ndarray],outpath: str = "comparison_moments.png",nbins: int = 40,) -> None:
    # Plot raw moment distributions for BEFORE vs AFTER.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    specs = [
        ("m1L", "⟨long⟩ (mm)"),
        ("m2L", "⟨long²⟩ (mm²)"),
        ("m1R", "⟨r⟩ (mm)"),
        ("m2R", "⟨r²⟩ (mm²)"),
    ]

    for ax, (key, xlabel) in zip(axes.flat, specs):
        before_vals = moments_before[key]
        after_vals = moments_after[key]
        edges = make_common_bins(before_vals, after_vals, nbins=nbins, pad=0.05)

        count_before, _ = np.histogram(before_vals, bins=edges)
        count_after, _ = np.histogram(after_vals, bins=edges)
        bin_widths = np.diff(edges)

        n_before = max(len(before_vals), 1)
        n_after = max(len(after_vals), 1)
        density_before = count_before / (n_before * bin_widths)
        density_after = count_after / (n_after * bin_widths)

        ax.stairs(density_before, edges, label="Before")
        ax.stairs(density_after, edges, label="After")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability density")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print("Saved:", os.path.abspath(outpath))
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main(
    before_file: str,
    after_file: str,
    direction: list[float] | tuple[float, float, float] | None = None,
    n_events: int | None = None,
    output: str | None = None,
    q_low: float = 0.01,
    q_high: float = 0.99,
    anchor: str = "before_min",  
    ecal_only: bool = True,
) -> None:
    print("Loading step data...")
    steps_before = load_steps_as_df(before_file, ecal_only=ecal_only)
    steps_after = load_steps_as_df(after_file, ecal_only=ecal_only)

    common_event_ids = np.intersect1d(steps_before.event_id.unique(), steps_after.event_id.unique())
    if n_events is not None:
        common_event_ids = common_event_ids[:n_events]
    print(f"Found {len(common_event_ids)} common events. Processing {len(common_event_ids)}...")

    profiles_before: list[dict] = []
    profiles_after: list[dict] = []
    moments_before = {"m1L": [], "m2L": [], "m1R": [], "m2R": []}
    moments_after = {"m1L": [], "m2L": [], "m1R": [], "m2R": []}

    # Print-only stats
    energy_closure_ratios: list[float] = []  # E_after / E_before per event (no cuts)
    frac_kept_before: list[float] = []
    frac_kept_after: list[float] = []
    containment_before = {"R50": [], "R80": [], "R90": [], "w10": [], "w50": [], "w90": []}
    containment_after = {"R50": [], "R80": [], "R90": [], "w10": [], "w50": [], "w90": []}

    # Accurate object counters (across processed events)
    total_objects_before = 0  # raw step count BEFORE clustering
    total_objects_after = 0  # clustered step count AFTER clustering

    for event_id in tqdm(common_event_ids, desc="Per-event processing"):
        before_event_steps = steps_before[steps_before.event_id == event_id]
        after_event_steps = steps_after[steps_after.event_id == event_id]
        if len(before_event_steps) == 0 or len(after_event_steps) == 0:
            continue

        # Accurate per-event object totals (no cuts)
        total_objects_before += len(before_event_steps)
        total_objects_after += len(after_event_steps)

        # Raw event energies (no cuts)
        total_energy_before = float(before_event_steps["energy"].sum())
        total_energy_after = float(after_event_steps["energy"].sum())
        if total_energy_before > 0:
            energy_closure_ratios.append(total_energy_after / total_energy_before)

        # 1) Axis
        if direction is not None:
            axis = np.asarray(direction, dtype=np.float64)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            centroid = np.average(before_event_steps[["x", "y", "z"]].values,axis=0,weights=before_event_steps["energy"].values)
        else:
            centroid, axis = refined_pca_axis_from_before(before_event_steps)

        # 2) Project
        cyl_before = project_to_axis(before_event_steps, centroid, axis)
        cyl_after = project_to_axis(after_event_steps, centroid, axis)

        # 3) Phi centering (use BEFORE)
        phi_center = circular_mean(cyl_before["phi"].values, weights=cyl_before["energy"].values)
        cyl_before["phi"] = wrap_to_pi(cyl_before["phi"].values - phi_center)
        cyl_after["phi"] = wrap_to_pi(cyl_after["phi"].values - phi_center)

        # 4) Shared cuts
        mask_before, mask_after, _ = shared_cut_masks(cyl_before, cyl_after, q_low=q_low, q_high=q_high)

        # f_keep BEFORE/AFTER using shared cuts
        kept_energy_before = float(cyl_before.loc[mask_before, "energy"].sum())
        kept_energy_after = float(cyl_after.loc[mask_after, "energy"].sum())
        frac_kept_before.append(kept_energy_before / total_energy_before if total_energy_before > 0 else np.nan)
        frac_kept_after.append(kept_energy_after / total_energy_after if total_energy_after > 0 else np.nan)

        # Apply cuts (for plotting/moments/containment)
        cyl_before = cyl_before[mask_before].reset_index(drop=True)
        cyl_after = cyl_after[mask_after].reset_index(drop=True)

        # 5) Re-anchoring
        if anchor == "before_min" and len(cyl_before) > 0:
            min_longitudinal = float(np.min(cyl_before["long"].values))
            cyl_before["long"] = cyl_before["long"].values - min_longitudinal
            cyl_after["long"] = cyl_after["long"].values - min_longitudinal

        # 6) Aligned bins (derive from BEFORE)
        if len(cyl_before) > 0:
            long_edges = np.histogram_bin_edges(cyl_before["long"].values, bins=DEFAULT_LONG_BINS)
            radial_edges = np.histogram_bin_edges(cyl_before["r"].values, bins=DEFAULT_RADIAL_BINS)
        else:
            long_edges = np.linspace(0, 1, DEFAULT_LONG_BINS + 1)
            radial_edges = np.linspace(0, 1, DEFAULT_RADIAL_BINS + 1)

        loge_edges = np.linspace(*DEFAULT_LOGE_RANGE, DEFAULT_LOGE_BINS + 1)

        # 7) Histograms
        (hist_long_before, long_edges), (hist_rad_before, radial_edges), hist_logE_before = event_histograms(cyl_before, long_edges, radial_edges, loge_edges)
        (hist_long_after, _), (hist_rad_after, _), hist_logE_after = event_histograms(cyl_after, long_edges, radial_edges, loge_edges)

        # 8) Moments (raw only)
        m1L_before, m2L_before, m1R_before, m2R_before = energy_weighted_moments(cyl_before)
        m1L_after, m2L_after, m1R_after, m2R_after = energy_weighted_moments(cyl_after)
        for key, val in zip(("m1L", "m2L", "m1R", "m2R"), (m1L_before, m2L_before, m1R_before, m2R_before)):
            moments_before[key].append(val)
        for key, val in zip(("m1L", "m2L", "m1R", "m2R"), (m1L_after, m2L_after, m1R_after, m2R_after)):
            moments_after[key].append(val)

        # 9) Containment (after cuts + anchoring)
        for dct, radial_vals, long_vals, weights in (
            (containment_before, cyl_before["r"].values, cyl_before["long"].values, cyl_before["energy"].values),
            (containment_after, cyl_after["r"].values, cyl_after["long"].values, cyl_after["energy"].values),
        ):
            qs_r = weighted_quantile(radial_vals, [0.5, 0.8, 0.9], weights)
            qs_w = weighted_quantile(long_vals, [0.1, 0.5, 0.9], weights)
            dct["R50"].append(qs_r[0]); dct["R80"].append(qs_r[1]); dct["R90"].append(qs_r[2])
            dct["w10"].append(qs_w[0]); dct["w50"].append(qs_w[1]); dct["w90"].append(qs_w[2])

        # Package per-event observables for profiles
        profiles_before.append(
            {
                "long_profile": (hist_long_before, long_edges),
                "r_profile": (hist_rad_before, radial_edges),
                "log_energy": hist_logE_before,
                "evt_energy_kept": float(np.sum(hist_long_before)),
                "num_objects": int(len(cyl_before)),
            }
        )
        profiles_after.append(
            {
                "long_profile": (hist_long_after, long_edges),
                "r_profile": (hist_rad_after, radial_edges),
                "log_energy": hist_logE_after,
                "evt_energy_kept": float(np.sum(hist_long_after)),
                "num_objects": int(len(cyl_after)),
            }
        )

    n_processed = len(profiles_before)
    if n_processed == 0:
        print("No events processed. Exiting.")
        return

    # Convert to arrays
    for dct in (moments_before, moments_after, containment_before, containment_after):
        for k in dct:
            dct[k] = np.asarray(dct[k], dtype=float)
    energy_closure_ratios = np.asarray(energy_closure_ratios, dtype=float)
    frac_kept_before = np.asarray(frac_kept_before, dtype=float)
    frac_kept_after = np.asarray(frac_kept_after, dtype=float)

    # --- Accurate BEFORE/AFTER object counts & reduction ---
    print("\n--- Object counts (across processed common events) ---")
    print(f"Processed events: {n_processed}")
    print(f"Raw steps BEFORE : {total_objects_before:,}")
    print(f"Clusters AFTER   : {total_objects_after:,}")
    if total_objects_before > 0:
        reduction_abs = total_objects_before - total_objects_after
        reduction_pct = 100.0 * (1.0 - (total_objects_after / total_objects_before))
        sign = "-" if reduction_abs >= 0 else "+"
        print(
            f"Reduction (AFTER vs BEFORE): {sign}{abs(reduction_abs):,} objects "
            f"({reduction_pct:.2f}%)"
        )
    else:
        print("Reduction (AFTER vs BEFORE): n/a (no BEFORE objects)")
    print("------------------------------------------------------\n")

    # Output paths
    base_output = "comparison.png" if (output is None or str(output).strip() == "") else output
    base_path = pathlib.Path(base_output)
    moments_output_path = base_path.with_name(base_path.stem + "_moments" + base_path.suffix)

    # Figure 1: profiles 
    plot_profiles_stairs(profiles_before, profiles_after, n_processed, outpath=str(base_path))

    # Figure 2: moment histograms (raw moments only)
    plot_moment_histograms(moments_before, moments_after, outpath=str(moments_output_path), nbins=40)

    # -------- Print-only metrics (no plots) --------
    def _mean_rms(values: np.ndarray) -> tuple[float, float]:
        values = values[np.isfinite(values)]
        return (np.nan, np.nan) if values.size == 0 else (float(np.mean(values)), float(np.std(values, ddof=1)))

    if energy_closure_ratios.size:
        mean_ratio, rms_ratio = _mean_rms(energy_closure_ratios)
        print(f"[Energy closure] E_after/E_before: mean={mean_ratio:.6f}, RMS={rms_ratio:.6f}")

    mean_keep_before, rms_keep_before = _mean_rms(frac_kept_before)
    mean_keep_after, rms_keep_after = _mean_rms(frac_kept_after)
    print(f"[f_keep] BEFORE: mean={mean_keep_before:.4f}, RMS={rms_keep_before:.4f}")
    print(f"[f_keep] AFTER : mean={mean_keep_after:.4f}, RMS={rms_keep_after:.4f}")

    print("\n[Containment Δ medians] (After − Before)")
    for key in ("R50", "R80", "R90", "w10", "w50", "w90"):
        med_b = np.nanmedian(containment_before[key]) if containment_before[key].size else np.nan
        med_a = np.nanmedian(containment_after[key]) if containment_after[key].size else np.nan
        delta_median = med_a - med_b if (np.isfinite(med_b) and np.isfinite(med_a)) else np.nan
        print(f" {key:>3}: Δmedian = {delta_median:+.3f} mm (Before={med_b:.3f}, After={med_a:.3f})")

    print("\nDone.")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Profiles + moment histograms; print-only stats for energy closure, ""f_keep, containment."))
    parser.add_argument("before", help="HDF5 with raw steps (BEFORE clustering)")
    parser.add_argument("after", help="HDF5 with clustered steps (AFTER clustering)")
    parser.add_argument("--direction",nargs=3,type=float,default=None,help="Optional fixed axis (x y z). If omitted, PCA axis from BEFORE is used per event.")
    parser.add_argument("--n_events", type=int, default=None, help="Limit to this many common events.")
    parser.add_argument("--output",type=str,default=None,help="Base output filename (e.g., comparison.png). Moments saved as *_moments.png")
    parser.add_argument("--q_low",type=float,default=0.01,help="Shared lower quantile for outlier cuts (computed from BEFORE).")
    parser.add_argument("--q_high",type=float,default=0.99,help="Shared upper quantile for outlier cuts (computed from BEFORE).")
    parser.add_argument("--anchor",choices=["before_min", "none"],default="before_min",help=("Re-anchoring of longitudinal coordinate. 'before_min' subtracts BEFORE's ""min(long) from BOTH datasets."))
    parser.add_argument("--no-ecal-only",dest="ecal_only",action="store_false",help="Disable ECAL-only selection.")
    args = parser.parse_args()

    main(
        args.before,
        args.after,
        direction=args.direction,
        n_events=args.n_events,
        output=args.output,
        q_low=args.q_low,
        q_high=args.q_high,
        anchor=args.anchor,
        ecal_only=args.ecal_only,
    )
