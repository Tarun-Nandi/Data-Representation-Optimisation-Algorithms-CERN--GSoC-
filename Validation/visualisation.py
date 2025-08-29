import argparse
import math
import os
from pathlib import Path
from typing import Dict, Tuple, List
import h5py
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


# ---------------- I/O ----------------
def load_steps_h5(file_path: str) -> Dict[str, np.ndarray]:
    # Load required data drom hdf5 files
    with h5py.File(file_path, "r") as h5file:
        steps_grp = h5file["steps"]
        arrays: Dict[str, np.ndarray] = {
            "position": steps_grp["position"][:].astype("f4"), 
            "energy": steps_grp["energy"][:].astype("f4"),      
            "event_id": steps_grp["event_id"][:].astype("i4"),  
        }
        if "subdetector" in steps_grp:
            arrays["subdetector"] = steps_grp["subdetector"][:].astype("u1")
        else:
            arrays["subdetector"] = np.zeros(len(arrays["energy"]), dtype="u1")
    return arrays


def pick_event_ids(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray]) -> np.ndarray:
    # Return sorted array of common event IDs present in both datasets."""
    return np.intersect1d(np.unique(before["event_id"]), np.unique(after["event_id"]))


# ------------- framing & scales -------------
def energy_weighted_quantile(values: np.ndarray,weights: np.ndarray,quantiles: Tuple[float, float]) -> Tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.size == 0 or np.sum(w) <= 0:
        return float("nan"), float("nan")
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cum_w = np.cumsum(w)
    cum_w /= cum_w[-1]
    q_lo = float(np.interp(quantiles[0], cum_w, v))
    q_hi = float(np.interp(quantiles[1], cum_w, v))
    return q_lo, q_hi


def compute_view_ranges(
    positions_before: np.ndarray,
    energies_before: np.ndarray,
    positions_after: np.ndarray,
    energies_after: np.ndarray,
    q_lo: float,
    q_hi: float,
    pad_fraction: float = 0.08,
) -> Dict[str, Tuple[float, float]]:
    # Compute identical x/y/z ranges using combined data and weighted quantiles.
    if positions_before.size == 0 and positions_after.size == 0:
        return {"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)}

    all_positions = (np.vstack([positions_before, positions_after]) if positions_after.size else positions_before)
    all_weights = (np.hstack([energies_before, energies_after]) if positions_after.size else energies_before)

    x_vals, y_vals, z_vals = all_positions[:, 0], all_positions[:, 1], all_positions[:, 2]

    def _range(vals: np.ndarray) -> Tuple[float, float]:
        lo, hi = energy_weighted_quantile(vals, all_weights, (q_lo, q_hi))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.min(vals)), float(np.max(vals))
            if lo == hi:
                lo -= 1.0
                hi += 1.0
        span = hi - lo
        return lo - pad_fraction * span, hi + pad_fraction * span

    return {"x": _range(x_vals), "y": _range(y_vals), "z": _range(z_vals)}


def nice_dtick(v_min: float, v_max: float, target_ticks: int = 6) -> float:
    span = abs(v_max - v_min)
    if span <= 0:
        return 1.0
    raw = span / max(target_ticks, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 5, 10):
        if mult * magnitude >= raw:
            return mult * magnitude
    return 10 * magnitude


def axis_config(axis_title: str, axis_range: Tuple[float, float]) -> dict:
    """Return a consistent axis configuration dict for 3D scenes."""
    dtick = nice_dtick(axis_range[0], axis_range[1], target_ticks=6)
    tick0 = math.floor(axis_range[0] / dtick) * dtick
    return dict(
        title=dict(text=axis_title, font=dict(color="white")),
        tickfont=dict(color="white"),
        range=list(axis_range),
        tickmode="linear",
        tick0=tick0,
        dtick=dtick,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.15)",
        zeroline=False,
        showbackground=False,
    )


# ---------- marker sizing (fixed settings) ----------
def log10_energy(energy_mev: np.ndarray) -> np.ndarray:
    # Return log10 of energy (MeV) with a small positive clip to avoid -inf.
    return np.log10(np.clip(energy_mev, 1e-9, None))


def size_by_energy_fixed(energy_mev: np.ndarray,min_size: float,max_size: float,cap_percentile: float) -> List[float]:
    if energy_mev.size == 0:
        return []
    positive = energy_mev[energy_mev > 0]
    cap = np.percentile(positive, cap_percentile) if positive.size else 1.0
    scale = np.clip(energy_mev / (cap if cap > 0 else 1.0), 0.0, 1.0)
    sizes = min_size + (max_size - min_size) * np.sqrt(scale)
    return sizes.astype(float).tolist()


# ------------- figure building -------------
def common_color_range(loge_before: np.ndarray, loge_after: np.ndarray) -> Tuple[float, float]:
    # Determine shared (cmin, cmax) for the color scale from both datasets.
    ranges: List[Tuple[float, float]] = []
    if loge_before.size:
        ranges.append((float(np.nanmin(loge_before)), float(np.nanmax(loge_before))))
    if loge_after.size:
        ranges.append((float(np.nanmin(loge_after)), float(np.nanmax(loge_after))))
    if not ranges:
        return 0.0, 1.0
    mins, maxs = zip(*ranges)
    return float(min(mins)), float(max(maxs))


def make_side_by_side_figure(
    positions_before: np.ndarray,
    energies_before: np.ndarray,
    positions_after: np.ndarray,
    energies_after: np.ndarray,
    axis_ranges: Dict[str, Tuple[float, float]],
    figure_title: str,
) -> go.Figure:
    # Build the two-panel 3D figure with shared color and axis settings.
    # Shared color scale and range
    loge_before = log10_energy(energies_before)
    loge_after = log10_energy(energies_after)
    cmin, cmax = common_color_range(loge_before, loge_after)
    colorscale_name = "Viridis"

    # Fixed marker sizes (lists + 'diameter' mode so differences are obvious)
    sizes_before = size_by_energy_fixed(energies_before, min_size=2.5, max_size=9.0, cap_percentile=95.0)
    sizes_after = size_by_energy_fixed(energies_after, min_size=2.5, max_size=9.0, cap_percentile=95.0)
    sizes_after = [s * (12.0 / 9.0) for s in sizes_after]  # emphasize AFTER slightly

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.04,
        column_widths=[0.5, 0.5],  
        subplot_titles=("Before (steps)", "After (clustered)"),
    )

    # BEFORE trace
    if positions_before.size:
        fig.add_trace(
            go.Scatter3d(
                x=positions_before[:, 0],
                y=positions_before[:, 1],
                z=positions_before[:, 2],
                mode="markers",
                name="Before (steps)",
                legendgroup="before",
                marker=dict(
                    size=sizes_before,
                    sizemode="diameter",
                    sizemin=1.0,
                    line=dict(width=0),
                    color=loge_before,
                    colorscale=colorscale_name,
                    cmin=cmin,
                    cmax=cmax,
                    opacity=0.60,
                    showscale=False,  # single shared colorbar (on AFTER)
                ),
                hovertemplate=(
                    "x=%{x:.1f} mm<br>y=%{y:.1f} mm<br>z=%{z:.1f} mm"
                    "<br>E=%{customdata:.3g} MeV<extra></extra>"
                ),
                customdata=energies_before,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # AFTER trace
    if positions_after.size:
        fig.add_trace(
            go.Scatter3d(
                x=positions_after[:, 0],
                y=positions_after[:, 1],
                z=positions_after[:, 2],
                mode="markers",
                name="After (clustered)",
                legendgroup="after",
                marker=dict(
                    size=sizes_after,
                    sizemode="diameter",
                    sizemin=1.0,
                    line=dict(width=0),
                    color=loge_after,
                    colorscale=colorscale_name,
                    cmin=cmin,
                    cmax=cmax,
                    opacity=0.85,
                    colorbar=dict(
                        x=1.02,
                        tickcolor="white",
                        tickfont=dict(color="white"),
                        outlinecolor="rgba(255,255,255,0.3)",
                        title=dict(text="log10(E)", font=dict(color="white")),
                    ),
                ),
                hovertemplate=(
                    "x=%{x:.1f} mm<br>y=%{y:.1f} mm<br>z=%{z:.1f} mm"
                    "<br>E=%{customdata:.3g} MeV<extra></extra>"
                ),
                customdata=energies_after,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # IDENTICAL axes (ranges + ticks/grids) on both scenes
    scene_shared = dict(
        xaxis=axis_config("x [mm]", axis_ranges["x"]),
        yaxis=axis_config("y [mm]", axis_ranges["y"]),
        zaxis=axis_config("z [mm]", axis_ranges["z"]),
        aspectmode="data",
        bgcolor="black",
    )
    fig.update_layout(
        title=dict(text=figure_title, font=dict(color="white")),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    fig.update_layout(scene=scene_shared, scene2=scene_shared)

    # Same camera for both scenes
    camera = dict(center=dict(x=0, y=0, z=0), eye=dict(x=1.5, y=1.5, z=1.1))
    fig.update_layout(scene_camera=camera, scene2_camera=camera)

    return fig


# ---------------- CLI ----------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D interactive shower visualization (side-by-side: before vs. clustered)."
    )
    parser.add_argument("before", help="Input HDF5 with unclustered steps (BEFORE).")
    parser.add_argument("after", help="Input HDF5 with clustered steps (AFTER).")
    parser.add_argument("--event",type=int,default=None,help="Event ID to visualize; defaults to first common event.")
    parser.add_argument("--subdet", type=int, default=None, help="Optional subdetector filter (e.g., 0 for ECAL).")
    parser.add_argument("--qlo",type=float,default=0.02,help="Lower energy-weighted quantile for framing (default 0.02).")
    parser.add_argument("--qhi",type=float,default=0.98,help="Upper energy-weighted quantile for framing (default 0.98).")
    parser.add_argument("--output",type=str,default=None,help="Output HTML file (default: derived from inputs).")
    args = parser.parse_args()

    if not os.path.isfile(args.before):
        raise FileNotFoundError(args.before)
    if not os.path.isfile(args.after):
        raise FileNotFoundError(args.after)
    if not (0.0 <= args.qlo < args.qhi <= 1.0):
        raise ValueError("qlo and qhi must satisfy 0 <= qlo < qhi <= 1.")

    steps_before = load_steps_h5(args.before)
    steps_after = load_steps_h5(args.after)

    common_event_ids = pick_event_ids(steps_before, steps_after)
    if common_event_ids.size == 0:
        raise RuntimeError("No common event IDs between the two files.")

    event_id = args.event if args.event is not None else int(common_event_ids[0])
    if event_id not in common_event_ids:
        raise RuntimeError(f"Event {event_id} not present in BOTH files.")

    # Filter by event (and optional subdetector)
    mask_before = steps_before["event_id"] == event_id
    mask_after = steps_after["event_id"] == event_id
    if args.subdet is not None:
        mask_before &= steps_before["subdetector"] == args.subdet
        mask_after &= steps_after["subdetector"] == args.subdet

    positions_before = steps_before["position"][mask_before]
    energies_before = steps_before["energy"][mask_before]
    positions_after = steps_after["position"][mask_after]
    energies_after = steps_after["energy"][mask_after]

    if positions_before.size == 0 and positions_after.size == 0:
        raise RuntimeError("No steps to visualize after filtering by event/subdetector.")

    axis_ranges = compute_view_ranges(positions_before,energies_before,positions_after,energies_after,q_lo=args.qlo,q_hi=args.qhi)

    figure_title = f"Shower 3D — Event {event_id}"
    if args.subdet is not None:
        figure_title += f" (subdet {args.subdet})"

    fig = make_side_by_side_figure(positions_before,energies_before,positions_after,energies_after,axis_ranges,figure_title)

    output_html_path = (args.output or f"shower3d_side_by_side_{Path(args.before).stem}_vs_{Path(args.after).stem}_evt{event_id}.html")
    pio.write_html(fig, output_html_path, auto_open=True, include_plotlyjs="cdn")
    print(f"Saved interactive visualization to: {output_html_path}")


if __name__ == "__main__":
    main()
