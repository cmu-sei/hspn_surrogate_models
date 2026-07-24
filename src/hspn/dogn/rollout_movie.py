#!/bin/sh
#
# HyperSPIN code - hspn_surrogate_models
# 
# Copyright 2025 Carnegie Mellon University.
# 
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
# 
# Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
# 
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
# 
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
# 
# DM25-0396

"""PVD to MP4 Movie Converter
Converts PVD files with mesh data to MP4 movies showing prediction, truth, and residual.
Carefully written to prevent any data mixing between fields.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pyvista as pv

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm


def check_if_2d(mesh):
    """Check if mesh is essentially 2D (all Z values are zero or nearly zero)."""
    z_values = mesh.points[:, 2]
    z_range = z_values.max() - z_values.min()
    return np.allclose(z_values, 0) or z_range < 1e-6 * mesh.length


def get_field_data(mesh, field_name):
    """Safely extract field data from mesh, checking both point and cell data."""
    if field_name in mesh.point_data:
        return mesh.point_data[field_name].copy()  # Always return a copy
    elif field_name in mesh.cell_data:
        return mesh.cell_data[field_name].copy()  # Always return a copy
    else:
        raise ValueError(f"Field '{field_name}' not found in mesh")


def calculate_color_limits(
    reader, time_values, prediction_field, truth_field, residual_field
):
    """Calculate color limits from first and last timesteps only."""
    logger.info("Calculating color limits from first and last timesteps...")

    # Lists to collect data
    pred_values = []
    truth_values = []
    residual_values = []

    # Sample only first and last timesteps
    for idx, t in enumerate([time_values[0], time_values[-1]]):
        logger.info(f"Analyzing timestep {idx+1}/2 (t={t:.3f})")

        reader.set_active_time_value(t)
        mesh = reader.read()
        if isinstance(mesh, pv.MultiBlock):
            mesh = mesh[0]

        # Get prediction data
        try:
            pred_data = get_field_data(mesh, prediction_field)
            pred_values.extend(pred_data.flatten())
        except ValueError as e:
            logger.error(f"Prediction field error at t={t}: {e}")
            raise

        # Get truth data
        try:
            truth_data = get_field_data(mesh, truth_field)
            truth_values.extend(truth_data.flatten())
        except ValueError as e:
            logger.error(f"Truth field error at t={t}: {e}")
            raise

        # Get or compute residual data
        try:
            residual_data = get_field_data(mesh, residual_field)
        except ValueError:
            # Compute residual if not present
            logger.info(f"Computing residual at t={t}")
            residual_data = pred_data - truth_data

        residual_values.extend(residual_data.flatten())

        # Debug: Check that prediction and truth are different
        diff = np.abs(pred_data - truth_data).max()
        logger.info(f"Max difference between prediction and truth at t={t}: {diff:.6e}")

    # Convert to arrays
    pred_values = np.array(pred_values)
    truth_values = np.array(truth_values)
    residual_values = np.array(residual_values)

    # Calculate combined limits for prediction and truth
    combined_values = np.concatenate([pred_values, truth_values])
    prediction_truth_clim = (
        np.percentile(combined_values, 2),
        np.percentile(combined_values, 98),
    )

    # Calculate residual limits (symmetric)
    residual_percentiles = (
        np.percentile(residual_values, 2),
        np.percentile(residual_values, 98),
    )
    max_abs_residual = max(abs(residual_percentiles[0]), abs(residual_percentiles[1]))
    residual_clim = (-max_abs_residual, max_abs_residual)

    # Histogram x-limits
    residual_xlim = (
        np.percentile(residual_values, 0.1),
        np.percentile(residual_values, 99.9),
    )
    max_abs_xlim = max(abs(residual_xlim[0]), abs(residual_xlim[1]))
    residual_xlim = (-max_abs_xlim, max_abs_xlim)

    # Log the calculated limits
    logger.info(f"Prediction range: [{pred_values.min():.3e}, {pred_values.max():.3e}]")
    logger.info(f"Truth range: [{truth_values.min():.3e}, {truth_values.max():.3e}]")
    logger.info(
        f"Prediction/Truth color limits: [{prediction_truth_clim[0]:.3e}, {prediction_truth_clim[1]:.3e}]"
    )
    logger.info(
        f"Residual color limits: [{residual_clim[0]:.3e}, {residual_clim[1]:.3e}]"
    )
    logger.info(
        f"Residual histogram x-limits: [{residual_xlim[0]:.3e}, {residual_xlim[1]:.3e}]"
    )

    return prediction_truth_clim, residual_clim, residual_xlim

def create_2d_chart_background(mesh):
    """Create a 2D chart with axes for background."""
    chart = pv.Chart2D()
    chart.x_range = [mesh.bounds[0], mesh.bounds[1]]
    chart.y_range = [mesh.bounds[2], mesh.bounds[3]]
    chart.x_label = "X position [m]"
    chart.y_label = "Y position [m]"
    chart.x_axis.tick_label_size = 15
    chart.y_axis.tick_label_size = 15
    chart.x_axis.label_size = 17
    chart.y_axis.label_size = 17
    chart.scatter([0], [0], color=pv.Color("green", opacity=0.0))
    chart.grid = False
    chart.size = (0.42, 0.93)
    chart.loc = (0.278, 0.0)
    for attr, val in [("opacity", 0.0), ("border_style", None)]:
        try:
            setattr(chart, attr, val)
        except Exception:
            pass
    return chart


def create_histogram_figure(residuals, title="Residual Distribution", xlim=None):
    """Create a matplotlib figure with residual histogram and statistics."""
    fig = plt.figure(figsize=(6, 4))

    # Create histogram with fixed x-axis limits
    if xlim is not None:
        bins = np.linspace(xlim[0], xlim[1], 26)
    else:
        bins = np.linspace(-5, 5, 26)

    plt.hist(
        residuals, bins=bins, density=True, alpha=0.7, color="blue", edgecolor="black"
    )

    # Calculate statistics
    mean = np.mean(residuals)
    std = np.std(residuals) + 1e-9
    min_val = np.min(residuals)
    max_val = np.max(residuals)

    # Add normal distribution overlay
    x = np.linspace(xlim[0] if xlim else min_val, xlim[1] if xlim else max_val, 100)
    normal_pdf = (1 / np.sqrt(2 * np.pi * std**2)) * np.exp(
        -0.5 * ((x - mean) / std) ** 2
    )
    plt.plot(
        x, normal_pdf, "r-", linewidth=2, label=f"Normal(μ={mean:.3e}, σ={std:.3e})"
    )

    plt.xlabel("Residual Value")
    plt.ylabel("Probability Density")
    plt.title(title)
    plt.legend()

    # Add statistics text
    stats_text = (
        f"Mean: {mean:.3e}\nStd: {std:.3e}\nMin: {min_val:.3e}\nMax: {max_val:.3e}"
    )
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    if xlim is not None:
        plt.xlim(xlim)

    plt.tight_layout()
    return fig


def create_residual_time_plot(times, mean_residuals, current_idx):
    """Create a plot of mean residual vs time."""
    fig = plt.figure(figsize=(6, 4))

    # Plot all data up to current time
    plt.plot(
        times[: current_idx + 1], mean_residuals[: current_idx + 1], "b-", linewidth=2
    )
    plt.plot(times[current_idx], mean_residuals[current_idx], "ro", markersize=8)

    plt.xlabel("Time")
    plt.ylabel("Mean Residual")
    plt.title("Mean Residual vs Time")
    plt.grid(True, alpha=0.3)

    # Set x-axis limits to full time range
    if len(times) > 1:
        plt.xlim(times[0], times[-1])

    plt.tight_layout()
    return fig


def create_info_text_figure(time, mesh_info, residual_stats):
    """Create a figure with information text."""
    fig = plt.figure(figsize=(6, 4), facecolor="white")
    ax = fig.add_subplot(111)
    ax.axis("off")

    # Create formatted text
    info_text = f"""Simulation Info
═══════════════════════════════════
Time: {time:.4f}

Mesh Statistics:
  • Points: {mesh_info['n_points']:,}
  • Cells: {mesh_info['n_cells']:,}
  • Type: {mesh_info['type']}

Current Residual Statistics:
  • Mean: {residual_stats['mean']:.6e}
  • Std Dev: {residual_stats['std']:.6e}
  • Min: {residual_stats['min']:.6e}
  • Max: {residual_stats['max']:.6e}
  • RMS: {residual_stats['rms']:.6e}
"""

    ax.text(
        0.05,
        0.95,
        info_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7),
    )

    plt.tight_layout()
    return fig

def create_movie(pvd_path, output_path, field_name, fps=24, quality=8, cmap="viridis"):
    """
    Create movie for a single field.
    
    Args:
        pvd_path: Input PVD file
        output_path: Output MP4 file
        field_name: Base field name (e.g. "u", "vx", "vy")
        fps: Frames per second
        quality: Video quality (0-10)
        cmap: Colormap
    
    Returns:
        str: Path to output file
    """
    create_movie_from_pvd(
        pvd_file=pvd_path,
        output_file=output_path,
        prediction_field=f"predicted_{field_name}",
        truth_field=f"ground_truth_{field_name}",
        residual_field=f"residual_{field_name}",
        fps=fps,
        quality=quality,
        cmap=cmap,
    )
    return str(output_path)

def create_movie_from_pvd(
    pvd_file,
    output_file,
    prediction_field="prediction",
    truth_field="truth",
    residual_field="residual",
    units="",
    fps=24,
    quality=8,
    cmap="viridis",
    num_workers=None,
):
    """Create an MP4 movie from a PVD file with multi-field visualization."""

    pvd_file = Path(pvd_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading PVD file: {pvd_file}")

    # Set up headless rendering
    pv.start_xvfb()

    reader = pv.get_reader(str(pvd_file))
    time_values = reader.time_values
    logger.info(f"Found {len(time_values)} time steps")

    # Read first timestep for mesh info
    reader.set_active_time_value(time_values[0])
    first_mesh = reader.read()
    if isinstance(first_mesh, pv.MultiBlock):
        first_mesh = first_mesh[0]

    logger.info(f"Mesh info: {first_mesh.n_points} points, {first_mesh.n_cells} cells")
    logger.info(f"Available arrays: {first_mesh.array_names}")

    # Check if data is 2D
    is_2d = check_if_2d(first_mesh)
    mesh_type = "2D" if is_2d else "3D"
    logger.info(f"Data type: {mesh_type}")

    # Calculate color limits from first and last timesteps
    prediction_truth_clim, residual_clim, residual_xlim = calculate_color_limits(
        reader, time_values, prediction_field, truth_field, residual_field
    )

    # Initialize arrays for time series data
    mean_residuals = []

    # Create plotter with 4x2 grid
    row_weights = [1, 1, 1, 0.5]
    col_weights = [1.5, 1]
    plotter = pv.Plotter(
        shape=(4, 2),
        row_weights=row_weights,
        col_weights=col_weights,
        window_size=(2672, 1503),
        off_screen=True,
    )
    plotter.enable_parallel_projection()
    plotter.enable_2d_style()

    # Open movie file
    logger.info(f"Creating movie: {output_file}")
    plotter.open_movie(str(output_file), framerate=fps, quality=quality)

    # Process each timestep
    for i, time_value in tqdm(
        enumerate(time_values), total=len(time_values), desc="Rendering"
    ):
        # Read mesh for this timestep
        reader.set_active_time_value(time_value)
        mesh = reader.read()
        if isinstance(mesh, pv.MultiBlock):
            mesh = mesh[0]

        # Create separate copies for each visualization to prevent any data mixing
        mesh_prediction = mesh.copy(deep=True)
        mesh_truth = mesh.copy(deep=True)
        mesh_residual = mesh.copy(deep=True)

        # Get the actual data arrays (for statistics)
        try:
            pred_data = get_field_data(mesh, prediction_field)
            truth_data = get_field_data(mesh, truth_field)

            # Get or compute residual
            try:
                residual_data = get_field_data(mesh, residual_field)
            except ValueError:
                residual_data = pred_data - truth_data
                # Add computed residual to the residual mesh
                if prediction_field in mesh.point_data:
                    mesh_residual.point_data[residual_field] = residual_data
                else:
                    mesh_residual.cell_data[residual_field] = residual_data
        except ValueError as e:
            logger.error(f"Error getting field data at t={time_value}: {e}")
            raise

        # Debug check
        if i == 0 or i == len(time_values) - 1:
            diff = np.abs(pred_data - truth_data).max()
            logger.info(f"Frame {i}: Max diff between pred and truth: {diff:.6e}")

        # Calculate statistics
        mean_residuals.append(np.sqrt(np.mean(residual_data**2)))
        residual_stats = {
            "mean": np.mean(residual_data),
            "std": np.std(residual_data),
            "min": np.min(residual_data),
            "max": np.max(residual_data),
            "rms": np.sqrt(np.mean(residual_data**2)),
        }

        mesh_info = {
            "n_points": mesh.n_points,
            "n_cells": mesh.n_cells,
            "type": mesh_type,
        }

        # Clear all subplots
        plotter.clear()

        # --- PREDICTION SUBPLOT (0, 0) ---
        plotter.subplot(0, 0)
        if is_2d:
            plotter.add_chart(create_2d_chart_background(mesh))
        plotter.add_text(
            f"Prediction (t={time_value:.3f})", position=(500, 350), font_size=8
        )
        plotter.add_mesh(
            mesh_prediction,
            scalars=prediction_field,
            cmap=cmap,
            clim=prediction_truth_clim,
            show_edges=False,
            edge_color=[0.5, 0.5, 0.5, 0.1],
            line_width=0.1,
            scalar_bar_args={
                "position_x": 0.1,
                "position_y": 0.1,
                "width": 0.05,
                "vertical": True,
            },
        )
        if is_2d:
            plotter.view_xy()
            plotter.camera.tight(padding=0.3)

        # --- TRUTH SUBPLOT (1, 0) ---
        plotter.subplot(1, 0)
        if is_2d:
            plotter.add_chart(create_2d_chart_background(mesh))
        plotter.add_text("Truth", position=(500, 350), font_size=8)
        plotter.add_mesh(
            mesh_truth,
            scalars=truth_field,
            cmap=cmap,
            clim=prediction_truth_clim,
            show_edges=False,
            edge_color=[0.5, 0.5, 0.5, 0.1],
            line_width=0.1,
            scalar_bar_args={
                "position_x": 0.1,
                "position_y": 0.1,
                "width": 0.05,
                "vertical": True,
            },
        )
        if is_2d:
            plotter.view_xy()
            plotter.camera.tight(padding=0.3)

        # --- RESIDUAL SUBPLOT (2, 0) ---
        plotter.subplot(2, 0)
        if is_2d:
            plotter.add_chart(create_2d_chart_background(mesh))
        plotter.add_text("Residual (Pred - Truth)", position=(500, 350), font_size=10)
        plotter.add_mesh(
            mesh_residual,
            scalars=residual_field,
            cmap="RdBu_r",
            clim=residual_clim,
            show_edges=False,
            edge_color=[0.5, 0.5, 0.5, 0.1],
            line_width=0.1,
            scalar_bar_args={
                "position_x": 0.1,
                "position_y": 0.1,
                "width": 0.05,
                "vertical": True,
            },
        )
        if is_2d:
            plotter.view_xy()
            plotter.camera.tight(padding=0.3)

        # --- MEAN RESIDUAL VS TIME PLOT (0, 1) ---
        plotter.subplot(0, 1)
        time_fig = create_residual_time_plot(time_values[: i + 1], mean_residuals, i)
        chart = pv.ChartMPL(time_fig, size=(0.9, 0.9), loc=(0.05, 0.05))
        chart.background_color = "white"
        plotter.add_chart(chart)
        plt.close(time_fig)

        # --- RESIDUAL HISTOGRAM (1, 1) ---
        plotter.subplot(1, 1)
        hist_fig = create_histogram_figure(
            residual_data, "Residual Distribution", xlim=residual_xlim
        )
        chart = pv.ChartMPL(hist_fig, size=(0.9, 0.9), loc=(0.05, 0.05))
        chart.background_color = "white"
        plotter.add_chart(chart)
        plt.close(hist_fig)

        # --- INFO TEXT (2, 1) ---
        plotter.subplot(2, 1)
        info_fig = create_info_text_figure(time_value, mesh_info, residual_stats)
        chart = pv.ChartMPL(info_fig, size=(0.9, 0.9), loc=(0.05, 0.05))
        chart.background_color = "white"
        plotter.add_chart(chart)
        plt.close(info_fig)

        # Link cameras for left column
        plotter.link_views([0, 2, 4])  # subplot indices for (0,0), (1,0), (2,0)

        # Write frame
        plotter.write_frame()

    # Close the movie file
    plotter.close()

    logger.success(f"Movie saved: {output_file}")
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.1f} MB")
    logger.info(f"Duration: {len(time_values) / fps:.1f} seconds at {fps} FPS")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Convert PVD files to MP4 movies showing prediction, truth, and residual"
    )

    parser.add_argument("input", help="Input PVD file")
    parser.add_argument("output", help="Output MP4 file")
    parser.add_argument(
        "--prediction", default="prediction", help="Prediction field name"
    )
    parser.add_argument("--truth", default="truth", help="Truth field name")
    parser.add_argument("--residual", default="residual", help="Residual field name")
    parser.add_argument("--units", default="", help="Units to use")
    parser.add_argument("--cmap", default="viridis", help="Colormap")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--quality", type=int, default=8, help="Video quality (0-10)")

    args = parser.parse_args()

    create_movie_from_pvd(
        pvd_file=args.input,
        output_file=args.output,
        prediction_field=args.prediction,
        truth_field=args.truth,
        residual_field=args.residual,
        units=f"[{args.units}]" if len(args.units) > 0 else "",
        fps=args.fps,
        quality=args.quality,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
