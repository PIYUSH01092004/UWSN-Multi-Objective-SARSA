"""
main.py
-------
Unified entry point for the UWSN Multi-Objective SARSA Routing Simulation.

Execution steps:
  1. Locate UrbanSound8K dataset and validate 2400 Hz MFCC pipeline.
  2. Print sample feature vectors from real audio files.
  3. Run comparative simulation: MO-SARSA vs EDORQ vs DBR.
  4. Display results table and save plots.

Usage:
    python main.py
    python main.py --quick     (fewer packets, faster run for testing)
    python main.py --densities 300 500 700
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# -------------------------------------------------
# Configure root logger before any imports
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("simulation.log", mode="w"),
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Project module imports
# -------------------------------------------------
from audio_processor import (
    AudioPayloadManager,
    find_urbansound_folder,
    collect_wav_files,
    process_audio,
    TARGET_SR,
    N_MFCC,
)
from uwsn_env import DENSITIES, make_environment, R_MAX, V_SOUND, T_MAX, E_INITIAL
import simulator as sim_module


# =============================================================================
# Step 1: Audio pipeline validation
# =============================================================================
def validate_audio_pipeline(n_samples: int = 5) -> AudioPayloadManager:
    """
    Locate UrbanSound8K dataset, process sample files at 2400 Hz,
    print MFCC feature vectors, and return initialised AudioPayloadManager.
    """
    print("\n" + "=" * 72)
    print("  STEP 1: Audio Pipeline Validation (UrbanSound8K @ 2400 Hz + MFCC)")
    print("=" * 72)

    # Auto-detect dataset
    base_dir = str(Path(__file__).parent)
    ds_folder = find_urbansound_folder(base_dir)

    if ds_folder:
        print(f"\n  Dataset found   : {ds_folder}")
        wav_files = collect_wav_files(ds_folder, max_per_fold=2)
        print(f"  Total WAV files : {len(wav_files)} (sampling first {n_samples})")
    else:
        print("\n  WARNING: UrbanSound8K dataset not found.")
        print("  Falling back to synthetic 2400 Hz acoustic waveforms.")
        wav_files = []

    sample_files = wav_files[:n_samples] if wav_files else [None] * n_samples

    print(f"\n  {'File':<45} {'MFCC[0..4]':<52} {'Bytes'}")
    print("  " + "-" * 105)

    for fp in sample_files:
        features, payload_bytes = process_audio(fp, TARGET_SR, N_MFCC)
        fname = os.path.basename(fp) if fp else "synthetic_2400Hz.wav"
        mfcc_str = "  ".join([f"{v:7.3f}" for v in features[:5]])
        print(f"  {fname:<45} [{mfcc_str}]  {payload_bytes}")

    print(f"\n  Sample rate : {TARGET_SR} Hz  |  MFCC features: {N_MFCC}  |  Payload: {N_MFCC * 4} bytes")

    # Initialise AudioPayloadManager (pre-processes up to 50 files)
    print("\n  Initialising AudioPayloadManager...")
    payload_mgr = AudioPayloadManager(ds_folder, max_files=50)
    sample_feat, sample_pb = payload_mgr.get_payload()
    print(f"  Pool size    : {len(payload_mgr)} feature vectors")
    print(f"  Sample feat  : {np.round(sample_feat[:7], 4)} ...")
    print(f"  Payload size : {sample_pb} bytes")
    print("\n  Audio pipeline VALIDATED.\n")

    return payload_mgr


# =============================================================================
# Step 2: Environment sanity check
# =============================================================================
def print_environment_summary() -> None:
    """Print acoustic channel parameters and a quick environment snapshot."""
    print("=" * 72)
    print("  STEP 2: UWSN Environment Parameters")
    print("=" * 72)

    env = make_environment(300, seed=0)
    n0 = env.nodes[0]
    neighbors = env.get_neighbors(n0)
    sink_nbs = env.get_sink_neighbors(n0)

    params = [
        ("Space dimensions",       "500m x 500m x 500m"),
        ("Surface sinks",          str(len(env.sinks))),
        ("Sensor nodes (example)", "300"),
        ("Drift range",            "1 - 3 m/s (horizontal)"),
        ("Transmission range",     f"{R_MAX} m"),
        ("Sound speed",            f"{V_SOUND} m/s"),
        ("Bitrate",                "10 kbps"),
        ("P_tx",                   "2.0 W"),
        ("P_rx",                   "0.1 W"),
        ("P_idle",                 "0.01 W"),
        ("T_max",                  f"{T_MAX:.6f} s"),
        ("Initial energy",         f"{E_INITIAL} J"),
        ("Node 0 position",        f"({n0.x:.1f}, {n0.y:.1f}, {n0.z:.1f}) m"),
        ("Node 0 neighbors",       str(len(neighbors))),
        ("Node 0 sink neighbors",  str(len(sink_nbs))),
    ]
    for k, v in params:
        print(f"  {k:<30} : {v}")

    ht_q0 = env.holding_time(0.0)
    ht_q5 = env.holding_time(5.0)
    print(f"\n  Holding time (Q=0) : {ht_q0:.6f} s")
    print(f"  Holding time (Q=5) : {ht_q5:.6f} s")
    print()


# =============================================================================
# Step 3: Run simulation
# =============================================================================
# =============================================================================
def run_simulation(densities: List[int],
                   payload_mgr: AudioPayloadManager,
                   n_mc_runs: int = sim_module.N_MC_RUNS,
                   n_packets: int = sim_module.N_PACKETS) -> pd.DataFrame:
    """Run the comparative simulation and return the results DataFrame."""
    print("=" * 72)
    print("  STEP 3: Comparative Routing Simulation (Monte Carlo Smoothed)")
    print("=" * 72)
    print(f"  Protocols    : Multi-Objective SARSA  |  EDORQ  |  DBR")
    print(f"  Densities    : {densities}")
    print(f"  MC Runs/point: {n_mc_runs}")
    print(f"  Packets/run  : {n_packets}  (Total {n_packets * n_mc_runs} pkts/point)")
    print(f"  Drift steps  : {sim_module.N_DRIFT_STEPS}  (dt={sim_module.DRIFT_DT}s)")
    print()

    t_start = time.perf_counter()
    df, plot_paths = sim_module.simulate_and_plot(
        densities=densities,
        payload_mgr=payload_mgr,
        n_mc_runs=n_mc_runs,
        n_packets=n_packets
    )
    elapsed = time.perf_counter() - t_start

    print(f"\n  Simulation completed in {elapsed:.1f} seconds.")
    return df, plot_paths


# =============================================================================
# Step 4: Display results
# =============================================================================
def display_results(df: pd.DataFrame, plot_paths: List[str]) -> None:
    """Pretty-print final results table and plot paths."""
    print("\n" + "=" * 72)
    print("  STEP 4: Final Results Summary")
    print("=" * 72)

    # Pivot for easier reading
    metrics = ["PDR (%)", "TEC (kJ)", "EBI", "APD (s)", "ANO"]
    for metric in metrics:
        print(f"\n  --- {metric} ---")
        pivot = df.pivot(index="density", columns="protocol", values=metric)
        # Reorder columns
        cols_ordered = [c for c in ["SARSA", "EDORQ", "DBR"] if c in pivot.columns]
        pivot = pivot[cols_ordered]
        pivot.columns = [f"MO-SARSA" if c == "SARSA" else c for c in pivot.columns]
        print(pivot.to_string())

    print("\n" + "=" * 72)
    print("  Saved Plots:")
    for p in plot_paths:
        print(f"    {p}")

    csv_path = os.path.join(sim_module.RESULTS_DIR, "simulation_results.csv")
    print(f"\n  Full CSV : {csv_path}")
    print("=" * 72)


# =============================================================================
# Argument parsing
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="UWSN Multi-Objective SARSA Routing Simulation"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast test mode with 3 MC runs and 25 packets."
    )
    parser.add_argument(
        "--mc-runs",
        type=int,
        default=sim_module.N_MC_RUNS,
        help="Number of Monte Carlo runs per density point (default: 15)."
    )
    parser.add_argument(
        "--densities",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Space-separated list of node densities to simulate (e.g. 300 500 700)."
    )
    parser.add_argument(
        "--packets",
        type=int,
        default=None,
        metavar="N",
        help="Override the number of packets per run."
    )
    return parser.parse_args()


# =============================================================================
# Entry point
# =============================================================================
def main() -> None:
    args = parse_args()

    # Apply CLI overrides
    n_mc_runs = 3 if args.quick else args.mc_runs
    n_packets = 25 if args.quick else (args.packets if args.packets else sim_module.N_PACKETS)

    densities = args.densities if args.densities else DENSITIES

    print("\n" + "#" * 72)
    print("#   UWSN MULTI-OBJECTIVE SARSA ROUTING SIMULATION")
    print("#   Research Paper: Energy-Balanced Opportunistic Routing via")
    print("#   Multi-Objective SARSA in Drifting Underwater WSNs")
    print("#" * 72)

    # ── Step 1: Audio ─────────────────────────────────────────────────────────
    payload_mgr = validate_audio_pipeline(n_samples=5)

    # ── Step 2: Environment ───────────────────────────────────────────────────
    print_environment_summary()

    # ── Step 3: Simulation ────────────────────────────────────────────────────
    df, plot_paths = run_simulation(densities, payload_mgr, n_mc_runs=n_mc_runs, n_packets=n_packets)

    # ── Step 4: Results ───────────────────────────────────────────────────────
    display_results(df, plot_paths)

    print("\n  ALL DONE. Simulation log saved to: simulation.log\n")


if __name__ == "__main__":
    main()
