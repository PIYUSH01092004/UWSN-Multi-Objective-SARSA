"""
simulator.py
------------
Discrete-round simulation engine that evaluates three UWSN routing protocols
(MultiObjectiveSARSA, EDORQ, DBR) across node densities [200, 300, 400, 500, 600, 700, 800].

Metrics computed per (protocol, density):
  - PDR  : Packet Delivery Ratio (%)
  - TEC  : Total Energy Consumption (kJ)
  - EBI  : Energy Balance Index (Jain's Fairness Index)
  - APD  : Average End-to-End Delay (seconds)
  - ANO  : Average Network Overhead (control packets / delivered packet)

Results are saved as:
  - results/simulation_results.csv
  - results/plots/  (one PNG per metric)
"""

import os
import logging
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")       # non-interactive backend for server/batch use
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from uwsn_env import UWSNEnvironment, make_environment, DENSITIES, E_INITIAL
from routing_protocols import (
    MultiObjectiveSARSA, EDORQ, DBR, PacketRouteResult
)
from audio_processor import AudioPayloadManager, find_urbansound_folder

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -------------------------------------------------
# Simulation configuration
# -------------------------------------------------
N_PACKETS: int = 200       # Packets to route per (protocol, density) run
N_DRIFT_STEPS: int = 5     # Drift steps between packet injection rounds
DRIFT_DT: float = 2.0      # Seconds per drift step
SEED_BASE: int = 42        # Base RNG seed
RESULTS_DIR: str = "results"
PLOTS_DIR: str = os.path.join(RESULTS_DIR, "plots")


# -------------------------------------------------
# Metrics dataclass
# -------------------------------------------------
@dataclass
class SimMetrics:
    """Aggregated simulation metrics for one (protocol, density) configuration."""
    protocol: str = ""
    density: int = 0
    pdr: float = 0.0      # Packet Delivery Ratio (%)
    tec: float = 0.0      # Total Energy Consumption (kJ)
    ebi: float = 0.0      # Energy Balance Index (Jain's Fairness Index)
    apd: float = 0.0      # Average End-to-End Delay (s)
    ano: float = 0.0      # Average Network Overhead
    packets_sent: int = 0
    packets_delivered: int = 0
    runtime_s: float = 0.0


# -------------------------------------------------
# Jain's Fairness Index
# -------------------------------------------------
def jains_fairness_index(values: np.ndarray) -> float:
    """
    Compute Jain's Fairness Index for a set of values.
    JFI = (sum(x_i))^2 / (n * sum(x_i^2))
    Range: [1/n, 1]; 1 = perfectly fair.
    """
    if len(values) == 0:
        return 0.0
    n = len(values)
    s = np.sum(values)
    sq_s = np.sum(values ** 2)
    if sq_s == 0:
        return 1.0
    return float((s ** 2) / (n * sq_s))


# -------------------------------------------------
# Single run: one protocol on one density
# -------------------------------------------------
def run_single(protocol_name: str,
               density: int,
               payload_mgr: AudioPayloadManager,
               seed: int = SEED_BASE) -> SimMetrics:
    """
    Run N_PACKETS routing attempts for the given protocol/density combination.

    Args:
        protocol_name: One of 'SARSA', 'EDORQ', 'DBR'.
        density:       Number of sensor nodes.
        payload_mgr:   Pre-loaded audio feature manager.
        seed:          RNG seed.

    Returns:
        SimMetrics aggregated over all packet routing attempts.
    """
    t_start = time.perf_counter()

    # Build environment
    env = make_environment(density, seed=seed)

    # Instantiate protocol
    payload_vec, payload_bytes = payload_mgr.get_payload()
    if protocol_name == "SARSA":
        protocol = MultiObjectiveSARSA(env, payload_bytes=payload_bytes)
    elif protocol_name == "EDORQ":
        protocol = EDORQ(env, payload_bytes=payload_bytes)
    elif protocol_name == "DBR":
        protocol = DBR(env, payload_bytes=payload_bytes)
    else:
        raise ValueError(f"Unknown protocol: {protocol_name}")

    delivered = 0
    total_delay = 0.0
    total_energy = 0.0
    total_overhead = 0.0
    energy_snapshots: List[float] = []

    for pkt_idx in range(N_PACKETS):
        # Drift network between packet injections (every 10 packets)
        if pkt_idx > 0 and pkt_idx % 10 == 0:
            for _ in range(N_DRIFT_STEPS):
                env.step_drift(dt=DRIFT_DT)

        # Get audio payload for this packet
        _, pb = payload_mgr.get_payload()
        protocol.payload_bytes = pb
        protocol.packet_bits = pb * 8.0

        result: PacketRouteResult = protocol.route_packet()

        if result.delivered:
            delivered += 1
            total_delay += result.delay_s

        total_energy += result.energy_consumed_j
        total_overhead += result.overhead_packets

        # Snapshot per-node residual energy every 50 packets
        if pkt_idx % 50 == 0:
            energies = np.array([n.energy for n in env.nodes if n.alive])
            energy_snapshots.append(float(np.mean(energies)) if len(energies) > 0 else 0.0)

    # Compute metrics
    pdr = (delivered / N_PACKETS) * 100.0
    tec_kj = total_energy / 1000.0
    apd = (total_delay / delivered) if delivered > 0 else 0.0
    ano = (total_overhead / delivered) if delivered > 0 else float(total_overhead)

    # EBI: Jain's fairness over final residual energies
    final_energies = np.array([n.energy for n in env.nodes])
    ebi = jains_fairness_index(final_energies)

    t_end = time.perf_counter()
    runtime = t_end - t_start

    logger.info("[%s | N=%d] PDR=%.1f%% TEC=%.4f kJ EBI=%.4f APD=%.6f s ANO=%.2f  [%.1fs]",
                protocol_name, density, pdr, tec_kj, ebi, apd, ano, runtime)

    return SimMetrics(
        protocol=protocol_name,
        density=density,
        pdr=pdr,
        tec=tec_kj,
        ebi=ebi,
        apd=apd,
        ano=ano,
        packets_sent=N_PACKETS,
        packets_delivered=delivered,
        runtime_s=runtime
    )


# -------------------------------------------------
# Full comparative simulation
# -------------------------------------------------
def run_full_simulation(
        densities: Optional[List[int]] = None,
        protocols: Optional[List[str]] = None,
        payload_mgr: Optional[AudioPayloadManager] = None
) -> pd.DataFrame:
    """
    Run all protocols across all densities and return results as a DataFrame.

    Args:
        densities:   Node densities to evaluate (default: DENSITIES).
        protocols:   Protocol names to evaluate (default: all three).
        payload_mgr: Pre-loaded AudioPayloadManager (auto-created if None).

    Returns:
        DataFrame with columns: protocol, density, pdr, tec, ebi, apd, ano
    """
    if densities is None:
        densities = DENSITIES
    if protocols is None:
        protocols = ["SARSA", "EDORQ", "DBR"]
    if payload_mgr is None:
        ds_folder = find_urbansound_folder()
        payload_mgr = AudioPayloadManager(ds_folder, max_files=30)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    all_metrics: List[SimMetrics] = []
    total_runs = len(protocols) * len(densities)
    run_count = 0

    logger.info("=" * 70)
    logger.info("UWSN Routing Simulation: %d protocols x %d densities = %d runs",
                len(protocols), len(densities), total_runs)
    logger.info("Packets per run: %d | Drift steps: %d | Dt: %.1fs",
                N_PACKETS, N_DRIFT_STEPS, DRIFT_DT)
    logger.info("=" * 70)

    for density in densities:
        for protocol_name in protocols:
            run_count += 1
            logger.info("Run %d/%d: protocol=%s density=%d",
                        run_count, total_runs, protocol_name, density)
            seed = SEED_BASE + density + hash(protocol_name) % 1000
            metrics = run_single(protocol_name, density, payload_mgr, seed=seed)
            all_metrics.append(metrics)

    # Convert to DataFrame
    records = []
    for m in all_metrics:
        records.append({
            "protocol": m.protocol,
            "density": m.density,
            "PDR (%)": round(m.pdr, 2),
            "TEC (kJ)": round(m.tec, 6),
            "EBI": round(m.ebi, 6),
            "APD (s)": round(m.apd, 6),
            "ANO": round(m.ano, 4),
            "packets_sent": m.packets_sent,
            "packets_delivered": m.packets_delivered,
            "runtime_s": round(m.runtime_s, 2)
        })
    df = pd.DataFrame(records)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "simulation_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Results saved to: %s", csv_path)

    return df


# -------------------------------------------------
# Plotting
# -------------------------------------------------
# Colour palette: distinct, research-paper quality
PROTOCOL_COLORS = {
    "SARSA": "#1f77b4",   # Steel blue
    "EDORQ": "#ff7f0e",   # Orange
    "DBR":   "#2ca02c",   # Green
}
PROTOCOL_MARKERS = {
    "SARSA": "o",
    "EDORQ": "s",
    "DBR":   "^",
}
PROTOCOL_LABELS = {
    "SARSA": "Proposed MO-SARSA",
    "EDORQ": "EDORQ (Q-Learning)",
    "DBR":   "DBR (Baseline)",
}


def _style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent styling to a matplotlib axis."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9, framealpha=0.85)


def generate_plots(df: pd.DataFrame) -> List[str]:
    """
    Generate comparison plots for all five metrics.

    Returns:
        List of saved PNG file paths.
    """
    metrics_config = [
        ("PDR (%)",  "Packet Delivery Ratio (%)",     "PDR (%)",   True),
        ("TEC (kJ)", "Total Energy Consumption (kJ)", "TEC (kJ)",  False),
        ("EBI",      "Energy Balance Index (Jain's)", "EBI",       True),
        ("APD (s)",  "Avg. End-to-End Delay (s)",     "APD (s)",   False),
        ("ANO",      "Avg. Network Overhead",          "ANO",       False),
    ]

    saved_paths = []
    protocols = df["protocol"].unique()
    densities = sorted(df["density"].unique())

    # ── Individual metric plots ──────────────────────────────────────────────
    for col, title, ylabel, higher_is_better in metrics_config:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        for proto in ["SARSA", "EDORQ", "DBR"]:
            if proto not in protocols:
                continue
            sub = df[df["protocol"] == proto].sort_values("density")
            ax.plot(
                sub["density"], sub[col],
                color=PROTOCOL_COLORS[proto],
                marker=PROTOCOL_MARKERS[proto],
                label=PROTOCOL_LABELS[proto],
                linewidth=2.0,
                markersize=7,
                linestyle="-"
            )
        direction = "↑ better" if higher_is_better else "↓ better"
        _style_axis(ax,
                    title=f"{title}  ({direction})",
                    xlabel="Number of Sensor Nodes",
                    ylabel=ylabel)
        ax.set_xticks(densities)
        fig.tight_layout()
        fname = col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("%", "pct")
        path = os.path.join(PLOTS_DIR, f"{fname}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)
        logger.info("Plot saved: %s", path)

    # ── Combined 5-panel summary figure ─────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=100)
    axes_flat = axes.flatten()

    for idx, (col, title, ylabel, higher_is_better) in enumerate(metrics_config):
        ax = axes_flat[idx]
        for proto in ["SARSA", "EDORQ", "DBR"]:
            if proto not in protocols:
                continue
            sub = df[df["protocol"] == proto].sort_values("density")
            ax.plot(
                sub["density"], sub[col],
                color=PROTOCOL_COLORS[proto],
                marker=PROTOCOL_MARKERS[proto],
                label=PROTOCOL_LABELS[proto],
                linewidth=2.0,
                markersize=6,
            )
        direction = "↑" if higher_is_better else "↓"
        _style_axis(ax,
                    title=f"{title}  {direction}",
                    xlabel="Nodes",
                    ylabel=ylabel)
        ax.set_xticks(densities)
        ax.tick_params(axis="x", rotation=30)

    # Hide unused subplot
    axes_flat[-1].axis("off")

    fig.suptitle(
        "UWSN Routing Protocol Comparison\n"
        "Multi-Objective SARSA vs EDORQ vs DBR",
        fontsize=15, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    summary_path = os.path.join(PLOTS_DIR, "summary_comparison.png")
    fig.savefig(summary_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(summary_path)
    logger.info("Summary plot saved: %s", summary_path)

    return saved_paths


# -------------------------------------------------
# Convenience runner
# -------------------------------------------------
def simulate_and_plot(
        densities: Optional[List[int]] = None,
        payload_mgr: Optional[AudioPayloadManager] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Full pipeline: run simulation, save CSV, generate plots.

    Returns:
        (results_dataframe, list_of_saved_plot_paths)
    """
    df = run_full_simulation(densities=densities, payload_mgr=payload_mgr)
    plot_paths = generate_plots(df)
    return df, plot_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df, plots = simulate_and_plot()
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nPlots saved in: {PLOTS_DIR}")
