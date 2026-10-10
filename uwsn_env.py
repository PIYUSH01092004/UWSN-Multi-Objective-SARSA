"""
uwsn_env.py
-----------
Underwater Wireless Sensor Network (UWSN) environment model.

Models a 3D 500m x 500m x 500m underwater space with:
- 4 surface sinks at z = 0
- Sensor nodes with configurable density (200-800 nodes)
- Random walk drift model (1-3 m/s horizontal)
- Acoustic channel parameters matching EDORQ/SARSA research context
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# -------------------------------------------------
# Acoustic channel constants
# -------------------------------------------------
SPACE_X: float = 500.0         # m
SPACE_Y: float = 500.0         # m
SPACE_Z: float = 500.0         # m
R_MAX: float = 100.0           # Transmission range (m)
V_SOUND: float = 1500.0        # Acoustic propagation speed (m/s)
BITRATE: float = 10_000.0      # bits/s (10 kbps)
P_TX: float = 2.0              # Transmission power (W)
P_RX: float = 0.1              # Reception power (W)
P_IDLE: float = 0.01           # Idle listening power (W)
T_MAX: float = (2.0 * R_MAX) / V_SOUND   # = 0.1333 s
N_SINKS: int = 4               # Surface sinks
E_INITIAL: float = 10.0        # Initial node energy (J)
DRIFT_MIN: float = 1.0         # Min drift speed (m/s)
DRIFT_MAX: float = 3.0         # Max drift speed (m/s)

# Node densities to evaluate
DENSITIES: List[int] = [200, 300, 400, 500, 600, 700, 800]


# -------------------------------------------------
# Data classes
# -------------------------------------------------
@dataclass
class SinkNode:
    """Surface sink node (fixed at z=0)."""
    node_id: int
    x: float
    y: float
    z: float = 0.0

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass
class SensorNode:
    """
    Underwater sensor node with energy tracking, drift, and Q-table.
    """
    node_id: int
    x: float
    y: float
    z: float
    energy: float = field(default=E_INITIAL)
    drift_vx: float = field(default=0.0)  # Drift velocity x (m/s)
    drift_vy: float = field(default=0.0)  # Drift velocity y (m/s)
    alive: bool = field(default=True)

    # Q-table: keys are (state_hash, neighbor_id) -> Q-value
    q_table: Dict[Tuple, float] = field(default_factory=dict)

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def distance_to(self, other_pos: np.ndarray) -> float:
        return float(np.linalg.norm(self.position() - other_pos))

    def depth(self) -> float:
        """Depth = z coordinate (0 = surface, 500 = bottom)."""
        return self.z

    def is_in_range(self, other_pos: np.ndarray, r_max: float = R_MAX) -> bool:
        return self.distance_to(other_pos) <= r_max

    def consume_energy(self, power_w: float, duration_s: float) -> None:
        """Deduct energy for given power draw and duration."""
        consumed = power_w * duration_s
        self.energy = max(0.0, self.energy - consumed)
        if self.energy <= 0.0:
            self.alive = False

    def get_q_value(self, state_key: tuple, neighbor_id: int) -> float:
        return self.q_table.get((state_key, neighbor_id), 0.0)

    def set_q_value(self, state_key: tuple, neighbor_id: int, value: float) -> None:
        self.q_table[(state_key, neighbor_id)] = float(value)


# -------------------------------------------------
# UWSN Environment
# -------------------------------------------------
class UWSNEnvironment:
    """
    3D Underwater Wireless Sensor Network environment.

    Manages:
    - Node placement and sink placement
    - Node drift simulation per time step
    - Neighbor discovery within R_MAX
    - Energy accounting helpers
    """

    def __init__(self, n_nodes: int, seed: int = 42):
        """
        Initialise the UWSN environment.

        Args:
            n_nodes: Number of sensor nodes to deploy.
            seed:    Random seed for reproducibility.
        """
        self.n_nodes = n_nodes
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.time: float = 0.0          # Simulation time (s)
        self.round: int = 0             # Round counter

        # Deploy surface sinks
        self.sinks: List[SinkNode] = self._deploy_sinks()

        # Deploy sensor nodes
        self.nodes: List[SensorNode] = self._deploy_nodes()

        logger.debug("UWSNEnvironment: %d nodes, %d sinks, seed=%d", n_nodes, N_SINKS, seed)

    # ── Deployment ──────────────────────────────────────────────────────────
    def _deploy_sinks(self) -> List[SinkNode]:
        """Place 4 surface sinks uniformly at z=0."""
        positions = [
            (SPACE_X * 0.25, SPACE_Y * 0.25),
            (SPACE_X * 0.75, SPACE_Y * 0.25),
            (SPACE_X * 0.25, SPACE_Y * 0.75),
            (SPACE_X * 0.75, SPACE_Y * 0.75),
        ]
        return [
            SinkNode(node_id=i, x=px, y=py, z=0.0)
            for i, (px, py) in enumerate(positions)
        ]

    def _deploy_nodes(self) -> List[SensorNode]:
        """Deploy sensor nodes randomly in 3D space (z > 0 to avoid surface layer)."""
        xs = self.rng.uniform(0.0, SPACE_X, self.n_nodes)
        ys = self.rng.uniform(0.0, SPACE_Y, self.n_nodes)
        # Nodes deploy between 10m and 490m depth
        zs = self.rng.uniform(10.0, SPACE_Z - 10.0, self.n_nodes)

        # Random drift velocities
        angles = self.rng.uniform(0.0, 2 * np.pi, self.n_nodes)
        speeds = self.rng.uniform(DRIFT_MIN, DRIFT_MAX, self.n_nodes)
        vx = speeds * np.cos(angles)
        vy = speeds * np.sin(angles)

        nodes = []
        for i in range(self.n_nodes):
            node = SensorNode(
                node_id=i,
                x=float(xs[i]),
                y=float(ys[i]),
                z=float(zs[i]),
                energy=E_INITIAL,
                drift_vx=float(vx[i]),
                drift_vy=float(vy[i]),
            )
            nodes.append(node)
        return nodes

    # ── Drift ───────────────────────────────────────────────────────────────
    def step_drift(self, dt: float = 1.0) -> None:
        """
        Advance node positions by one time step dt (seconds) using
        horizontal random-walk drift model. Positions wrap within [0, SPACE_X/Y].
        """
        for node in self.nodes:
            if not node.alive:
                continue

            # Small random perturbation to drift direction (Brownian component)
            d_angle = self.rng.uniform(-0.1, 0.1)
            speed = np.sqrt(node.drift_vx ** 2 + node.drift_vy ** 2)
            angle = np.arctan2(node.drift_vy, node.drift_vx) + d_angle

            # Clamp speed within [DRIFT_MIN, DRIFT_MAX]
            speed = np.clip(speed + self.rng.uniform(-0.1, 0.1), DRIFT_MIN, DRIFT_MAX)
            node.drift_vx = speed * np.cos(angle)
            node.drift_vy = speed * np.sin(angle)

            # Update position with wraparound
            node.x = (node.x + node.drift_vx * dt) % SPACE_X
            node.y = (node.y + node.drift_vy * dt) % SPACE_Y
            # z remains constant (vertical buoyancy assumed stable)

        self.time += dt
        self.round += 1

    # ── Neighbor discovery ──────────────────────────────────────────────────
    def get_neighbors(self, node: SensorNode, r_max: float = R_MAX) -> List[SensorNode]:
        """
        Return list of alive sensor nodes within R_MAX of the given node
        (excluding itself).
        """
        neighbors = []
        for other in self.nodes:
            if other.node_id == node.node_id or not other.alive:
                continue
            if node.is_in_range(other.position(), r_max):
                neighbors.append(other)
        return neighbors

    def get_sink_neighbors(self, node: SensorNode, r_max: float = R_MAX) -> List[SinkNode]:
        """Return sinks within R_MAX of the given node."""
        reachable = []
        for sink in self.sinks:
            if node.is_in_range(sink.position(), r_max):
                reachable.append(sink)
        return reachable

    # ── Energy helpers ──────────────────────────────────────────────────────
    def transmission_duration(self, payload_bytes: int) -> float:
        """Time to transmit payload_bytes at BITRATE bps."""
        bits = payload_bytes * 8.0
        return bits / BITRATE

    def propagation_delay(self, distance: float) -> float:
        """One-way acoustic propagation delay for a given distance."""
        return distance / V_SOUND

    def alive_count(self) -> int:
        """Number of currently alive sensor nodes."""
        return sum(1 for n in self.nodes if n.alive)

    def total_residual_energy(self) -> float:
        """Sum of residual energy across all alive nodes."""
        return sum(n.energy for n in self.nodes if n.alive)

    def mean_residual_energy(self) -> float:
        """Mean residual energy across all alive nodes."""
        alive = [n.energy for n in self.nodes if n.alive]
        return float(np.mean(alive)) if alive else 0.0

    def neighbor_energy_variance(self, node: SensorNode) -> float:
        """Variance of residual energy among neighbors of the given node."""
        neighbors = self.get_neighbors(node)
        if not neighbors:
            return 0.0
        energies = np.array([nb.energy for nb in neighbors])
        return float(np.var(energies))

    def reset(self) -> None:
        """Reset all nodes to initial state (preserves topology)."""
        self.time = 0.0
        self.round = 0
        for node in self.nodes:
            node.energy = E_INITIAL
            node.alive = True
            node.q_table.clear()

    # ── State encoding ──────────────────────────────────────────────────────
    def get_state(self, node: SensorNode) -> tuple:
        """
        Encode the current observable state for a node as a discrete tuple.
        State = (depth_bin, energy_bin) for compact Q-table indexing.
        """
        depth_bin = int(node.z // 50)                         # 10 depth bins
        energy_bin = int(node.energy // (E_INITIAL / 10.0))   # 10 energy bins
        depth_bin = min(depth_bin, 9)
        energy_bin = min(energy_bin, 10)
        return (depth_bin, energy_bin)

    # ── Holding time formula ─────────────────────────────────────────────────
    @staticmethod
    def holding_time(q_value: float, t_max: float = T_MAX) -> float:
        """
        Compute the holding/back-off time for opportunistic forwarding.
        H_j = [1 - (2/pi) * arctan(Q)] * T_max

        Nodes with higher Q-values wait less time before forwarding.
        """
        return (1.0 - (2.0 / np.pi) * np.arctan(q_value)) * t_max


# -------------------------------------------------
# Factory: build environment for a given density
# -------------------------------------------------
def make_environment(n_nodes: int, seed: int = 42) -> UWSNEnvironment:
    """Create a fresh UWSNEnvironment for the given node count."""
    return UWSNEnvironment(n_nodes=n_nodes, seed=seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    env = make_environment(300, seed=0)
    print(f"Nodes alive    : {env.alive_count()}")
    print(f"Sinks deployed : {len(env.sinks)}")
    n0 = env.nodes[0]
    neighbors = env.get_neighbors(n0)
    print(f"Node 0 neighbors (within {R_MAX}m): {len(neighbors)}")
    print(f"Holding time (Q=0): {env.holding_time(0.0):.6f} s")
    print(f"Holding time (Q=5): {env.holding_time(5.0):.6f} s")
    env.step_drift(dt=10.0)
    print(f"After 10s drift - Node 0 pos: ({n0.x:.2f}, {n0.y:.2f}, {n0.z:.2f})")
