from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import DiagonalGate

import gc

import numpy as np
import networkx as nx
from scipy.optimize import minimize


def cut_value(bitstring: str, graph: nx.Graph) -> float:
    """Return half the number of edges crossing the cut."""
    z = [int(b) for b in bitstring]
    return sum(z[u] != z[v] for u, v in graph.edges()) / 2


def compute_maxcut_costs(graph: nx.Graph) -> np.ndarray:
    """Pre-compute the Max-Cut cost for every basis state."""
    n = graph.number_of_nodes()
    num_states = 1 << n            # 2 ** n
    costs = np.empty(num_states, dtype=float)
    for k in range(num_states):
        bstr = format(k, f"0{n}b")
        costs[k] = cut_value(bstr, graph)
    return costs


def apply_initial_state(qc: QuantumCircuit) -> None:
    """Apply Hadamards to all qubits."""
    for q in range(qc.num_qubits):
        qc.h(q)


def apply_cost_unitary(qc: QuantumCircuit,
                       qubits,
                       costs: np.ndarray,
                       gamma: float) -> None:
    """Apply the diagonal cost unitary."""
    diag = np.exp(-1j * gamma * costs)
    qc.append(DiagonalGate(diag), qubits)
    gc.collect()


def apply_mixer_unitary(qc: QuantumCircuit,
                        beta: float) -> None:
    """Apply RX rotations on all qubits."""
    for q in range(qc.num_qubits):
        qc.rx(2 * beta, q)


def build_qaoa_circuit(params: np.ndarray,
                       layers: int,
                       graph: nx.Graph,
                       costs: np.ndarray) -> QuantumCircuit:
    """Return a full QAOA circuit for the given parameters."""
    qc = QuantumCircuit(graph.number_of_nodes())
    gammas, betas = np.split(params, 2)
    qubits = range(qc.num_qubits)

    apply_initial_state(qc)

    for l in range(layers):
        apply_cost_unitary(qc, qubits, costs, gammas[l])
        apply_mixer_unitary(qc, betas[l])

    qc.save_probabilities()

    return qc


def qaoa_expectation(params: np.ndarray,
                     backend: AerSimulator,
                     layers: int,
                     graph: nx.Graph,
                     costs: np.ndarray) -> float:
    """Return the negative expected cut value (for minimisation)."""
    qc = build_qaoa_circuit(params, layers, graph, costs)
    result = backend.run(qc, shots = None, save_input = False).result()
    probs = result.data(0)["probabilities"]
    return -(probs @ costs)


if __name__ == "__main__":
    n_qubits = 15
    layers = 8
    backend = AerSimulator(method="statevector", device = "GPU")

    # Generate a random graph and pre-compute costs
    G = nx.gnp_random_graph(n_qubits, 0.5, seed=42)
    costs = compute_maxcut_costs(G)

    # Initial guess
    init_params = np.tile([0.5, 1.5], layers)

    res = minimize(
        qaoa_expectation,
        init_params,
        args=(backend, layers, G, costs),
        method="COBYLA",
        options={"maxiter": 100},
    )

    print("Optimal parameters:", res.x)
    print("Optimal value:", res.fun)
