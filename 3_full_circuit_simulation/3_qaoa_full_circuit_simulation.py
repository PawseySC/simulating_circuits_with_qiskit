from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator

import networkx as nx
import numpy as np
from scipy.optimize import minimize


def apply_initial_state(qc: QuantumCircuit) -> None:
    """Initial uniform superposition."""
    for q in range(qc.num_qubits):
        qc.h(q)


def apply_cost_unitary(qc: QuantumCircuit,
                       graph: nx.Graph,
                       gamma) -> None:
    """RZZ on each edge with angle 2*gamma."""
    for i, j in graph.edges():
        qc.rzz(2 * gamma, i, j)


def apply_mixer_unitary(qc: QuantumCircuit,
                        beta) -> None:
    """RX on every qubit."""
    for q in range(qc.num_qubits):
        qc.rx(2 * beta, q)


def build_qaoa_circuit(layers: int,
                       graph: nx.Graph) -> tuple[QuantumCircuit, list]:
    """Return the parameterised QAOA circuit and its parameter list."""
    qc = QuantumCircuit(graph.number_of_nodes())

    gammas = ParameterVector("gamma", length=layers)
    betas = ParameterVector("beta", length=layers)

    apply_initial_state(qc)

    for l in range(layers):
        apply_cost_unitary(qc, graph, gammas[l])
        apply_mixer_unitary(qc, betas[l])

    params = list(gammas) + list(betas)
    return qc, params


# Observable, the max-cut cost operator.
def graph_to_sparse_pauli_op(graph: nx.Graph) -> SparsePauliOp:
    """Return 0.5 * sum_{(u,v) in E} Z_u Z_v as a SparsePauliOp."""
    n = graph.number_of_nodes()
    paulis = []
    coeffs = []

    for u, v in graph.edges():
        label = ["I"] * n
        label[u] = "Z"
        label[v] = "Z"
        paulis.append("".join(label))
        coeffs.append(0.5)

    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))


def qaoa_expectation(params: np.ndarray,
                     qc: QuantumCircuit,
                     obs: SparsePauliOp,
                     estimator: Estimator) -> float:
    """Negative expected cut value (minimisation target)."""
    job = estimator.run(qc, obs, params)
    value = job.result().values[0]
    return -value


if __name__ == "__main__":
    n_qubits = 15
    layers = 8

    # Problem graph
    graph = nx.gnp_random_graph(n_qubits, 0.5, seed=42)

    # Build circuit and observable
    qc, qc_params = build_qaoa_circuit(layers, graph)
    cost_op = graph_to_sparse_pauli_op(graph)

    # Estimator primitive (no shots with approximation = True: 
    # exact statevector expectation)
    estimator = Estimator(
        backend_options={},
        run_options={"shots": None},
        approximation=True,
        skip_transpilation=True,
    )

    init_params = np.tile([0.5, 1.5], layers)

    result = minimize(
        qaoa_expectation,
        init_params,
        args=(qc, cost_op, estimator),
        method="COBYLA",
        options={"maxiter": 100},
    )

    print("Optimal parameters:", result.x)
    print("Optimal value:", result.fun)
