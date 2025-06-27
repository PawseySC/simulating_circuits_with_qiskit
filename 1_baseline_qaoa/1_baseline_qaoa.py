from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import DiagonalGate

import numpy as np
import networkx as nx
from scipy.optimize import minimize

def cut_value(bitstring, G):
    z = [int(b) for b in bitstring]
    return sum(z[u] != z[v] for u, v in G.edges()) / 2

def compute_maxcut_costs(G):
    """
    Return a numpy array `costs` of length 2^n, where
    costs[k] = cut_value(bitstring(k), G)
    """
    n = G.number_of_nodes()
    num_states = 1 << n
    costs = np.empty(num_states, dtype=float)
    for k in range(num_states):
        bitstr = format(k, f'0{n}b')
        costs[k] = cut_value(bitstr, G)
    return costs

def qaoa_initial_state(qc, G):
    for q in range(G.number_of_nodes()):
        qc.h(q)
    return qc

def qaoa_cost_unitary(qc, qreg, costs, gamma):
    UQ = DiagonalGate(np.exp(-1j * gamma * costs))
    qc.append(UQ, qreg)
    return qc

def qaoa_mixing_unitary(qc, G, beta):
    for q in range(G.number_of_nodes()):
        qc.rx(2 * beta, q)
    return qc

def make_qaoa_circuit(params, layers, G, costs):
    qc = QuantumCircuit(G.number_of_nodes())
    gammas, betas = np.split(params, 2)
    qreg = range(G.number_of_nodes())

    qc = qaoa_initial_state(qc, G)
    for layer in range(layers):
        qc = qaoa_cost_unitary(qc, qreg, costs, gammas[layer])
        qc = qaoa_mixing_unitary(qc, G, betas[layer])

    qc.save_state()
    return qc

def qaoa_objective_function(input_parameters, backend, layers, G, costs):
    qc = make_qaoa_circuit(input_parameters, layers, G, costs)
    job = backend.run(qc)
    result = job.result()
    psi = result.get_statevector()
    probs = np.abs(psi) ** 2
    return -probs @ costs

if __name__ == "__main__":
    n = 15
    layers = 16
    G = nx.gnp_random_graph(n, 0.5, seed=42)
    costs = compute_maxcut_costs(G)

    initial_parameters = np.array([0.5, 1.5] * layers)

    backend = AerSimulator(method = "statevector")

    opt_result = minimize(
        qaoa_objective_function,
        initial_parameters,
        args=(backend, layers, G, costs),
        method="COBYLA",
        options={"maxiter": 100}
    )

    print("Optimal parameters:", opt_result.x)
    print("Optimal value:", opt_result.fun)
