from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator

import networkx as nx
from scipy.optimize import minimize

def qaoa_initial_state(qc, G):
    # solution
    n = G.number_of_nodes()
    for q in range(n):
        qc.h(q)
    return qc

def qaoa_cost_unitary(qc, G, gamma):
    # solution
    for (i, j) in G.edges():
        qc.rzz(2 * gamma, i, j)
    return  qc

def qaoa_mixing_unitary(qc, G, beta):
    # solution
    n = G.number_of_nodes()
    for q in range(n):
        qc.rx(2 * beta, q)
    return qc

def make_qaoa_circuit(qc, layers, G):

    gammas = ParameterVector("gammas", length = layers)
    betas = ParameterVector("betas", length = layers)

    qc = qaoa_initial_state(qc, G)
    
    for layer in range(layers):
        qc = qaoa_cost_unitary(qc, G, gammas[layer])
        qc = qaoa_mixing_unitary(qc, G, betas[layer])
            
    return qc, list(gammas) + list(betas)

def graph_to_sparse_pauli_op(G):
    nodes = list(G.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    n_qubits = len(nodes)

    pauli_strings = []
    coeffs = []
    for u, v, data in G.edges(data=True):
        p = ['I'] * n_qubits
        p[idx[u]] = 'Z'
        p[idx[v]] = 'Z'
        pauli_strings.append(''.join(p))
        coeffs.append(0.5)

    return SparsePauliOp.from_list(list(zip(pauli_strings, coeffs)))


def qaoa_objective_function_estimator(input_parameters, qc, circuit_parameters, observables, estimator):

    exp_val = estimator.run(qc, observables, input_parameters)
    exp = - exp_val.result().values[0]
    return exp


if __name__ == "__main__":
    n = 25
    backend  = AerSimulator(method = "statevector", device = "GPU")

    layers = 8
    G = nx.gnp_random_graph(n, 0.5, seed = 42)

    C = graph_to_sparse_pauli_op(G)
    qc = QuantumCircuit(G.number_of_nodes())
    qc, circuit_parameters = make_qaoa_circuit(qc, layers, G)

    qc = transpile(qc,
                              backend=backend,
                             optimization_level=2)

    
    estimator = Estimator(backend_options = {"method":"statevector", "device": "GPU"}, run_options = {"shots": None}, approximation = True, skip_transpilation = True)

    initial_parameters = layers * [0.5, 1.5]
    opt_result = minimize(qaoa_objective_function_estimator, initial_parameters, args = (qc, circuit_parameters, C, estimator), method = "COBYLA", options = {"maxiter":100})
