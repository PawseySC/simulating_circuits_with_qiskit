from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import DiagonalGate

import gc

import numpy as np
import networkx as nx
from scipy.optimize import minimize

def cut_value(bitstring, G):
    z = [int(b) for b in bitstring]
    return sum(z[u] != z[v] for u, v in G.edges())/2


def compute_maxcut_costs(G):
    """
    Return a numpy array `costs` of length 2^n, where
    costs[k] = cut_value(bitstring(k), G)
    and bitstring(k) is the n-bit binary representation of k.
    """
    n = G.number_of_nodes()
    num_states = 1 << n  # 2**n
    costs = np.empty(num_states, dtype=float)
    for k in range(num_states):
        bitstr = format(k, f'0{n}b')
        costs[k] = cut_value(bitstr, G)
    return costs

def qaoa_initial_state(qc, G):
    # solution
    n = G.number_of_nodes()
    for q in range(n):
        qc.h(q)
    return qc

n = 14

layers = 8
G = nx.gnp_random_graph(n, 0.5, seed = 42)


costs = compute_maxcut_costs(G) 


_phases = np.empty_like(costs, dtype=np.complex128)   # global scratch
def qaoa_cost_unitary(qc, qreg, gamma):
    np.multiply(costs, -1j*gamma, out=_phases, casting='unsafe')
    np.exp(_phases, out=_phases)
    qc.append(DiagonalGate(_phases.copy()), qreg)     # 1 small copy
    return qc

def qaoa_mixing_unitary(qc, G, beta):
    # solution
    n = G.number_of_nodes()
    for q in range(n):
        qc.rx(2 * beta, q)
    return qc

def make_qaoa_circuit(params, layers, G):

    qc = QuantumCircuit(G.number_of_nodes())
    gammas, betas = np.split(params, 2)

    qreg = range(G.number_of_nodes())

    qc = qaoa_initial_state(qc, G)
    
    for layer in range(layers):
        qc = qaoa_cost_unitary(qc, qreg, gammas[layer])
        qc = qaoa_mixing_unitary(qc, G, betas[layer])
            
    #qc.save_probabilities()
    qc.save_statevector()

    return qc

def qaoa_objective_function(input_parameters, costs):
    backend  = AerSimulator(method = "statevector", device = "GPU", precision = "single", max_memory_mb = -1)

    qc = make_qaoa_circuit(input_parameters, layers, G)
    #qc = transpile(qc, backend)
    #job = backend.run(qc, shots = None, save_input = False)
    job = backend.run(qc, shots = None)
    #probs =job.result().data(0)['probabilities']
    probs = np.abs(job.result().get_statevector())**2
    exp = -probs @ costs
    #del(qc)
    #del(backend)
    #gc.collect()
    return exp


if __name__ == "__main__":
    initial_parameters = layers * [0.5, 1.5]
    opt_result = minimize(qaoa_objective_function, initial_parameters, args = (costs), method = "COBYLA", options = {"maxiter":100})
