# `1_baseline_qaoa.py`

* This is a minimal QAOA workflow for Max-Cut, simulated with Qiskit-Aer and optimised by SciPy.
* It solves the max-cut problem for a Random Erdős–Rényi graph, $n = 15$, edge probability $p = 0.5$.
* Solution costs:

  * `cut_value(bitstring, G)`: computes the  cut size $\tfrac12\bigl|\{(u,v)\in E : z_u \ne z_v\}\bigr|$.
  * `compute_maxcut_costs(G)`: returns a NumPy array of all $2^{n}$ costs for the diagonal cost unitary.
* The circuit:

  * `qaoa_initial_state` – Hadamards produce $\lvert +\rangle^{\otimes n}$.
  * `qaoa_cost_unitary` – `DiagonalGate` implements $U_C(\gamma)=\mathrm{diag}\bigl(e^{-i\gamma C_z}\bigr)$.
  * `qaoa_mixing_unitary` – per-qubit $R_X$ giving $U_B(\beta)=\bigotimes_{j} e^{-i\beta X_j}$.
* The full circuit is constructed with `make_qaoa_circuit(params, p, G, costs)`,  repeating $(U_C,U_B)$ for $p$ layers 
* The final operation `save_state()` tells the backend to return the final statevector.
* The objective function is defined by `qaoa_objective_function`, which returns $-\langle C\rangle$; minimising it maximises the expected cut.
* Statevector simulation performed by Qiskit-Aer: `AerSimulator(method="statevector")`.
* We use a low number of COBYLA iterations (100) for benchmarking purposes - should be higher for final results.

## Profiling 

To scale up effectively we need to understand how our program is behaves.

There are many different tools for this, `cProfile` is included with the Python standard libraries.

Here we'll use the lightweight sampling profiler **Scalene**.

```bash
python -m pip install scalene
python -m scalene 1_baseline_qaoa.py          # HTML report
python -m scalene --cli 1_baseline_qaoa.py    # CLI report
```

Scalene shows CPU, GPU (NVIDIA), and memory hotspots.
