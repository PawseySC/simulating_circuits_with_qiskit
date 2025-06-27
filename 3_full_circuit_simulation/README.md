# Full-Circuit Simulation

Here we replace NumPy-heavy phase-shift with a fully parameterised quantum circuit to reduce the memory-footprint use and speed up evaluation.

* `qaoa_cost_unitary` builds the phase-shift layer entirely with `RZZ` gates, so every operation is now part of the circuit itself.

* `graph_to_sparse_pauli_op` creates the cost operator $\hat{C}$ as a `SparsePauliOp` - this is the observable whose expectation value we will measure.

* The circuit is parameterised with two `ParameterVector`s (`gamma`, `beta`).
* By constructing a parameterised circuit we can define and transpile it once, and reuse the same circuit - binding new numbers to its parameters before each evaluation of the objective function.

* The `Estimator` class in `qaoa_objective_function_estimator` evaluates
  $\langle \hat{C} \rangle$ exactly (by passing `approximation = True` and `shots=None`). It returns a single scalar, all we need for the classical optimiser.

With one transpiled template reused across iterations and optimised computation of the expectation value, the program’s peak memory drops by roughly an order of magnitude. For larger $n$ this approach also runs far faster than our earlier NumPy-based method.
