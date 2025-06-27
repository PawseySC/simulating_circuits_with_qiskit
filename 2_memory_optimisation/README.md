**Memory-Saving Tips**

Will minimal changes to the code we can reduce the overall memory footprint.

1. Re-use one temporary NumPy array for the phase-shift values instead of allocating a new array in every layer.
2. Ask Aer to return the final probability vector (`qc.save_probabilites()`) rather than the full statevector.
3. Skip saving the input circuit. Storing it takes roughly $p \times 2^n$ complex numbers - effectively $p$ extra copies of the statevector.
