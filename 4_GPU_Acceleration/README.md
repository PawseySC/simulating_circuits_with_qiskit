# GPU Acceleration and Profiling on AMD GPUs (ROCm)

To use GPU acceleration on Setonix first setup a GPU-enabled Qiskit-Aer python virtual environment as describe here:

https://github.com/PawseySC/qiskit_aer_rocm

To enable GPU acceleration.

```python
sim = AerSimulator(device="GPU")                     # state-vector
est = Estimator(backend_options={"device": "GPU"})   # primitives
```

## Profiling

Scalene is great for CPU-side Python, but it has no hooks into AMD GPUs.
Instead we’ll use **rocprof** – a ROCm profiler that can grab both time stamps and low-level hardware counters.

A **metric file** tells rocprof exactly which performance-monitor counters (PMCs) to sample for every kernel launch.

```text
# metrics_input.txt
pmc: GPUBusy          # overall device activity
pmc: VALUUtilization  # ALU utilisation
pmc: MemUnitBusy      # memory-pipeline load
pmc: L2CacheHit       # cache-hit ratio
```

Run your programme through rocprof like so:

```bash
rocprof -i metrics_input.txt      \   # load the PMC list
        --stats --basenames on    \   # add timing CSV, shorten kernel names
        -o results                \   # prefix for the output CSVs
        python 4_2_qaoa_partial_circuit_simulation_gpu.py
```

rocprof creates two CSV files – `results.csv` (the four counters) and `results.stats.csv` (per-kernel timing / occupancy).

## What do the counters mean?

| Counter             | In plain English                                                                  |
| ------------------- | --------------------------------------------------------------------------------- |
| **GPUBusy**         | *kernel time* with at least one Compute Unit active.                         |
| **VALUUtilization** | How busy the arithmetic units are (% of active threads issuing ALU instructions). |
| **MemUnitBusy**     | How saturated the global-memory pipeline is.                                      |
| **L2CacheHit**      | Fraction of memory requests served from on-chip L2 (locality).                    |

### Rules of-thumb:

* **GPUBusy < 60 %**:  GPU often idle.
* **VALUUtil ≥ 70 % & MemUnit ≤ 50 %**: compute-bound.
* **MemUnit ≥ 70 % & L2Hit < 60 %**:  memory-bound.

#### Glossary

| Term                  | Quick definition                                                  |
| --------------------- | ----------------------------------------------------------------- |
| **Kernel**            | One GPU function launch executed by many threads.                 |
| **CU (Compute Unit)** | AMD’s hardware block that schedules & runs threads (≈ NVIDIA SM). |
| **VALU**              | Vector ALU that performs SIMD maths inside a CU.                  |
| **ALU**               | Generic arithmetic-logic unit (VALU is the vector version).       |
| **Threads**           | Lightweight contexts that run in lock-step batches (wavefronts).  |
| **L2 Cache**          | On-die cache shared by all CUs, reducing HBM/DRAM traffic.        |
