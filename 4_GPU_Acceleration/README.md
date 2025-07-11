# GPU Acceleration and Profiling on AMD GPUs (ROCm)

To use GPU acceleration on Setonix first setup a GPU-enabled Qiskit-Aer python virtual environment as describe here:

https://github.com/PawseySC/qiskit_aer_rocm

To enable GPU acceleration.

```python
sim = AerSimulator(device="GPU")                     # state-vector
est = Estimator(backend_options={"device": "GPU"})   # primitives
```

## Profiling

Scalene can be used on Setonix for CPU-side Python, but it has no hooks into AMD GPUs.
Instead we’ll use **rocprof** – a ROCm profiler that can record time stamps and access low-level hardware counters.

A **metric file** tells rocprof exactly which performance-monitor counters (PMCs) to sample for every kernel launch.

```text
# metrics_input.txt
pmc: GPUBusy          # overall device activity
pmc: VALUUtilization  # ALU utilisation
pmc: MemUnitBusy      # memory-pipeline load
pmc: L2CacheHit       # cache-hit ratio
```

Run your program through rocprof like so:

```bash
rocprof -i metrics_input.txt      \   # PMC list
        --stats --basenames on    \   # export timing CSV, shorten kernel names
        -o results                \   # prefix for the output CSVs
        python 4_2_qaoa_partial_circuit_simulation_gpu.py
```

The out will be two CSV files – `results.csv` (the four counters) and `results.stats.csv` (per-kernel timing / memory occupancy).
These files aren't very human-readable, `gpu_util.py` parses the rocprof output using Pandas and reports useful figures of merit:

```bash
python gpu_util.py results.stats.csv
```

## What do the counters mean?

| Counter             | Definition                                                                        |
| ------------------- | --------------------------------------------------------------------------------- |
| **GPUBusy**         | *kernel time* with at least one Compute Unit active.                              |
| **VALUUtilization** | How busy the arithmetic units are (% of active threads issuing ALU instructions). |
| **MemUnitBusy**     | How saturated the global-memory pipeline is.                                      |
| **L2CacheHit**      | Fraction of memory requests served from on-chip L2 (locality).                    |

### Rules of-thumb:

* **GPUBusy < 60 %**:  GPU often idle.
* **VALUUtil ≥ 70 % & MemUnit ≤ 50 %**: compute-bound.
* **MemUnit ≥ 70 % & L2Hit < 60 %**:  memory-bound.

#### Glossary

| Term                  | Definition                                                        |
| --------------------- | ----------------------------------------------------------------- |
| **Kernel**            | A GPU function that is executed in-parallel on many threads.      |
| **CU (Compute Unit)** | AMD’s hardware block that schedules & runs threads (≈ NVIDIA SM). |
| **VALU**              | Vector ALU that performs vectorised math operations inside a CU.  |
| **ALU**               | Generic arithmetic-logic unit (VALU is the vector version).       |
| **Threads**           | Processes that run in-parralell in lock-step batches (wavefronts).|
| **L2 Cache**          | On-die cache (memory) shared by all CUs.                          |
