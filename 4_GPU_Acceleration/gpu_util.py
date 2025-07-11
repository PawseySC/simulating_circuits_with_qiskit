"""
gpu_wall_util.py  <metrics_input.stats.csv>

Prints the fraction of wall-clock time the GPU spent executing
kernels + derived metrics
"""

import sys, pandas as pd

if len(sys.argv) != 2:
    sys.exit("Usage: python gpu_wall_util.py metrics_input.stats.csv")

stats = pd.read_csv(sys.argv[1])

# ------------------------------------------------------------------
# 1) Compute program runtime and GPU utilisation.
# ------------------------------------------------------------------
t_start = stats['BeginNs'].min()
t_end   = stats['EndNs'].max()
wall_ns = t_end - t_start
busy_ns = stats['DurationNs'].sum()
wall_util = 100.0 * busy_ns / wall_ns

# ------------------------------------------------------------------
# 2) Process PMCs data  (we need the *metrics* CSV for this part)
# ------------------------------------------------------------------
metrics_csv = sys.argv[1].replace('.stats.csv', '.csv')
pmc_cols = ["GPUBusy", "VALUUtilization", "MemUnitBusy", "L2CacheHit"]
# Average over program run-time.
pmc = pd.read_csv(metrics_csv)[pmc_cols].mean() 

#-------------------------------------------------------------------
# 3) Summarise Results.
#-------------------------------------------------------------------
print(f"""
Wall-clock GPU utilisation : {wall_util:5.1f} %
Avg GPUBusy                : {pmc['GPUBusy']         :5.1f} %
Avg VALUUtilization        : {pmc['VALUUtilization'] :5.1f} %
Avg MemUnitBusy            : {pmc['MemUnitBusy']     :5.1f} %
Avg L2CacheHit             : {pmc['L2CacheHit']      :5.1f} %
""")

# ------------------------------------------------------------------
# 3) "Rule of thumb" verdict.
# ------------------------------------------------------------------
if wall_util < 60:
    verdict = "GPU Under-utilised."
elif pmc["VALUUtilization"] >= 70 and pmc["MemUnitBusy"] <= 50:
    verdict = "Compute-bound."
elif pmc["MemUnitBusy"] >= 70 and pmc["VALUUtilization"] <= 50 and pmc["L2CacheHit"] < 60:
    verdict = "Memory-bound."
else:
    verdict = "Balanced / probably OK."

print("Verdict :", verdict)
