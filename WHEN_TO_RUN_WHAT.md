# When to Run What - Decision Guide

## Quick Decision Tree

```
Do you want to compare BEFORE vs AFTER performance?
│
├─ YES → Run: python kernel_benchmark.py
│         (This runs everything automatically - before, optimize, after, compare)
│
└─ NO → Do you want kernel optimization applied?
        │
        ├─ YES → Run: python agent_concurrency.py --optimize
        │         (Applies optimization, then runs workload)
        │
        └─ NO → Run: python main.py
                 (Normal run without optimization)
```

## Detailed Scenarios

### Scenario 1: "I want to benchmark and see improvements"
**When:** You need to measure and report kernel optimization improvements
**Run:** 
```bash
python kernel_benchmark.py
```
**Why:** This runs the full before/after comparison and generates a report table

---

### Scenario 2: "I just want to run with optimization enabled"
**When:** You want optimized performance but don't need comparison data
**Run:**
```bash
python agent_concurrency.py --optimize
```
**Why:** Faster - runs once with optimization, no comparison overhead

---

### Scenario 3: "I want to apply optimization manually, then run my own tests"
**When:** You want to control when optimization is applied
**Run:**
```bash
sudo ./optimize_kernel.sh    # Apply optimization
python main.py               # Run your workload normally
```
**Why:** Gives you full control over timing

---

### Scenario 4: "I just want to run normally without optimization"
**When:** Baseline run or testing without kernel changes
**Run:**
```bash
python main.py
# OR
python agent_concurrency.py  # (without --optimize flag)
```
**Why:** Standard execution without any kernel tuning

---

## Use Case Examples

### 📊 **For Your Report/Paper**
**Use:** `python kernel_benchmark.py`
- Generates comparison table you can copy/paste
- Shows percentage improvements
- Creates JSON files with all metrics
- **Time:** ~2x longer (runs twice)

### 🚀 **For Production/Regular Use**
**Use:** `python agent_concurrency.py --optimize`
- One-time optimization, then runs
- Faster than benchmark
- Still gets optimized performance
- **Time:** Normal runtime

### 🔬 **For Experimentation**
**Use:** Manual approach
```bash
sudo ./optimize_kernel.sh
python main.py --config config/config.yaml --num_tasks 10
```
- Full control
- Can test different optimization combinations
- **Time:** Normal runtime

### 🧪 **For Quick Testing**
**Use:** `python main.py`
- No optimization overhead
- Fastest startup
- Good for debugging
- **Time:** Fastest

---

## Comparison Table

| Tool | When to Use | Time | Output |
|------|-------------|------|--------|
| `kernel_benchmark.py` | Need before/after comparison for report | ~2x runtime | Comparison table + JSON |
| `agent_concurrency.py --optimize` | Want optimization, don't need comparison | Normal runtime | Standard logs |
| `sudo ./optimize_kernel.sh` | Manual optimization only | Seconds | Kernel parameters set |
| `main.py` | Normal run, no optimization | Normal runtime | Standard logs |

---

## Recommended Workflow

### First Time Setup:
1. **Test normal run:**
   ```bash
   python main.py --num_tasks 1
   ```

2. **Run full benchmark:**
   ```bash
   python kernel_benchmark.py
   ```

3. **Review results:**
   ```bash
   cat kernel_benchmark_results/comparison.txt
   ```

### Regular Use:
- **For reports:** `python kernel_benchmark.py`
- **For daily work:** `python agent_concurrency.py --optimize`

---

## Quick Reference

```bash
# Benchmark (before/after comparison) - USE FOR REPORTS
python kernel_benchmark.py

# Run with optimization - USE FOR REGULAR WORK
python agent_concurrency.py --optimize

# Manual optimization only
sudo ./optimize_kernel.sh

# Normal run (no optimization)
python main.py
```





