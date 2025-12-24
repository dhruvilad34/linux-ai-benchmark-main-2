import json, sys
from collections import defaultdict

def load(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def get(rows, tag, gpu_id):
    # last occurrence wins
    for r in reversed(rows):
        if r.get("tag") == tag and r.get("gpu_id") == gpu_id:
            return r
    return None

def fmt(x):
    return "NA" if x is None else f"{x:.3f}"

def main(path):
    rows = load(path)
    gpus = sorted({r.get("gpu_id") for r in rows if "gpu_id" in r and r.get("gpu_id") is not None})

    print("\n=== MEMORY BEFORE/AFTER SUMMARY (GB) ===\n")
    header = (
        "GPU | Stage                  | vram_used | allocated | reserved\n"
        "----|------------------------|----------:|----------:|---------:"
    )
    print(header)

    for gpu in gpus:
        stages = [
            (f"before_model_load_gpu{gpu}", f"after_model_load_gpu{gpu}", "delta_model_load"),
            (f"before_inference_gpu{gpu}", f"after_inference_gpu{gpu}", "delta_inference"),
        ]

        for before_tag, after_tag, delta_name in stages:
            b = get(rows, before_tag, gpu)
            a = get(rows, after_tag, gpu)

            def val(r, k):
                return None if r is None else r.get(k)

            b_vram = val(b, "gpu_vram_used_gb")
            a_vram = val(a, "gpu_vram_used_gb")
            b_alloc = val(b, "gpu_mem_allocated_gb")
            a_alloc = val(a, "gpu_mem_allocated_gb")
            b_res = val(b, "gpu_mem_reserved_gb")
            a_res = val(a, "gpu_mem_reserved_gb")

            # before
            print(f"{gpu}   | {before_tag:22} | {fmt(b_vram):>9} | {fmt(b_alloc):>9} | {fmt(b_res):>8}")
            # after
            print(f"{gpu}   | {after_tag:22} | {fmt(a_vram):>9} | {fmt(a_alloc):>9} | {fmt(a_res):>8}")

            # delta (if both present)
            if all(v is not None for v in [b_vram, a_vram, b_alloc, a_alloc, b_res, a_res]):
                print(
                    f"{gpu}   | {delta_name:22} | {fmt(a_vram - b_vram):>9} | {fmt(a_alloc - b_alloc):>9} | {fmt(a_res - b_res):>8}"
                )
            else:
                print(f"{gpu}   | {delta_name:22} |        NA |        NA |       NA")
    print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/summarize_memory.py logs/memory_snapshots.jsonl")
        sys.exit(1)
    main(sys.argv[1])
