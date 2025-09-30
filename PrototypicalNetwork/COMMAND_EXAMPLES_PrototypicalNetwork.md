# Command Examples — Prototypical Network Pipeline

This file collects ready-to-run command-line examples for
`prototypical_network_pipeline.py`. Copy–paste and adapt paths as needed.

> Assumptions:
> - Data directory: `./data`
> - Output directory: `./results`
> - Script: `prototypical_network_pipeline.py` in the current folder


## 1) Quick start

### Linux / macOS (bash)
```bash
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results
```

### Windows (PowerShell)
```powershell
py .\prototypical_network_pipeline.py --data-dir .\data --out-dir .\results
```

Default behavior: computes both regimes and saves **only the fused ranking** with Dyson weight α = 0.50 (min–max normalization).


## 2) Emit specific outputs

### Dyson-only
```bash
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results --emit dyson
# PowerShell
py .\prototypical_network_pipeline.py --data-dir .\data --out-dir .\results --emit dyson
```

### Normal-only
```bash
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results --emit normal
# PowerShell
py .\prototypical_network_pipeline.py --data-dir .\data --out-dir .\results --emit normal
```

### Dyson + Normal + Fused
```bash
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results --emit dyson,normal,fused
# PowerShell
py .\prototypical_network_pipeline.py --data-dir .\data --out-dir .\results --emit dyson,normal,fused
```


## 3) Fusion weights & normalization

### Multiple fused weights (α ∈ {0.9, 0.7, 0.5})
```bash
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results \
  --emit fused --fuse-weights 0.9,0.7,0.5
# PowerShell
py .\prototypical_network_pipeline.py --data-dir .\data --out-dir .\results `
  --emit fused --fuse-weights 0.9,0.7,0.5
```

### Change normalization scheme (z-score)
```bash
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results \
  --emit fused --norm-scheme zscore
# PowerShell
py .\prototypical_network_pipeline.py --data-dir .\data --out-dir .\results `
  --emit fused --norm-scheme zscore
```


## 4) Choose the ProtoNet scoring method

`--proto-method` ∈ `{probability | distance | cosine | all}`

```bash
# Probability (softmax over negative squared distances) — recommended
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results --emit fused --proto-method probability

# Distance (use -||z - prototype||^2, higher = closer)
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results --emit fused --proto-method distance

# Cosine similarity ([-1,1] rescaled to [0,1])
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results --emit fused --proto-method cosine

# Run all three methods and save outputs with method suffixes
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results --emit dyson,normal,fused --proto-method all
```
PowerShell equivalents (line-wrapped):
```powershell
py .\prototypical_network_pipeline.py --data-dir .\data --out-dir .\results `
  --emit dyson,normal,fused --proto-method all
```


## 5) Metrics (Precision / Recall / F1 @k)

```bash
# Treat Dyson as the positive class (default)
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results \
  --emit fused --emit-metrics --metrics-target dyson

# Treat Normal as the positive class
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results \
  --emit fused --emit-metrics --metrics-target normal
```
PowerShell:
```powershell
py .\prototypical_network_pipeline.py --data-dir .\data --out-dir .\results `
  --emit fused --emit-metrics --metrics-target dyson
```


## 6) File name overrides

```bash
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results \
  --train-file-dyson train.csv --train-file-normal trainNormal.csv \
  --test-file test_normal.csv --num-ds-file num_ds.txt --num-norm-file numNorm.txt
```


## 7) Duplicate ID policy (before fusion)

```bash
# Options: error | drop-keep-best | mean | max | min
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results \
  --emit fused --on-duplicate drop-keep-best
```


## 8) Reproducibility, device, and training hyperparameters

```bash
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results \
  --proto-method probability --epochs 100 --batch-size 64 --lr 5e-4 \
  --embedding-dim 64 --hidden-dim 128 --seed 42 --device auto
```

Force CPU or CUDA:
```bash
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results --device cpu
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results --device cuda
```


## 9) Full example (multi-weight fusion + all methods + metrics)

### bash
```bash
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results \
  --emit dyson,normal,fused --proto-method all \
  --fuse-weights 0.9,0.7,0.5 --norm-scheme minmax \
  --emit-metrics --metrics-target dyson \
  --epochs 100 --batch-size 64 --lr 5e-4 --embedding-dim 64 --hidden-dim 128 \
  --seed 42 --device auto
```

### PowerShell
```powershell
py .\prototypical_network_pipeline.py --data-dir .\data --out-dir .\results `
  --emit dyson,normal,fused --proto-method all `
  --fuse-weights 0.9,0.7,0.5 --norm-scheme minmax `
  --emit-metrics --metrics-target dyson `
  --epochs 100 --batch-size 64 --lr 5e-4 --embedding-dim 64 --hidden-dim 128 `
  --seed 42 --device auto
```


## 10) Windows CMD (caret `^` line wrap)

```cmd
py prototypical_network_pipeline.py --data-dir .\data --out-dir .\results ^
  --emit dyson,normal,fused ^
  --proto-method all ^
  --fuse-weights 0.9,0.7,0.5 ^
  --emit-metrics --metrics-target dyson
```


---

**Notes**
- Ensure CSVs include `source_id` followed by numeric feature columns.
- `num_ds.txt` and `numNorm.txt` must contain an integer (first integer in the file is parsed).
- The script writes logs (`protonet_*.log`) and a config snapshot (`protonet_config.json`) to the output directory.
