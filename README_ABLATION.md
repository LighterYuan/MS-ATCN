# MS-ATCN Binary Ablation Patch

This patch only adds:

- `run_ablation_binary.sh`

It does not modify:

- `src/train.py`
- `src/model.py`
- `src/data.py`
- `src/metrics.py`
- any baseline files

Run from project root:

```bash
cd /root/autodl-tmp/project2
unzip -o ms_atcn_ablation_patch.zip
chmod +x run_ablation_binary.sh
bash run_ablation_binary.sh
```

Generated output directories:

- `results/cic_binary/ablation/no_multiscale`
- `results/cic_binary/ablation/no_attention`
- `results/cic_binary/ablation/no_focal_loss`
- `results/cic_binary/ablation/full_ms_atcn`
- `results/nsl_binary/ablation/...`
- `results/unsw_binary_fix/ablation/...`
