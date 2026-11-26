# Reproducibility

1. Install the pinned Python dependencies using [QUICKSTART.md](QUICKSTART.md).
2. Verify the source SHA-256 manifest with `python verify_integrity.py`.
3. Run the frozen native check. The historical command is `python verify_manifest.py`.
4. Run `python run.py --keep ../twm-induct-e36-repro demo` to retain outputs and compare them against [archived results](research/E36/E36_RESULTS/E36_METRICS.json). The direct native demo command is `python e36_n2_rollout.py`.

## Training or calibration

The native command trains the feature mapping and uses support transitions for the episode-specific ridge step.

```bash
python run.py --keep ../twm-induct-e36-training native -- e36_meta_ridge.py --steps 750 --seeds 17,29,43 --batch 16
```

[original guide](research/E36/README_RU.md) · [Original release guide](docs/REPRODUCTION.md) · [Release validation](validation/linux-cpu.md)
