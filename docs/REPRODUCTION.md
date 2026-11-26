# Reproduction guide

1. Follow installation in [README](../README.md). CPU only was tested in the release preparation environment.
2. `python verify_integrity.py` checks the byte-level hashes of distributed source artifacts without ML dependencies.
3. `python run.py check` invokes: `python verify_manifest.py` in `research/E36`. Exit code is propagated, including failures.
4. `python run.py --keep ../experiment demo` invokes: `python e36_n2_rollout.py`. The copied data and output remain in `../experiment`. Consult original README before long runs.
5. Training / extended experiment: `python e36_meta_ridge.py --steps 750 --seeds 17,29,43 --batch 16`. Execute in a separate copied tree, preserve originals, and read the historical protocol and unlock rules. Do not tune using the released EXAM labels. Bash commands may require Linux/WSL.

## Metric definitions
Open dev L2/L3 H16 exact final states 84/240 and 108/240; CART remains stronger on composition. Interpret H90/H16 and exact final state versus full trajectory using the native protocol, not as a shared benchmark. NLL is lower-is-better. Read the original source reports for the denominators and negative controls.

## Fresh preparation check
`PASS`. Raw log is in `../validation/current_check.log`. The current check is not full end-to-end retraining, and `demo` is a documented native entry point, not necessarily newly executed in full during preparation.
