# Quick start

Use a Linux x86-64 CPU environment and Python 3.13 (verified with 3.13.5). Allow at least 1 GiB of working space for the extracted repository and run copies, plus Python and dependency space. The commands run without service credentials.

```bash
git clone https://github.com/skatzrskx55q/twm-induct-e36.git
cd twm-induct-e36
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python verify_integrity.py
python run.py check
python run.py --keep ../twm-induct-e36-demo demo
```

Use a fresh path in `--keep` to preserve results. You can also extract the separately attached [research ZIP](https://github.com/skatzrskx55q/twm-induct-e36/releases/tag/v0.1.0-research) and start at virtual environment setup.

Run `python example.py` for a checkpoint-backed toy prediction. `python run.py demo` runs the original N2 benchmark.

Selected checks exercise the documented native entry points. See [the validation record](validation/linux-cpu.md) and [original guide](research/E36/README_RU.md).
