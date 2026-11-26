# Model card — INDUCT E36 — Meta-Ridge World Dynamics

## Intended use
Controlled synthetic world-dynamics research, reproduction, ablation and extensions. Not validated for real-world forecasts, production automation, robotics, financial or safety decisions.

## Design
42,748 parameters; meta-trained feature mapping plus episode-specific ridge from eight support transitions, edge-to-node prediction.

## Evaluation
Open dev L2/L3 H16 exact final states 84/240 and 108/240; CART remains stronger on composition.

## Known limitations
Fixed binary graph and engineered features; no fresh sealed exam. Earlier seed17 retraining max weight diff 3.46e-6, not bitwise reproduction.

## Data and weights
The archive includes the meta-trained E36 seed checkpoints, E34 data and frozen synthetic evaluation protocol.

The original protocols define split, features and seeds. See the [release validation](../validation/linux-cpu.md).

## Technical entry points
[e36_meta_ridge.py](../research/E36/e36_meta_ridge.py) | [e36_n2_rollout.py](../research/E36/e36_n2_rollout.py) | [e36_controls.py](../research/E36/e36_controls.py) | [E36_RESULTS/](../research/E36/E36_RESULTS/) | [E34/](../research/E36/E34/) | [E36_FROZEN_PROTOCOL.json](../research/E36/E36_FROZEN_PROTOCOL.json)

## Evidence status
Historical status: Positive known-law result; no promotion. Fresh selected check: PASS. See `../validation/current_check.log`. Only the documented research environment is evaluated; there is no warranty of generalization.
