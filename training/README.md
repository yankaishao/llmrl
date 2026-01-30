# Symbolic RL Training (CPU)

Install CPU-only PyTorch to avoid large CUDA wheels:
```
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install stable-baselines3 gymnasium numpy tensorboard
```

Train a quick policy:
```
python train_ppo.py --total-timesteps 50000 --save-path ../hri_safety_ws/policies/ppo_policy.zip
```

Evaluate and export CSV:
```
python eval_policy.py --policy-path ../hri_safety_ws/policies/ppo_policy.zip --episodes 100 --out-csv eval_results.csv
```

The training script writes `ppo_policy.meta.json` alongside the policy.
