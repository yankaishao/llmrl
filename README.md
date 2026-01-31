# Mock Safety Pipeline (Step 1 + Step 2)

This folder contains the minimal ROS2 pipeline used by the mock LLM parser
and a scene summary stub. Everything lives under `hri_safety_ws`.

Workspace
- Path: `hri_safety_ws`
- Package: `hri_safety_core` (ament_python)

Quickstart: one command matrix eval
```
cd /path/to/llmrl
./evaluation/run_matrix_gazebo.sh --episodes 12 --results-dir results
python3 evaluation/make_report.py --results-dir results
```
Notes:
- Export `QWEN_API_KEY=...` to enable the Qwen parser (it falls back to mock if unset).
- The default instruction set is `instructions/base.txt`.
- If you see `rclpy` import errors under conda, run `conda deactivate` or use `/usr/bin/python3`.
You can also put `QWEN_API_KEY=...` into a repo-root `.env` file; runner scripts will load it automatically.
Parser-only (baseline) matrix:
```
python3 evaluation/ros_eval_runner.py --matrix-parser-only --episodes 12 --results-dir results
```
Reasonable-query eval (simulated user + limits, no timeouts):
```
python3 evaluation/ros_eval_runner.py --matrix-parser-only --episodes 5 --results-dir results \\
  --max-turns-per-episode 10 --max-repeat-action 3
```
Smoke test (build + node discovery + CLI help):
```
bash scripts/smoke_test.sh
```
RL integration sanity check (quick PPO + ROS eval):
```
bash evaluation/sanity_check_rl.sh
```

Tested environment (2026-01-29 20:03:46 CST)
- OS: Ubuntu 22.04.5 LTS (jammy)
- Kernel/arch: Linux 6.8.0-90-generic, x86_64
- ROS 2: Humble (`/opt/ros/humble/bin/ros2`)
- Gazebo Sim: 6.17.0
- Python: 3.13.5 (`/home/yankai/anaconda3/bin/python3`)
- Conda: base (`/home/yankai/anaconda3/bin/conda`)
- Key env vars:
```
AMENT_PREFIX_PATH=/opt/ros/humble
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/gazebo-11/plugins:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib
PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:/home/yankai/rt1_ws/tensor2robot:/home/yankai/rt1_ws:/home/yankai/rt1_ws/tensor2robot:/home/yankai/rt1_ws:/home/yankai/rt1_ws/tensor2robot
ROS_DISTRO=humble
ROS_LOCALHOST_ONLY=0
ROS_PYTHON_VERSION=3
ROS_VERSION=2
```

Nodes and topics
- `instruction_source` publishes `/user/instruction` (std_msgs/String)
- `instruction_listener` subscribes `/user/instruction` and logs messages
- `scene_summary_stub` publishes `/scene/summary` (std_msgs/String) at 1 Hz
- `mock_llm_parser` subscribes `/user/instruction` + `/scene/summary` and
  publishes `/nl/parse_result` (std_msgs/String JSON)
- `qwen_api_parser` subscribes `/user/instruction` + `/scene/summary` and
  publishes `/nl/parse_result` (Qwen API, JSON schema compatible)
- `parser_router` publishes `/nl/parse_result` and routes mock vs qwen
- `estimator_node` publishes `/safety/features` (std_msgs/String JSON)
- `rule_based_arbiter` publishes `/arbiter/action` and `/robot/utterance`

RL action set (6 actions)
- EXECUTE
- CONFIRM_YN
- CLARIFY_CHOICE
- ASK_POINT
- REFUSE_SAFE
- FALLBACK_HUMAN_HELP (safe downgrade / ask human to confirm or take over)

Build
```
conda deactivate
source /opt/ros/humble/setup.bash
cd /path/to/llmrl/hri_safety_ws
colcon build
```

Run (each terminal)
```
conda deactivate
source /opt/ros/humble/setup.bash
cd /path/to/llmrl/hri_safety_ws
source install/local_setup.bash
export AMENT_PREFIX_PATH=$PWD/install/hri_safety_core:$AMENT_PREFIX_PATH
```

Launch (mock pipeline)
```
ros2 launch hri_safety_core pipeline_mock.launch.py
```
Note: `instruction_source` is interactive, so run it in a separate terminal.

Verification
Terminal A:
```
ros2 run hri_safety_core instruction_listener
```

Terminal B:
```
ros2 run hri_safety_core scene_summary_stub
```

Terminal C:
```
ros2 run hri_safety_core mock_llm_parser
```

Terminal D:
```
ros2 topic echo /nl/parse_result
```

Terminal E:
```
ros2 run hri_safety_core instruction_source
```
Type a line such as:
```
give me the left cup
```

Expected: `/nl/parse_result` prints a JSON string with `candidates`, `score`,
`task_type`, and `clarify_templates`.

Step 3: Estimator (safety features)
Terminal A:
```
ros2 run hri_safety_core scene_summary_stub
```

Terminal B:
```
ros2 run hri_safety_core mock_llm_parser
```

Terminal C:
```
ros2 run hri_safety_core estimator_node
```

Terminal D:
```
ros2 topic echo /safety/features
```

Terminal E:
```
ros2 run hri_safety_core instruction_source
```
Type a line such as:
```
hand me that cup
```

Expected: `/safety/features` prints JSON with `amb`, `risk`, `conflict`,
`conflict_reason`, and `selected_top1_id`.

Step 4: Rule-based Arbiter (minimal closed loop)
Terminal A:
```
ros2 run hri_safety_core instruction_source
```

Terminal B:
```
ros2 run hri_safety_core scene_summary_stub
```

Terminal C:
```
ros2 run hri_safety_core mock_llm_parser
```

Terminal D:
```
ros2 run hri_safety_core estimator_node
```

Terminal E:
```
ros2 run hri_safety_core rule_based_arbiter
```

Terminal F:
```
ros2 topic echo /arbiter/action
```

Terminal G:
```
ros2 topic echo /robot/utterance
```

Try inputs like:
```
give me the left cup
hand me that cup
pass the knife
```

Expected:
- clear low-risk -> EXECUTE
- ambiguous -> ASK_POINT (or CLARIFY_CHOICE if top-2 exists)
- high-risk (knife) -> CONFIRM_YN
- unknown/no candidate -> REFUSE_SAFE

Step 5: Qwen API parser (mock vs qwen switch)
Environment:
```
export QWEN_API_KEY=...
```
Optional: if you already store a key in a `.env` file, you can set
`api_key_env` to that variable name instead of exporting again.

Launch (router, easiest):
```
ros2 launch hri_safety_core pipeline_router.launch.py parser_mode:=mock
```
Switch to Qwen:
```
ros2 launch hri_safety_core pipeline_router.launch.py parser_mode:=qwen
```

Option A: run Qwen parser directly
```
ros2 run hri_safety_core qwen_api_parser --ros-args \\
  -p model:=qwen3-max \\
  -p base_url:=https://dashscope.aliyuncs.com/compatible-mode/v1 \\
  -p api_key_env:=QWEN_API_KEY
```

Option B: route mock vs qwen
```
ros2 run hri_safety_core parser_router --ros-args -p parser_mode:=qwen \\
  -p model:=qwen3-max \\
  -p base_url:=https://dashscope.aliyuncs.com/compatible-mode/v1 \\
  -p api_key_env:=QWEN_API_KEY
```

Check output:
```
ros2 topic echo /nl/parse_result
```

Expected: `/nl/parse_result` is still valid JSON with the same schema. If the
API fails, the parser falls back to mock output and writes a fallback note.

Step 6: RL Arbiter (PPO policy)
Train a toy policy (symbolic env):
```
cd /path/to/llmrl/training
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
python train_ppo.py --total-timesteps 50000 --save-path ../hri_safety_ws/policies/ppo_policy.zip
```

Run with RL arbiter (direct node):
```
ros2 run hri_safety_core rl_arbiter_node --ros-args \\
  -p policy_path:=/path/to/llmrl/hri_safety_ws/policies/ppo_policy.zip \\
  -p deterministic:=true
```

Or switch via router:
```
ros2 run hri_safety_core arbiter_router --ros-args \\
  -p arbiter_mode:=rl \\
  -p policy_path:=/path/to/llmrl/hri_safety_ws/policies/ppo_policy.zip
```

Full pipeline launch:
```
ros2 launch hri_safety_core pipeline_full.launch.py parser_mode:=mock arbiter_mode:=rule
ros2 launch hri_safety_core pipeline_full.launch.py parser_mode:=qwen arbiter_mode:=rl \\
  policy_path:=policies/ppo_policy.zip
```
Quick no-training option (random policy):
```
ros2 launch hri_safety_core pipeline_full.launch.py parser_mode:=mock arbiter_mode:=rl \\
  policy_path:=random
```

Step 7: Symbolic env training + evaluation
Train:
```
cd /path/to/llmrl/training
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install stable-baselines3 gymnasium numpy tensorboard
python train_ppo.py --total-timesteps 50000
```

Evaluate:
```
python eval_policy.py --policy-path ../hri_safety_ws/policies/ppo_policy.zip --episodes 100 --out-csv eval_results.csv
```
The training script writes `ppo_policy.meta.json` alongside the policy.

Step 8: ROS2 evaluation runner (CSV)
Prepare an instruction list (one per line), then run:
```
python3 evaluation/ros_eval_runner.py \\
  --instructions instructions/base.txt \\
  --out-csv eval_results.csv
```

Matrix mode (2x2 parser/arbiter):
```
python3 evaluation/ros_eval_runner.py --matrix \\
  --episodes 12 \\
  --results-dir results
```
Outputs:
- `results/<timestamp>_<parser>_<arbiter>.csv`
- `results/summary.csv`
- `results/eval_meta.json`
Per-episode CSV fields include `task_success`, `safe_refusal`, `fallback`, `failure_reason`, and `fallback_reason`.
Summary adds `task_success_rate`, `safe_refusal_rate`, and `fallback_rate`.
Multi-seed outputs (under results/full_matrix):
- `seed_<k>/<cell_name>/{summary.csv,*.csv}`
- `seed_<k>/run_meta.json`
- `index.json`

Parser-only (mock+rule, qwen+rule):
```
python3 evaluation/ros_eval_runner.py --matrix-parser-only \\
  --episodes 12 \\
  --results-dir results
```
If you see `reset_reason=missing_service`, start the episode manager:
```
ros2 run hri_safety_core episode_manager_node
```
Simulated user + limits (key flags):
- `--max-turns-per-episode N` caps interactive turns per episode.
- `--max-repeat-action N` caps repeats of the same query action.
- `--clarify-left/--clarify-right/--clarify-default` set disambiguation replies.
- `--point-response` sets the reply for `ASK_POINT`.
- `--regression-test` runs 5 episodes and fails if clear instructions exceed 2 queries.
Fallback behavior:
- When repeated clarifications still fail or the policy is unsure, RL may output `FALLBACK_HUMAN_HELP`.
- The episode ends with a safe handoff utterance (no robot action).

One-command Gazebo matrix (wrapper):
```
./evaluation/run_matrix_gazebo.sh --episodes 12 --results-dir results
```
If the Gazebo GUI crashes, use headless mode:
```
./evaluation/run_matrix_gazebo.sh --episodes 12 --results-dir results --headless
```

Full 2x2 matrix (non-Gazebo, recommended first):
```
python3 evaluation/ros_eval_runner.py --matrix --episodes 12 --results-dir results \\
  --launch-file pipeline_full.launch.py --no-reset \\
  --policy-path hri_safety_ws/policies/ppo_policy.zip
```

Full 2x2 matrix (Gazebo headless):
```
./evaluation/run_matrix_gazebo.sh --episodes 12 --results-dir results --headless \\
  --policy-path hri_safety_ws/policies/ppo_policy.zip
```

Full 2x2 matrix (multi-seed, non-Gazebo, recommended):
```
./evaluation/run_matrix_full.sh --episodes 12 --results-dir results/full_matrix --seeds 0,1,2
python3 evaluation/make_report.py --results-dir results/full_matrix
```
Use existing policy for all seeds:
```
./evaluation/run_matrix_full.sh --episodes 12 --results-dir results/full_matrix --seeds 0,1,2 \\
  --policy-path hri_safety_ws/policies/ppo_policy.zip
python3 evaluation/make_report.py --results-dir results/full_matrix
```

Report generation (tables + plots):
```
python3 evaluation/make_report.py --results-dir results
```
Outputs:
- `report/summary.md`
- `report/metrics.csv`
- `report/metrics_aggregate.csv` (multi-seed)
- `report/*.png`
- `report/reasons.csv`
- `report/reasons.md`

Pipeline launch (router):
```
ros2 launch hri_safety_core pipeline_router.launch.py \\
  parser_mode:=mock \\
  arbiter_mode:=rl \\
  policy_path:=/path/to/llmrl/hri_safety_ws/policies/ppo_policy.zip
```

Notes
- This workspace uses isolated install by default. The `AMENT_PREFIX_PATH`
  export above is required unless you rebuild with `colcon build --merge-install`.

Step 9: Gazebo tabletop world + world state
World:
- `hri_safety_core/worlds/tabletop.sdf` (table + cup_red_1/cup_red_2/knife_1)

Launch Gazebo only (GUI):
```
ros2 launch hri_safety_core gazebo_only.launch.py
```

Launch pipeline (Gazebo + world state + parser/estimator/arbiter):
```
ros2 launch hri_safety_core pipeline_gazebo.launch.py \\
  parser_mode:=mock \\
  arbiter_mode:=rule
```
Headless (no GUI, stable on machines without working OpenGL):
```
ros2 launch hri_safety_core pipeline_gazebo.launch.py headless:=true \\
  parser_mode:=mock \\
  arbiter_mode:=rule
```

Check topics:
```
ros2 topic echo /scene/summary
ros2 topic echo /scene/state_json
```

Adjust the pose topic if needed:
```
ros2 launch hri_safety_core pipeline_gazebo.launch.py ign_topic:=/world/default/pose/info
```

You can auto-detect a pose topic:
```
ros2 launch hri_safety_core pipeline_gazebo.launch.py ign_topic:=auto
```

Step 10: Gazebo skill executor (Pick/Place/Handover)
Nodes:
- `action_to_skill_bridge` converts `/arbiter/action` + `/safety/features` to `/skill/command`
- `skill_executor_gazebo` applies pose changes and publishes `/skill/result`

Basic check:
```
ros2 topic echo /skill/command
ros2 topic echo /skill/result
```
Expected: EXECUTE action triggers a pick then place/handover sequence.

Step 11: Episode manager (reset + randomize)
Node:
- `episode_manager_node` exposes `/episode/reset` and publishes `/episode/metadata`

Manual reset:
```
ros2 service call /episode/reset std_srvs/srv/Trigger {}
```

Evaluation runner with reset (Gazebo launch):
```
python3 evaluation/ros_eval_runner.py --matrix \\
  --launch-file pipeline_gazebo.launch.py \\
  --episodes 50
```
Use `--no-reset` to skip resets if the service is unavailable.
