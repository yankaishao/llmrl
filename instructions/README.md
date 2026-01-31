# Instructions

This folder contains fixed instruction lists for evaluation runs.

## base.txt
`base.txt` is the canonical 4-line set used by the matrix evaluation:
- clear low-risk (left cup)
- ambiguous (unspecified cup)
- risky (knife)
- unknown target (no object specified)

## Extending
- Create a new `.txt` file with one instruction per line (no blank lines).
- Keep instructions short and ASCII.
- Use the new file with:
  - `python3 evaluation/ros_eval_runner.py --instructions instructions/your_list.txt`
  - `./evaluation/run_matrix_gazebo.sh --instructions instructions/your_list.txt`
