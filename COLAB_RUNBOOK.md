# FoodOps REINFORCE Colab Runbook

1. Open Colab and switch runtime to `T4 GPU`.
2. Clone the repo:
   ```bash
   !git clone https://github.com/preeyankajdp-glitch/dataops_incident_lab.git
   %cd dataops_incident_lab
   ```
3. Install dependencies:
   ```bash
   !pip install -q torch transformers peft accelerate matplotlib
   !pip install -e ./foodops_env
   ```
4. Optional HF token:
   ```bash
   %env HF_TOKEN=your_token_here
   ```
5. Run the smoke test:
   ```bash
   !python train_foodops.py --smoke --steps 3 --output-dir ./smoke_output
   ```
6. Launch real training:
   ```bash
   !python train_foodops.py --steps 150 --seed 7 --output-dir ./training_output
   ```
7. Expected runtime on a T4:
   - smoke: ~3-6 minutes
   - full run: ~45-90 minutes
8. Final artifacts land in `./training_output/`:
   - `training_metrics.csv`
   - `reward_curve.png`
   - `before_after_bars.png`
   - `confusion_matrix.png`
   - `eval_report.json`
   - `run_summary.md`
9. If memory gets tight, reduce the step count first:
   ```bash
   !python train_foodops.py --steps 50 --output-dir ./training_output_small
   ```
