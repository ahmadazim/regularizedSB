# Regularized Schrödinger Bridge

## Quickstart
1. Prepare a dataset that contains source and target particles (NumPy `.npz/.npy`). For a toy Gaussian shift example:
   ```
   python scripts/generate_toy_data.py --output data/toy_dataset.npz
   ```
   This saves `source`/`target` arrays you can reference in configs.
2. Edit `configs/toy_dGauss.yaml` (or create a new config) to point `dataset.source_path`/`dataset.target_path` to your data and adjust solver/penalty settings.
3. From the project root run:
   ```
   python main.py --config configs/toy_dGauss.yaml
   ```
   Outputs (models, summaries, samples) are written to `logging.output_dir/experiment_name`.

## SLURM
Submit the same run on a GPU node:
```
sbatch scripts/slurm_train.sh configs/toy_dGauss.yaml
```
Logs will appear in `logs/`, and the script tees stdout to a config‑specific logfile.
