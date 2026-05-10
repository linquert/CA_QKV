"""Minimal Colab runner helper.

Usage in Colab after uploading/cloning this folder:

    !pip install -r requirements.txt
    import wandb; wandb.login()  # optional
    !python run_experiment.py --config configs/base_left_copy.yaml

For a fast no-W&B smoke test:

    !python run_experiment.py --config configs/debug_left_copy.yaml
"""
