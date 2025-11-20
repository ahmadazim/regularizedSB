import argparse
from src.regularizedSB.config import load_config
from src.regularizedSB.solver import solve_from_config


def parse_args():
    ap = argparse.ArgumentParser(description="Regularized SB runner (config-driven).")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    result = solve_from_config(cfg)
    # Print a brief summary to stdout
    mean = result["mean"]
    std = result["std"]
    print(f"Final mean: {mean}")
    print(f"Final std:  {std}")


if __name__ == "__main__":
    main()


