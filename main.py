# from src.data_loader import get_images
from experiments.run_experiment import run_experiment
from src.tf_setup import configure_tf


def main():
    configure_tf()
    results = run_experiment()
    return results


if __name__ == "__main__":
    main()
