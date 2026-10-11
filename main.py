from pathlib import Path

from src.regression import run_regression_experiment


TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")
OUTPUT_DIR = Path("outputs")


def main():
    run_regression_experiment(TRAIN_PATH, TEST_PATH, OUTPUT_DIR)


if __name__ == "__main__":
    main()
