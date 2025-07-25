from src.data_loader import get_images
from experiments.run_experiment import run_experiment
from src.tf_setup import configure_tf
from src.split_data import split_data_train_test
from src.evaluate import evaluate_on_test_set


def main():
    configure_tf()

    # general model fitting:
    run_experiment()

    # model_path = "models/saved/cnn_200x200_overlap0.8_entropy1.0_5layer_fold3.keras"
    # model_path = "models/saved/cnn_200x200_overlap0.5_entropy2.0_3layer_fold2.keras"
    # image_list, labels = get_images()
    # _, X_test, _, y_test = split_data_train_test(image_list, labels)
    # evaluate_on_test_set(X_test, y_test, model_path=model_path)


if __name__ == "__main__":
    main()
