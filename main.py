from src.data_loader import get_images
from experiments.run_experiment import run_experiment
from src.tf_setup import configure_tf
from src.split_data import split_data_train_test
from src.evaluate import evaluate_on_test_set


def test_one_model(model_path='models/saved/cnn_200x200_overlap0.8_entropy1.0_5layer_fold3.keras'):
    '''
    test one selected model against the full test dataset
    '''
    image_list, labels = get_images()
    _, X_test, _, y_test = split_data_train_test(image_list, labels)
    evaluate_on_test_set(X_test, y_test, model_path=model_path, single_img=False)


def test_all_models():
    '''
    NOTE we only want 1 conf matrix for the best model
    load each .keras or .h5 model and test it against the full dataset
    save precision and f1 scores to a dict in the following format:
    results = {
        'model_name': {
            'precision': precision,
            'f1': f1
        },...
    }
    search thru the results dict, locate the best model by precision then f1 score, then accuracy,
    and print the name of the best model
    '''
    pass


def main():
    # Do not comment me out
    configure_tf()

    # model fitting and experimentation:
    # run_experiment()

    # model testing
    test_one_model()


if __name__ == "__main__":
    main()
