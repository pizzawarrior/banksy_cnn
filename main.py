import matplotlib.pyplot as plt
from src.data_loader import get_images
from experiments.run_experiment import run_experiment
from src.tf_setup import configure_tf
from src.split_data import split_data_train_test
from src.evaluate import evaluate_on_test_set


# def test_one_model(model_path='models/saved/cnn_200x200_overlap0.8_entropy1.0_5layer_fold3.keras', single_img=False):
def test_one_model(model_path='models/saved/cnn_200x200_overlap0.5_entropy1.5_5layer_fold4.keras', single_img=False):
    '''
    test one selected model against either the the full test dataset or a single image.
    if single image: assumes image is already part of the `images` dataset.
    '''
    image_list, labels = get_images()
    _, X_test, _, y_test = split_data_train_test(image_list, labels)

    if single_img is True:
        # select your test image
        test_img = X_test[1]
        # display image for reference
        plt.imshow(test_img, cmap='gray')
        plt.show()
        evaluate_on_test_set(X_test, y_test, model_path=model_path, single_img=True)
    else:
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

    # model testing; full test dataset
    test_one_model(single_img=False)


if __name__ == "__main__":
    main()
