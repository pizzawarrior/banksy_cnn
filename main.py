import matplotlib.pyplot as plt
from src.data_loader import get_images
from experiments.run_experiment import run_experiment
from src.tf_setup import configure_tf
from src.split_data import split_data_train_test
from src.evaluate import evaluate_on_test_set
from src.utils import get_trained_model_paths, get_model_name, get_saved_metrics, save_metrics, sort_models
from src.get_metrics import show_conf_matrix


# TODO: add other IDEAL STATE functions


def get_best_model(dir_path='models/saved/fully_trained'):
    '''
    comb thru the models/saved/fully_trained metadata files and sort the
    metrics.
    default dir_path is set to fully trained models, but will also work
    for cv trained models too.
    return the path to the best model.
    this can be used to display the current best model information, or for
    loading the model and testing it on a new image.
    '''

    # test refers to whether we are using a tested model or not
    tested = True if dir_path == 'models/saved/fully_trained' else False

    metadata_dict = get_saved_metrics(dir_path, tested=tested)
    ranked_models = sort_models(metadata_dict, tested)  # retrieves top n models, in rank order
    best_model = list(ranked_models.items())[0]  # tuple: (model_path, dict_of_metrics)
    model_path = best_model[0]
    metrics = best_model[1]

    print(f'The best model is {get_model_name(model_path)}, with:')
    print('\n'.join(f'{k}: {v}' for k, v in metrics.items()))

    return model_path


def test_one_model(model_path, single_img=False, cm_plot_save_path='plots/confusion_matrices'):
    '''
    test one selected model against either the the full test dataset or a single image.
    we generate performance metrics and save them to the metadata file.
    NOTE: this works for either cv trained models or fully trained ones.
    returns:
        - test_metrics: a dict of accuracy, f1, precision, recall scores
        - converted image-level binary prediction(s), np.array
        - true labels for image(s), np.array
    if single image is True:
        - assumes image is already part of the `images` dataset.
        - also returns all tile-level predictions, np.array
    otherwise if testing on a full test set:
        - we generate and save a confusion matrix.
    '''
    image_list, labels = get_images()
    _, X_test, _, y_test = split_data_train_test(image_list, labels)

    if single_img is True:
        test_img = X_test[1]  # placeholder:: select your test image
        plt.imshow(test_img, cmap='gray')  # display image for reference
        plt.show()
        # test_metrics, image_pred_binary, image_true_labels, tile_predictions
        results = evaluate_on_test_set(X_test, y_test, model_path=model_path, single_img=True)
        test_metrics, image_pred_binary, image_true_labels, tile_predictions = results
        save_metrics(test_metrics, model_path)
        return test_metrics, image_pred_binary, image_true_labels, tile_predictions

    else:
        results = evaluate_on_test_set(X_test, y_test, model_path=model_path, single_img=False)
        test_metrics, image_pred_binary, image_true_labels = results
        file_path_model_name = (cm_plot_save_path, get_model_name(model_path))  # for saving cm plot
        show_conf_matrix(image_true_labels, image_pred_binary, file_path_model_name, save=True)
        save_metrics(test_metrics, model_path)
        return test_metrics, image_pred_binary, image_true_labels


# delete or move me
# this lives here for the moment to avoid circular dependencies
def test_best_models():
    '''
    get the paths for all fully traind models and test each one.
    Store their metrics and return various results.
    likely a one-time-use function.
    '''
    model_paths = get_trained_model_paths()
    for model_path in model_paths:
        test_one_model(model_path)
    return


def main():
    # Do not comment me out: this configures tf to access the local GPU
    configure_tf()

    ##############

    # model fitting and experimentation:
    run_experiment()

    ##############

    # single model testing; full test dataset:
    # model_path = models/saved/cv_results/cnn_200x200_overlap0.5_entropy2.0_3layer_fold4.keras - best model
    # _, image_pred_binary, image_true_labels = test_one_model(model_path, single_img=False)

    ##############

    # testing all saved models on the full test set
    # test_best_models()

    ##############

    # retrieve the best fully trained model
    # get_best_model(dir_path='models/saved/fully')


if __name__ == "__main__":
    main()
