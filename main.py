import matplotlib.pyplot as plt
from src.data_loader import get_images
from experiments.run_experiment import run_experiment
from src.tf_setup import configure_tf
from src.split_data import split_data
from src.evaluate import evaluate_on_test_set
from src.train_full import train_full
from src.utils import get_model_name, get_saved_metrics, save_metrics, sort_models, filter_top_cv_models
from src.get_metrics import show_conf_matrix, show_training_accuracy_plots


def get_best_model(dir_path='models/saved/fully_trained'):
    '''
    comb thru the metadata files (from both cv_results and fully_trained) and sort thru the
    metrics to find the best model.
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


def train_test_one_model(X_train, X_test, y_train, y_test):
    '''
    this allows us to train one model on the full dataset, and test it.
    it is intended to be used for experimentation using the cv-trained model hyperparams as a reference.
    use this for fine tuning each of the params.
    '''
    hyperparams = {
        'tile_h': 200, 'tile_w': 200, 'overlap': 0.8, 'entropy_threshold': 1.8,
        'architecture': '3layer', 'learning_rate': 0.0001, 'batch_size': 32,
        'classification_threshold': .5
    }
    _, history, result_paths = train_full(X_train, y_train, hyperparams)
    model_path = result_paths[0]
    return test_one_model(X_test, y_test, model_path, single_img=False, history=history)


def test_one_model(X_test, y_test, model_path, single_img=False, history=None):
    '''
    test one selected model against either the the full test dataset or a single image.
    we generate performance metrics and save them to the metadata file.
    NOTE: this works for either cv trained models or fully trained ones.
    returns:
        - test_metrics: a dict of accuracy, f1, precision, recall scores
        - converted image-level binary prediction(s): np.array
        - true labels for image(s): np.array
    if single image is True:
        - assumes image is already part of the `images` dataset.
        - also returns all tile-level predictions: np.array
    otherwise if testing on a full test set:
        - we generate and save a confusion matrix.
    '''
    if single_img is True:
        test_img = X_test[1]  # placeholder: select your test image index
        plt.imshow(test_img, cmap='gray')  # display image for reference
        plt.show()
        results = evaluate_on_test_set(X_test, y_test, model_path=model_path, single_img=True)
        test_metrics, image_pred_binary, image_true_labels, tile_predictions = results
        save_metrics(test_metrics, model_path)
        return test_metrics, image_pred_binary, image_true_labels, tile_predictions

    else:
        results = evaluate_on_test_set(X_test, y_test, model_path=model_path, single_img=False)
        test_metrics, image_pred_binary, image_true_labels = results
        model_name = get_model_name(model_path)
        cm_path_model_name = ('plots/confusion_matrices', model_name)  # for saving cm plot
        acc_path_model_name = ('plots/accuracy_graphs', model_name)  # for saving cm plot
        show_training_accuracy_plots(history, acc_path_model_name)
        show_conf_matrix(image_true_labels, image_pred_binary, cm_path_model_name, save=True)
        save_metrics(test_metrics, model_path)
        return test_metrics, image_pred_binary, image_true_labels


def main():
    # Do not comment me out: this configures tf to access the local GPU
    configure_tf()

    ##############
    # get training and testing datasets
    # image_list, labels = get_images()
    # X_train, X_test, y_train, y_test = split_data(image_list, labels)

    ##############
    # building cv trained models from scratch
    # run_experiment()

    ##############
    # single model testing, full test dataset:
    # model_path = 'models/saved/cv_results/cnn_200x200_overlap0.5_entropy2.0_3layer_fold4.keras'  # - best model
    # _, image_pred_binary, image_true_labels = test_one_model(X_test, y_test, model_path)

    ##############
    # retrieve the best fully trained model
    # get_best_model()  # default is: dir_path='models/saved/fully_trained'

    ##############
    # Get current top n models from either fully_trained or cv_results
    # If cv_results then tested=False
    # metadata = get_saved_metrics('models/saved/fully_trained', tested=True)
    # ranked_models = sort_models(metadata, tested=True)
    # filter_top_cv_models(ranked_models, n=5)

    ##############
    # train/ test one model using hyperparams pasted into the fn below
    train_test_one_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
