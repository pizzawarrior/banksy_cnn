import matplotlib.pyplot as plt
from src.data_loader import get_images
# from experiments.run_experiment import run_experiment
from src.tf_setup import configure_tf
from src.split_data import split_data_train_test
from src.evaluate import evaluate_on_test_set
from src.utils import get_trained_model_paths, get_model_name
from src.get_metrics import show_conf_matrix


def get_best_model():
    '''
    comb thru the models/saved/fully_trained metadata files and sort the
    metrics. return the path to the best model.
    '''
    pass


def test_one_model(model_path, single_img=False, plot_save_path='plots/confusion_matrices'):
    '''
    TODO: can we add saving to this??? save the test results to the
    metadata file WITHOUT OVERIDING IT.
    test one selected model against either the the full test dataset or a single image.
    if single image: assumes image is already part of the `images` dataset.
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
    else:
        results = evaluate_on_test_set(X_test, y_test, model_path=model_path, single_img=False)
        test_metrics, image_pred_binary, image_true_labels = results

    # call save_metrics()

    file_path_model_name = (plot_save_path, get_model_name(model_path))
    show_conf_matrix(image_true_labels, image_pred_binary, file_path_model_name, save=True)


# delete or move me
def test_best_models():
    model_paths = get_trained_model_paths()
    for model_path in model_paths:
        test_one_model(model_path)
    return


# TODO: add other IDEAL STATE functions


    ## leftovers from prior testing
    # for i, model_path in enumerate(metadata_list, 1):
    #     print(f'\n{"*"*50}')
    #     print(f'Now testing model {i} of {len(metadata_list)}')

    #     # TODO: this is where we call train_full()
    #     test_metrics, image_pred_binary, image_true_labels = evaluate_on_test_set(
    #         X_test,
    #         y_test,
    #         model_path=model_path,
    #         single_img=False)

    #     results.append(
    #         {
    #             'model': model_path,
    #             'test_metrics': test_metrics,
    #             'image_pred_binary': image_pred_binary,
    #             'image_true_labels': image_true_labels
    #         }
    #     )

    # ranked_models = sorted(
    #     results,
    #     key=lambda model: (
    #         model['test_metrics']['img_precision'],
    #         model['test_metrics']['img_f1'],
    #         model['test_metrics']['img_accuracy']),
    #     reverse=True
    # )

    # best_model = ranked_models[0]
    # print(f'The best model is {best_model["model"]}, with:\n')
    # print('\n'.join(f'{k}: {v}' for k, v in best_model['test_metrics'].items()))
    # show_conf_matrix(best_model['image_true_labels'], best_model['image_pred_binary'])
    # return results


def main():
    # Do not comment me out: this configues tf to access the local GPU
    configure_tf()

    # model fitting and experimentation:
    # run_experiment()

    # TODO: update this mess
    # single model testing; full test dataset
    # # model_path = 'models/saved/cnn_200x200_overlap0.8_entropy1.0_5layer_fold3.keras'  # good model
    # model_path = 'models/saved/cnn_250x250_overlap0.8_entropy1.0_3layer_fold2.keras'  # test (delete)
    # _, image_pred_binary, image_true_labels = test_one_model(model_path, single_img=False)
    # show_conf_matrix(image_true_labels, image_pred_binary)

    # testing all saved models on the full test set


if __name__ == "__main__":
    main()
