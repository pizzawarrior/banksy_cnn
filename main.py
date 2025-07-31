import os
import matplotlib.pyplot as plt
from src.data_loader import get_images
from experiments.run_experiment import run_experiment
from src.tf_setup import configure_tf
from src.split_data import split_data_train_test
from src.evaluate import evaluate_on_test_set
from src.get_metrics import show_conf_matrix


# def test_one_model(model_path, single_img=False):
def test_one_model(model_path, single_img=False):
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
        return evaluate_on_test_set(X_test, y_test, model_path=model_path, single_img=True)
    else:
        return evaluate_on_test_set(X_test, y_test, model_path=model_path, single_img=False)


def test_all_models():
    '''
    NOTE we only want 1 conf matrix for the best model
    load each .keras or .h5 model and test it against the full dataset
    save precision and f1 scores to a dict in the following format:
    search thru the results dict, locate the best model by precision then f1 score, then accuracy,
    and print the name of the best model
    '''
    image_list, labels = get_images()
    _, X_test, _, y_test = split_data_train_test(image_list, labels)

    models_list = []

    # relative path of saved models
    dir_path = 'models/saved'
    for dir_path, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith('.keras'):
                models_list.append(os.path.join(dir_path, filename))

    results = []

    if not models_list:
        return f'There are no models in the current directory: {dir_path}'

    for i, model_path in enumerate(models_list, 1):
        print(f'\n{"*"*50}')
        print(f'Now testing model {i} of {len(models_list)}')
        test_metrics, image_pred_binary, image_true_labels = evaluate_on_test_set(
            X_test,
            y_test,
            model_path=model_path,
            single_img=False)

        results.append(
            {
                'model': model_path,
                'test_metrics': test_metrics,
                'image_pred_binary': image_pred_binary,
                'image_true_labels': image_true_labels
            }
        )

    ranked_models = sorted(
        results,
        key=lambda model: (
            model['test_metrics']['img_precision'],
            model['test_metrics']['img_f1'],
            model['test_metrics']['img_accuracy']),
        reverse=True
    )

    best_model = ranked_models[0]
    print(f'The best model is {best_model["model"]}, with:\n')
    print('\n'.join(f'{k}: {v}' for k, v in best_model['test_metrics'].items()))
    show_conf_matrix(best_model['image_true_labels'], best_model['image_pred_binary'])
    return results


def main():
    # Do not comment me out
    configure_tf()

    # model fitting and experimentation:
    # run_experiment()

    # single model testing; full test dataset
    # # model_path = 'models/saved/cnn_200x200_overlap0.8_entropy1.0_5layer_fold3.keras'  # good model
    # model_path = 'models/saved/cnn_250x250_overlap0.8_entropy1.0_3layer_fold2.keras'  # test (delete)
    # _, image_pred_binary, image_true_labels = test_one_model(model_path, single_img=False)
    # show_conf_matrix(image_true_labels, image_pred_binary)

    # testing all saved models on the full test set
    test_all_models()


if __name__ == "__main__":
    main()
