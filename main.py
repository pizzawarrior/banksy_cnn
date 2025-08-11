import os
import json
import matplotlib.pyplot as plt
from src.data_loader import get_images
from src.get_metadata import get_metadata
from experiments.run_experiment import run_experiment
from src.tf_setup import configure_tf
from src.split_data import split_data_train_test
from src.evaluate import evaluate_on_test_set
from src.get_metrics import show_conf_matrix


# def test_one_model(model_path, single_img=False):
def test_one_model(model_path, single_img=False):
    '''
    TODO: confirm this works with a fully traind model.
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


def train_top_models():
    '''
    read the metadata files and parse the performance metrics of each cv model.
    create a new dict with the path to the model as the key and {precision, f1, accuracy}
    as the value. sort these metrics in descending order to isolate the top n models.
    ensure that each n model is unique (recall that 4 models with the same config are saved), we only
    want to add one of those to the top n list.
    train the top n models, save the metrics.
    '''

    def _get_metrics(dir_path='models/saved/cv_results'):
        metadata_dict = {}

        # find all metadata paths, parse the files, and add the metrics to a dict
        for dir_path, _, filenames in os.walk(dir_path):
            for filename in filenames:
                if filename.endswith('metadata.json'):
                    file_path = os.path.join(dir_path, filename)
                    metadata_dict[file_path] = get_metadata(file_path)
        if not metadata_dict:
            return f'There are no metadata files in the current directory: {dir_path}'
        return metadata_dict

    def _get_top_n_models(metadata_dict, n=5):
        ranked_models = dict(sorted(
            metadata_dict.items(),
            key=lambda model: (
                model[1]['precision'],
                model[1]['f1'],
                model[1]['accuracy']),
            reverse=True
        ))

        unique_models = {}
        seen_configs = set()
        i = 0
        print(f'The top {n} models, in order, are:')
        for k, v in ranked_models.items():
            trimmed_name = k[:-15]  # '#_metadata.json' = 15 chars
            if trimmed_name not in seen_configs:
                seen_configs.add(trimmed_name)
                unique_models[k] = v
                print(k[24:])  # this leaves out the path: 'models/saved/cv_results/'
                i += 1
            if i == n:
                break
        return unique_models

    def _get_metadata_hyperparams(unique_models):
        '''
        NOTE: we read from the metadata file so we can delete all cv models!
        '''
        hyperparams = []
        for path in list(unique_models.keys()):
            with open(path, 'r') as f:
                data = json.load(f)
            hyperparams.append(data['hyperparameters'])
        return hyperparams

    # image_list, labels = get_images()
    # X_train, _, y_train, _ = split_data_train_test(image_list, labels)
    metadata_dict = _get_metrics()
    top_n_models = _get_top_n_models(metadata_dict)
    hyperparameters = _get_metadata_hyperparams(top_n_models)

    # Ok, now we can fully train the top 5 models and record the performance


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
    # configure_tf()

    # model fitting and experimentation:
    # run_experiment()

    # single model testing; full test dataset
    # # model_path = 'models/saved/cnn_200x200_overlap0.8_entropy1.0_5layer_fold3.keras'  # good model
    # model_path = 'models/saved/cnn_250x250_overlap0.8_entropy1.0_3layer_fold2.keras'  # test (delete)
    # _, image_pred_binary, image_true_labels = test_one_model(model_path, single_img=False)
    # show_conf_matrix(image_true_labels, image_pred_binary)

    # testing all saved models on the full test set
    train_top_models()


if __name__ == "__main__":
    main()
