from sklearn.metrics import (accuracy_score,
                             f1_score,
                             precision_score,
                             recall_score,
                             roc_auc_score
                             )


def calc_metrics(y_test, y_pred, y_pred_proba, data_object):
    '''
    calculate comprehensive classification metrics
    data_object == 'tile' or 'img'
    '''

    if data_object not in ['img', 'tile']:
        raise ValueError('Data object for calculating performance metrics must be either `tile` or `img`.')

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        # 'auc': roc_auc_score(y_test, y_pred_proba)  # TODO: uncomment me for training!!!!
    }

    return {data_object + '_' + k: v for k, v in metrics.items()}
