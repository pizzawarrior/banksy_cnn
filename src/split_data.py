from sklearn.model_selection import train_test_split


def split_data_train_test(image_list, labels):
    '''
    takes the length of the full dataset and return random
    indices for training and testing sets.
    ~80% train, 20% test = 44 train, 12 test
    '''
    X_train, X_test, y_train, y_test = train_test_split(image_list, labels, test_size=0.2, stratify=labels, random_state=420)
    print(f'X_train has length: {len(X_train)}')
    print(f'y_train has length: {len(y_train)}')
    print(f'X_test has length: {len(X_test)}')
    print(f'y_test has length: {len(y_test)}')
    return X_train, X_test, y_train, y_test
