from sklearn.metrics import accuracy_score
from transfer_functions import T1


def feature_selection(obj, X_train, Y_train, X_val, Y_val):
    """Wraps the feature selection task for optimization purposes.

    Args:
        obj (Classifier): A classifier instance.
        X_train (np.array): An array of training data.
        Y_train (np.array): An array of training labels.
        X_val (np.array): An array of validation data.
        Y_val (np.array): An array of validation labels.

    Returns:
        The wrapped function itself.
        
    """

    def f(w):
        """Fits the classifier and compute its accuracy over the validation set.

        Args:
            w (float): Array of variables.

        Returns:
            1 - accuracy as the objective function.

        """

        # Gathering `w` as a vector of input features
        x = w[:, 0]

        # Passing down to the transfer function
        features = T1(x)

        # Remaking training and validation sets with selected features
        X_train_selected = X_train[:, features]
        X_val_selected = X_val[:, features]

        # Creates the classifier itself
        clf = obj()

        # Fits training data into the classifier
        clf.fit(X_train_selected, Y_train)

        # Predicts new data
        preds = clf.predict(X_val_selected)

        # Calculating accuracy
        acc = accuracy_score(Y_val, preds)

        return 1 - acc

    return f
