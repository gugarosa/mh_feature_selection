from sklearn.metrics import accuracy_score


def bool_feature_selection(obj, X_train, Y_train, X_val, Y_val):
    """Wraps the boolean-based feature selection task for optimization purposes.

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
        features = w[:, 0].astype(bool)
        
        # Checking if array is empty
        if not any(features):
            # If yes, penalizes the objective function
            return 1

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


def feature_selection(obj, transfer, X_train, Y_train, X_val, Y_val):
    """Wraps the feature selection task for optimization purposes.

    Args:
        obj (Classifier): A classifier instance.
        transfer (callable): A transfer function used to map continuous to binary data.
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
        features = transfer(x)

        # Checking if array is empty
        if not any(features):
            # If yes, penalizes the objective function
            return 1

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
