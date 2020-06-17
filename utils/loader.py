from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_dataset():
    """
    """

    #
    X, Y = load_iris(return_X_y=True)

    #
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.95, random_state=42)

    return X_train, X_val, Y_train, Y_val