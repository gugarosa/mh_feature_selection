import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s


def load_dataset(dataset, val_split=0.25, test_split=0.2, seed=0):
    """Loads a dataset and splits into training, validation and testing sets.

    Args:
        dataset (str): String corresponding to the dataset to be loaded.
        val_split (float): Validation set percentage (applied after train/test split).
        test_split (test_split): Testing set percentage.
        seed (int): Random seed.

    Returns:
        Training, validation and testing sets.
        
    """

    # Defining the path to the file
    input_path = f'data/{dataset}'

    # Loading a .txt file to a numpy array
    txt = l.load_txt(input_path + '.txt')

    # Parsing the pre-loaded numpy array
    X, Y = p.parse_loader(txt)

    # Splitting data into training and testing sets
    X_train, X_test, Y_train, Y_test = s.split(X, Y, percentage=1-test_split, random_state=seed)

    # Splitting training data into training and validation sets
    X_train, X_val, Y_train, Y_val = s.split(X_train, Y_train, percentage=1-val_split, random_state=seed)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test
