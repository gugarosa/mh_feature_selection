import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
import opfython.utils.converter as c


def load_dataset(dataset, val_split=0.25, test_split=0.2, seed=0):
    """
    """

    # Defining the path to the file
    input_path = f'data/{dataset}'

    # Converting from OPF to text
    c.opf2txt(input_path + '.dat', output_file=input_path + '.txt')

    # Loading a .txt file to a numpy array
    txt = l.load_txt(input_path + '.txt')

    # Parsing the pre-loaded numpy array
    X, Y = p.parse_loader(txt)

    # Splitting data into training and testing sets
    X_train, X_test, Y_train, Y_test = s.split(X, Y, percentage=1-test_split, random_state=seed)

    # Splitting training data into training and validation sets
    X_train, X_val, Y_train, Y_val = s.split(X_train, Y_train, percentage=1-val_split, random_state=seed)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test
