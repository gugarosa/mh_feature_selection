from opfython.models.supervised import SupervisedOPF
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


class Classifier:
    """A Classifier class to help users in selecting distinct classifiers from the command line.

    """

    def __init__(self, obj):
        """Initialization method.

        Args:
            obj (BaseClassifier | OPF): A BaseClassifier or OPF child instance.

        """

        # Creates a property to hold the class itself
        self.obj = obj


# Defines a classifier dictionary constant with the possible values
CLF = dict(
    dt=Classifier(DecisionTreeClassifier()),
    linear_svc=Classifier(LinearSVC()),
    lr=Classifier(LogisticRegression()),
    nb=Classifier(GaussianNB()),
    opf=Classifier(SupervisedOPF()),
    rf=Classifier(RandomForestClassifier()),
    svc=Classifier(SVC()),
)


def get_clf(name):
    """Gets a classifier by its identifier.

    Args:
        name (str): Classfier's identifier.

    Returns:
        An instance of the Classifier class.

    """

    # Tries to invoke the method
    try:
        # Returns the corresponding object
        return CLF[name]

    # If object is not found
    except:
        # Raises a RuntimeError
        raise RuntimeError(f'Classifier {name} has not been specified yet.')
