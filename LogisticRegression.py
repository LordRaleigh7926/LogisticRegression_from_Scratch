import numpy as np

# The sigmoid function 
def sigmoid(x: float) -> float :

    """
    Compute the sigmoid function for a given input.

    Parameters:
        x (float): The input value.

    Returns:
        float: The sigmoid value of the input.

    """

    return 1/(1+np.exp(-x))

# Used to get the accuracy of the binary model in percentage
def Accuracy(y_true, y_pred) -> float:

    """
    Calculate the accuracy of the predicted values.

    Parameters:
    - y_true (np.ndarray): The true labels.
    - y_pred (np.ndarray): The predicted labels.

    Returns:
    - float: The accuracy of the predicted values, expressed as a percentage.

    Example:
        >> y_true = np.array([1, 0, 1, 1, 0])
        >> y_pred = np.array([1, 1, 0, 1, 0])
        >> accuracy = Accuracy(y_true, y_pred)
        >> print(accuracy)
        60.0

    Note:
        The accuracy is calculated as (1 - mean absolute error) * 100.
    """

    return 100 - np.mean(np.abs(y_pred-y_true))*100


class LogisticRegression:

    """
    LogisticRegression class implements logistic regression algorithm for binary classification.

    Methods:
        - train(x_train: np.ndarray, y_train: np.ndarray, alpha: float = 2.0, epoch: int = 10000, verbose: bool = False) -> None:
            Trains the logistic regression model using gradient descent optimization.
            Parameters:
                - x_train (np.ndarray): The training input data of shape (n_features, n_samples).
                - y_train (np.ndarray): The training target labels of shape (1, n_samples).
                - alpha (float): The learning rate for gradient descent. Default is 2.0.
                - epoch (int): The number of iterations for gradient descent. Default is 10000.
                - verbose (bool): Whether to print the training accuracy at each epoch. Default is False.

        - predict(x: np.ndarray) -> np.ndarray:
            Predicts the target labels for the given input data using the trained logistic regression model.
            Parameters:
                - x (np.ndarray): The input data of shape (n_features, n_samples).
            Returns:
                - np.ndarray: The predicted target labels of shape (1, n_samples).

    Attributes:
        - weights (np.ndarray): The learned weights of shape (n_features, 1).
        - bias (float): The learned bias term.

    Note:
        - The logistic regression model uses the sigmoid activation function for binary classification.
        - The accuracy of the model can be calculated using the Accuracy function provided in the code snippet.
    """

    # Public Methods
    __all__ = ["__init__", "train", "predict"]

    # Constructor
    def __init__(self) -> None:
        pass

    # Function which implements Gradient_Descent 
    def Gradient_Descent(self, x :np.ndarray, y : np.ndarray, alpha: float, epoch:int) -> dict:

        """
        Perform gradient descent optimization to train the logistic regression model.

        Parameters:
            - x (np.ndarray): The training input data of shape (n_features, n_samples).
            - y (np.ndarray): The training target labels of shape (1, n_samples).
            - alpha (float): The learning rate for gradient descent.
            - epoch (int): The number of iterations for gradient descent.

        Returns:
            - dict: A dictionary containing the learned weights and bias of the logistic regression model.

        Notes:
            - The logistic regression model uses the sigmoid activation function for binary classification.
            - The accuracy of the model can be calculated using the Accuracy function provided in the code snippet.
        """

        features = x.shape[0] # Number of features
        m = x.shape[1] # Number of datapoints

        w = np.zeros((features, 1)) # Shape of weights
        b = 0.0 # Initializing bias


        # Iterating through each epoch
        for _ in range(epoch):


            # Calculating wTx + b
            z = np.sum(np.dot(w.T, x), axis=0) + b

            # Getting y_pred (which is 'a') and reshaping it because by default it is a rank 1 array
            a = sigmoid(z).reshape(1,-1)

            # derivative of z in respect to Loss - dL/dz 
            dz = a - y

            # derivative of weights in respect to Loss - dL/dw 
            dw = (1/m)*(np.dot(x,dz.T)).sum(axis=1, keepdims=True)

            # derivative of bias in respect to Loss - dL/db
            db = (1/m)*(np.sum(dz))

            # Updating Bias and Weights
            w = w - alpha*dw
            b = b - alpha*db


        # Putting weights and bias in a dictionary for returing them
        params = {"weights" : w, "bias" : b}

        # Prints the Training Accuracy 
        print(f"\nTrain Accuracy = {Accuracy(y, a)}\n")

        return params

    # Used to train the model
    def train(self, x_train: np.ndarray, y_train: np.ndarray, alpha : float = 2.0, epoch : int = 100000, verbose: bool = False) -> None:

        """
        Trains the logistic regression model using gradient descent optimization.

        Parameters:
            - x_train (np.ndarray): The training input data of shape (n_features, n_samples).
            - y_train (np.ndarray): The training target labels of shape (1, n_samples).
            - alpha (float): The learning rate for gradient descent. Default is 2.0.
            - epoch (int): The number of iterations for gradient descent. Default is 10000.
            - verbose (bool): Whether to print the training accuracy at each epoch. Default is False.

        Returns:
            None

        Note:
            - The logistic regression model uses the sigmoid activation function for binary classification.
            - The accuracy of the model can be calculated using the Accuracy function provided in the code snippet.
        """

        # Transposes X
        x_train = x_train.T

        # Getting optimal weights and biases from the Gradient_Descent
        params = self.Gradient_Descent(x_train, y_train, alpha, epoch)

        # Storing them in class variables 
        self.weights = params["weights"]
        self.bias = params["bias"]


    # Used to predict test data
    def predict(self, x:np.ndarray) -> np.ndarray:

        """
        Predicts the target labels for the given input data using the trained logistic regression model.

        Parameters:
            - x (np.ndarray): The input data of shape (n_samples, n_features).

        Returns:
            - np.ndarray: The predicted target labels of shape (n_samples, ).

        Note:
            - The logistic regression model uses the sigmoid activation function for binary classification.
            - The predicted values are rounded off to either 0 or 1 based on a threshold of 0.5.
            - The original predictions (before rounding off) are stored in the 'prediction_original' attribute of the class.
        """

        # Transposing x
        x = x.T

        # Getting predicted values and reshaping them
        z = np.sum(np.dot(self.weights.T, x), axis=0) + self.bias
        a = sigmoid(z).reshape(1,-1)
        
        # Rounding off predicted values 
        y_pred = np.where(a[0] >= 0.5, 1, 0)

        # Storing original predictions
        self.prediction_original = a[0]

        # Returning the predicted values 
        return y_pred




