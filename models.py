import torch
import numpy as np
from STensor import STensor


class LogisticRegressionModel:

    """
    For outer optimizer working cycle should be:
    For k in 1.....N:
        y_pred = [model.predict(x_i) for random x_i in X]
        loss = model.calculateLoss(y_pred, y_true)
        dL_dW, dL_db = model.calculateGradients(X, y_pred=y_pred, y_true=y_true)

        ### Compute the update for the weights

        model.scaleParameters(scale_factor)  (optionally)
        model.updateParameters(delta_W, delta_B)
    """
    def __init__(self, params_amount: int):
        """
        :param params_amount: Integer number represents how many parameters does the model store (except the bias)
        """
        self.weights = STensor(np.zeros(params_amount))
        self.bias = STensor([0])
        self.params_amount = params_amount

    def calculateZ(self, X: STensor) -> STensor:
        """
        Function Calculates Z = W * X + b
        :param X: Tensor with the arguments
        :return: Tensor Z, value, that must be put into the sigmoid function
        """
        return self.weights @ X.T + self.bias

    def predict(self, x: STensor) -> STensor:
        """
        y^ = sigma(z) = sigma(W^T X + b)
        :param x: Tensor with the arguments
        :return: y^, Tensor with the value (value of probability of entity with X being class 1)
        """
        z = self.calculateZ(x)
        if z.item() >= 0:
            return STensor(1 / (1 + torch.exp(-1 * z)))
        return torch.exp(z) / (1 + torch.exp(z))

    def calculateLoss(self, y_pred: STensor, y_true: STensor) -> STensor:
        """
        :param y_true:
        :param y_pred:
        :return:
        """
        y_zero_loss = torch.dot(y_true, torch.log(y_pred + 1e-9))
        y_one_loss = torch.dot(
            STensor(np.ones(self.params_amount) - y_true),
            torch.log(STensor(np.ones(self.params_amount)) - y_pred)
        )
        return -1 * (y_zero_loss + y_one_loss) / self.params_amount

    @staticmethod
    def calculateGradients(X: STensor, y_pred: STensor, y_true: STensor) -> tuple[STensor, STensor]:
        """
        :param X:
        :param y_pred:
        :param y_true:
        :return:
        """

        difference = y_pred - y_true
        dL_db = STensor(torch.mean(difference))
        JW = X.T @ difference
        dL_dW = STensor([torch.mean(grad) for grad in JW])   # VERY UNCERTAIN ABOUT THIS IN MATHEMATICAL SENSE
        return dL_dW, dL_db

    def updateParameters(self, deltaW: STensor, deltaB: STensor) -> None:
        self.weights = self.weights + deltaW
        self.bias = self.bias + deltaB

    def scaleParameters(self, scale_factor: float) -> None:
        self.weights = STensor(scale_factor * self.weights)
        self.bias = STensor(scale_factor * self.bias)

    def predictClass(self, X: STensor) -> int:
        return 1 if self.predict(X).item() >= 0.5 else 0
