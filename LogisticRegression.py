import torch
from sklearn.metrics import accuracy_score
from torch import Tensor
import random

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class LogisticRegression:
    def __init__(self, X_train, y_train):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.weight: Tensor = torch.zeros(y_train.shape[1], X_train.shape[1], requires_grad=True).to(device)
        # print(f'Initial shape of weight vector is {self.weight.size()}')
        self.X_train = X_train
        self.y_train = y_train

    @staticmethod
    def logistic(z: Tensor) -> Tensor:
        return 1 / (1 + torch.exp(-z))

    @staticmethod
    def logistic_reg_model(X: Tensor, w: Tensor) -> Tensor:
        return LogisticRegression.logistic(X @ w.t())

    @staticmethod
    def binary_cross_entropy(predicted: Tensor, true: Tensor) -> Tensor:
        BCE = -(true * predicted.log() + (1 - true) * (1 - predicted).log()).mean()
        return BCE

    def forward_pass(self, X=None, y=None) -> Tensor:
        if X is None or y is None:
            X = self.X_train
            y = self.y_train
        # print(f"In logistic regression, before applying logistic function X size is {X.size()}")
        predictions = LogisticRegression.logistic_reg_model(X, self.weight)
        loss = LogisticRegression.binary_cross_entropy(predictions, y)
        return loss

    def predict(self, X) -> Tensor:
        with torch.no_grad():
            wet_predictions = LogisticRegression.logistic_reg_model(X, self.weight)
            return torch.round(wet_predictions)


class History:
    def __init__(self):
        self.accuracy_history = []
        self.loss_history = []
        self.local_accuracy_history = []


class Division:
    def __init__(self, amount, prob):
        self.amount = amount
        self.prob = prob

    def __str__(self):
        return f"Division({self.amount}, {self.prob})"


class L2GDNode:
    def __init__(self, X: Tensor, y: Tensor) -> None:
        self.X = X
        self.y = y
        assert self.X.size()[0] == self.y.size()[0]
        items = self.X.size()[0]
        train_items = int(items * 0.9)
        self.X_train = self.X[:train_items, :]
        self.y_train = self.y[:train_items, :]
        self.X_test = self.X[train_items:, :]
        self.y_test = self.y[train_items:, :]
        del self.X, self.y, items, train_items
        self.model: LogisticRegression = LogisticRegression(self.X_train, self.y_train)
        self.history = History()

    def get_accuracy(self, X: Tensor = None, y: Tensor = None) -> float:
        if X is None or y is None:
            X = self.X_test
            y = self.y_test
        return accuracy_score(
            self.model.predict(X).to("cpu"),
            y.to("cpu")
        )

    @staticmethod
    def createUniformly(X, y, i, n):
        batch = X.size()[0] // n
        return L2GDNode(
            X[i * batch:(i + 1) * batch, :],
            y[i * batch:(i + 1) * batch, :]
        )

    @staticmethod
    def createFromDivision(X, y, division: Division):
        idx = []
        for i in range(division.amount):
            sample = 1.0 if random.random() < division.prob else 0.0
            while True:
                index = random.randint(0, y.size()[0] - 1)
                if y[index].item() == sample:
                    idx.append(index)
                    break
        return L2GDNode(X[idx], y[idx])


class L2SGD_plus_Node:
    def __init__(self, X: Tensor, y: Tensor, m: int) -> None:
        self.X = X
        self.y = y
        assert X.size()[0] == y.size()[0]
        items = self.X.size()[0]
        train_items = int(items * 0.9)
        self.X_train = self.X[:train_items, :]
        self.y_train = self.y[:train_items, :]
        self.X_test = self.X[train_items:, :]
        self.y_test = self.y[train_items:, :]
        del self.X, self.y, items, train_items

        self.model: LogisticRegression = LogisticRegression(self.X_train, self.y_train)
        self.d = self.X_train.size()[1]
        self.m = m
        self.data_size = self.model.X_train.size()[0]
        self.minibatch_size = self.data_size // self.m
        self.J = torch.zeros((self.d, m)).to(device)
        self.psi = torch.zeros(self.d).to(device)
        self.history = History()

    def get_minibatch(self, j: int) -> tuple:
        assert self.minibatch_size * j is int
        minibatch = (
            self.X_train[self.minibatch_size * j: self.minibatch_size * (j + 1), :],
            self.y_train[self.minibatch_size * j: self.minibatch_size * (j + 1)]
        )
        # print(f"While getting minibatch, the size of minibatch X was {mini-batches[0].size()}")
        return minibatch

    @staticmethod
    def createFromDivision(X, y, m, division: Division):
        idx = []
        for i in range(division.amount):
            sample = 1.0 if random.random() < division.prob else 0.0
            while True:
                index = random.randint(0, y.size()[0] - 1)
                if y[index].item() == sample:
                    idx.append(index)
                    break
        return L2SGD_plus_Node(X[idx], y[idx], m)

    def get_accuracy(self, X: Tensor = None, y: Tensor = None) -> float:
        if X is None or y is None:
            X = self.X_test
            y = self.y_test
        return accuracy_score(
            self.model.predict(X).to("cpu"),
            y.to("cpu")
        )
