import numpy as np

from .base import BaseActivation

class Tanh(BaseActivation):

    @staticmethod
    def forward(input):
        return np.tanh(input)

    @staticmethod
    def grad(input):
        return 1 - np.tanh(input)**2
