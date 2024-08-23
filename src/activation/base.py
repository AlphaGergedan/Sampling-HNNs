from abc import ABC, abstractmethod
from numpy import ndarray

class BaseActivation(ABC):

    @staticmethod
    @abstractmethod
    def forward(input: ndarray) -> ndarray:
        pass

    @staticmethod
    @abstractmethod
    def grad(input: ndarray) -> ndarray:
        pass
