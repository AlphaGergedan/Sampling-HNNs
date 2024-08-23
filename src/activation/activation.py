from .type import ActivationType

from .tanh import Tanh

class Activation():
    @staticmethod
    def new(activation: ActivationType):
        match activation:
            case ActivationType.TANH:
                return Tanh()
            case _:
                raise NotImplementedError("specified activation function is not implemented yet")
