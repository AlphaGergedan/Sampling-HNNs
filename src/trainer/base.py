from abc import ABC, abstractmethod

from util import DeviceType


class BaseTrainer(ABC):
    @abstractmethod
    def train(self, model, train_inputs, train_dt_truths, train_input_x_0, train_input_x_0_H_truth, device: DeviceType, train_H_truths=None):
        """

        @param model                    : model to be trained
        @param train_inputs             : training inputs, i.e. (q, p)
        @param train_dt_truths          : time derivatives of the training inputs, i.e. (q_dot, p_dot)
        @param train_input_x_0          : known point where we know the true function value, i.e. (q_0, p_0)
        @param train_input_x_0_H_truth  : the Hamiltonian function value at train_input_x_0
        @param train_H_truths           : only used for the supervised SWIM method in S-HNN
        """
        _, input_dim = train_inputs.shape
        assert train_inputs.shape == train_dt_truths.shape
        assert train_input_x_0.shape == (1, input_dim)
        assert train_input_x_0_H_truth.shape == (1, 1)
