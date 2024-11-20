import torch
from torch.optim.adam import Adam

from util import DeviceType
from data.dataloader import get_batch

from .base import BaseTrainer

class TraditionalTrainer(BaseTrainer):
    def __init__(self, total_steps, batch_size, learning_rate, weight_decay, device, **kwargs):
        super(TraditionalTrainer, self).__init__()

        self.total_steps = total_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.training_device = device

    def train(self, model, train_inputs, train_dt_truths, train_input_x_0, train_input_x_0_H_truth, device=DeviceType.GPU, dtype=torch.float64, train_H_truths=None):
        super(TraditionalTrainer, self).train(model, train_inputs, train_dt_truths, train_input_x_0, train_input_x_0_H_truth, device, dtype, train_H_truths)
        assert train_H_truths is None # only used for supervised SWIM in S-HNN

        from model.mlp import MLP
        from model.hnn import HNN

        assert isinstance(model, MLP) or isinstance(model, HNN)

        # model training mode
        model.train()

        # move the model to the device
        if device is DeviceType.GPU:
            model = model.cuda()
        else:
            model = model.cpu()

        optim = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        train_losses = []

        print("Step No. : Loss (Squared L2 Error)")
        for step in range(self.total_steps):
            # place inputs to the same device as the model
            x = get_batch(train_inputs, step, self.batch_size, requires_grad=True, dtype=dtype, device=self.training_device)
            dt_true = get_batch(train_dt_truths, step, self.batch_size, requires_grad=False, dtype=dtype, device=self.training_device)
            loss = self.__step(step, model, optim, x, dt_true)
            train_losses.append(loss)

        # followings are for fixing integration constants for HNNs
        if isinstance(model, HNN):
            x = get_batch(train_input_x_0, 0, 1, requires_grad=False, dtype=dtype, device=self.training_device)
            H_true = get_batch(train_input_x_0_H_truth, 0, 1, requires_grad=False, dtype=dtype, device=self.training_device)
            self.__fit_H_last_layer_bias(model, x, H_true)

        # move the model to cpu
        model.cpu()

        return train_losses

    def __step(self, step, model, optim, x, dt_true):

        # compute the loss
        dt_pred = model.forward(x) # forward automatically creates computation graph when computing dxdt
        loss = (dt_true - dt_pred).pow(2).mean() # least squares optimization

        # backward + optimize
        loss.backward(retain_graph=False); optim.step(); optim.zero_grad()

        if step % 1000 == 0:
            print(f"-> Loss at step {step}\t:\t{loss}")

        return loss.item()

    def __fit_H_last_layer_bias(self, model, x, H_true):
        H_pred = model.H(x)
        bias = H_true - (H_pred - model.H_last_layer().bias)
        model.H_last_layer().bias = torch.nn.Parameter(bias)
