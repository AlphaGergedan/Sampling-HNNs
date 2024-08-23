from torchinfo import summary

from hamiltonian.base import BaseHamiltonian
from model.base import BaseModel

from model.mlp import MLP
from model.s_mlp import S_MLP

from model.hnn import HNN
from model.s_hnn import S_HNN


def print_model_summary(model: BaseModel, target: BaseHamiltonian, kwargs):
    print(f"=================================================================")
    print(f"==                                                             ==")
    print(f"== Target:                                                       ")
    print(target)
    print(f"==                                                             ==")
    print(f"== Model Summary:                                                ")
    if isinstance(model, MLP) or isinstance(model, HNN):
        summary(model)
    elif isinstance(model, S_MLP):
        print(model.pipeline)
    elif isinstance(model, S_HNN):
        print(model.mlp.pipeline)
    else:
        raise ValueError(f"Model of type {type(model)} not supported yet")
    print(f"==                                                             ==")
    print(f"== kwargs:                                                       ")
    print(kwargs)
    print(f"==                                                             ==")
    print(f"=================================================================")

# Some models may not support all errors (e.g. plain MLP does not support Hamiltonian recovery),
# that's why None should be handled here
# ( error_dt, error_H, error_H_grad )
Errors = tuple[ float|None, float|None, float|None ]

def print_errors(train_errors: Errors, test_errors: Errors):
    train_error_dt, train_error_H, train_error_H_grad = train_errors
    test_error_dt, test_error_H, test_error_H_grad = test_errors

    print()
    print(f"┌────────────────────────────────────────────────────────┐")
    print(f"│   Target             Train Errors        Test Errors   │")

    if train_error_dt and test_error_dt:
        print(f"│ - x_dot     :        {train_error_dt     : >10.4E}          {test_error_dt     : >10.4E}    │")
    if train_error_H and test_error_H:
        print(f"│ - H(x)      :        {train_error_H      : >10.4E}          {test_error_H      : >10.4E}    │")
    if train_error_H_grad and test_error_H_grad:
        print(f"│ - H_grad(x) :        {train_error_H_grad : >10.4E}          {test_error_H_grad : >10.4E}    │")
    print(f"└────────────────────────────────────────────────────────┘ ")
    print()
