def get_model_error(model, inputs, truths):
    dt_true, H_true, H_grad_true = truths

    error_dt = model.evaluate_dt(inputs, dt_true)
    error_H = model.evaluate_H(inputs, H_true)
    error_H_grad = model.evaluate_H_grad(inputs, H_grad_true)

    return (error_dt, error_H, error_H_grad)
