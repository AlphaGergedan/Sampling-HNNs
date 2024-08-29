import numpy as np
from scipy.integrate import solve_ivp

def flow_map_rk45(x0, dH, dt_flow_true=1e-4, dt_obs=1e-1):
    """
    Given initial state x0, simulate the exact flow using a strictly smaller time step dt_flow_true and
    return the final state after dt_obs seconds
    """
    assert dt_obs > dt_flow_true
    # RK45
    t_span = [0, dt_obs]
    t_eval = np.arange(0., round(dt_obs + dt_flow_true, int(np.log10(int(1/dt_flow_true)))), dt_flow_true)

    J = np.array([[0, 1],
                  [-1, 0]])
    dt = lambda _, x: J@dH(x.reshape(x0.shape)).reshape(-1)

    x1 = solve_ivp(dt, t_span, x0, dense_output=False, t_eval=t_eval, rtol=1e-13).y.T[-1]

    return x1
