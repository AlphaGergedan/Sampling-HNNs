import numpy as np

class SemiImplicitEuler():
    """
    See symplectic euler, this is the explicit implementation of it for the first alternative:

    Implicit q method:
        q_next = q_prev + step_size * q_dot(q_next, p_prev) = q_prev + step_size * dH/dp(q_next, p_prev)
        p_next = p_prev + step_size * p_dot(q_next, p_prev) = p_prev - step_size * dH/dq(q_next, p_prev)

    For separable system this explicit implementation is the same as symplectic euler.
    """

    def __init__(self):
        pass

    def step(self, x_prev, dH, step_size):
        """
        Given that we have K previous states and a (D/2)-DOF system
        solve the explicit Euler equation and compute the next state
        Note that the method is explicit regardless of the used Hamiltonian

        @param x_prev           : state x at previous time step, of shape (K,D)
        @param dH               : gradient of the Hamiltonian (dH/dq, dH/dp): (K,D) -> (K,D)
        @param step_size               : time step size

        @returns x_next         : state x at next time step n+1 of shape (K,D)
        """
        _, D = x_prev.shape # (K,D)

        q_prev, p_prev = np.split(x_prev, 2, axis=1)

        # update q
        dHdp_prev = dH(x_prev)[:,D//2:]
        q_next = q_prev + step_size * dHdp_prev

        # update p implicitly
        x_implicit_q = np.concatenate((q_next, p_prev)).reshape(-1, D)
        dHdq_implicit_q = dH(x_implicit_q)[:, :D//2]
        p_next = p_prev - step_size * dHdq_implicit_q

        return np.hstack((q_next, p_next))

