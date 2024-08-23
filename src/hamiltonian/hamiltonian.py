import numpy as np

from .type import HamiltonianType
from .spring import Spring
from .single_pendulum import SinglePendulum
from .lotka_volterra import LotkaVolterra
from .double_pendulum import DoublePendulum
from .henon_heiles import HenonHeiles


class Hamiltonian():
    @staticmethod
    def new(target: HamiltonianType, **kwargs):
        """
        Returns the target Hamiltonian DOF, domain limits, and object.

        @param target: target Hamiltonian system type

        @return input_dim, domain_limits (q and p limits), target_hamiltonian
        """
        match target:
            case HamiltonianType.SPRING:
                q_lims = [ [-1., 1.] ]
                p_lims = [ [-1., 1.] ]

                return 2, (q_lims, p_lims), Spring()

            case HamiltonianType.SINGLE_PENDULUM:
                q_lims = [ [-np.pi, np.pi ] ]
                p_lims = [ [-1, 1] ]

                return 2, (q_lims, p_lims), SinglePendulum()

            case HamiltonianType.DOUBLE_PENDULUM:
                q_lims = [ [-np.pi, np.pi], [-np.pi, np.pi] ]
                p_lims = [ [-1., 1.], [-1., 1.] ]

                return 4, (q_lims, p_lims), DoublePendulum()

            case HamiltonianType.HENON_HEILES:
                q_lims = [ [ -5, 5 ], [-5, 5] ]
                p_lims = [ [ -5, 5 ], [-5, 5] ]

                return 4, (q_lims, p_lims), HenonHeiles()

            case _:
                raise NotImplementedError(f"Hamiltonian type {type} is not implemented yet.")
