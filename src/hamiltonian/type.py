from enum import StrEnum

class HamiltonianType(StrEnum):
    SPRING = "spring"
    SINGLE_PENDULUM = "single_pendulum"
    LOTKA_VOLTERRA = "lotka_volterra"
    DOUBLE_PENDULUM = "double_pendulum"
    HENON_HEILES = "henon_heiles"
