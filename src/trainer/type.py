from enum import StrEnum

class TrainerType(StrEnum):
    # traditional neural network trainer using iterative loss/backward/optimize train loop
    TRADITIONAL = "traditional"

    # sampling using algorithms such as ELM, SWIM
    SAMPLER = "sample"

