from .type import TrainerType
from .traditional_trainer import TraditionalTrainer
from .sampler import Sampler


class Trainer():
    @staticmethod
    def new(trainer_type: TrainerType, **kwargs):
        match trainer_type:
            case TrainerType.TRADITIONAL:
                return TraditionalTrainer(**kwargs)
            case TrainerType.SAMPLER:
                return Sampler(**kwargs)
            case _:
                raise NotImplementedError("trainer type not implemented yet")
