from xtml.utils import get_xtml_version
from xtml.torch.datamodel import DataLoaderConfig, TrainConfig, TorchTrainer


__version__ = get_xtml_version()
__all__ = ["DataLoaderConfig", "TrainConfig", "TorchTrainer"]
