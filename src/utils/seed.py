"""모든 라이브러리의 난수 시드를 한 번에 고정."""
import os
import random
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """재현성 확보를 위한 시드 고정."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
