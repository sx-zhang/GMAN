from .nonadaptivea3c_train import nonadaptivea3c_train
from .nonadaptivea3c_val import nonadaptivea3c_val
from .gman_train import gman_train
from .gman_val import gman_val

trainers = [ 
    'vanilla_train',
    'learned_train',
]

testers = [
    'vanilla_val',
    'learned_val',
]

variables = locals()