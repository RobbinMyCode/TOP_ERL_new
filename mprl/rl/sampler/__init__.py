from typing import Literal

from .abstract_sampler import *
from .black_box_sampler import *
from .temporal_correlated_sampler import *
from .top_erl_sampler import *


def sampler_factory(typ: Literal["BlackBoxSampler",
                                 "TemporalCorrelatedSampler",
                                 "TopErlSampler"],
                    **kwargs):
    """
    Factory methods to instantiate a sampler
    Args:
        typ: sampler class type
        **kwargs: keyword arguments

    Returns:

    """
    return eval(typ + "(**kwargs)")
