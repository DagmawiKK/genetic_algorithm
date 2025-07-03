from hyperon.atoms import OperationAtom, ValueAtom, E
from hyperon.ext import *
from hyperon import *

import random
import time
import math

# Initialize seed for consistent runs, can be updated dynamically
random.seed(123)


def _update_seed():
    """Internal function to update the random seed for better randomness."""
    random.seed(time.time_ns())

def getRandFloatArray(length):
    _update_seed()
    arr = (ValueAtom(random.uniform(0.0, 1.0)) for _ in range(length.get_object().value))
    return [E(*arr)]
def gaussian_random(mean, stddev):
    _update_seed()
    mean = mean.get_object().value
    stddev = stddev.get_object().value
    return [ValueAtom(random.gauss(mean, stddev))]


@register_atoms
def my_glob_atoms():
    return {
        'randfloatarr!': OperationAtom("randfloatarr!", getRandFloatArray, unwrap=False),
        'gaussian-random': OperationAtom("gaussian-random", gaussian_random, unwrap=False),

    }
