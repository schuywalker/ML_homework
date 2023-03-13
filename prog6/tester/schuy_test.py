import numpy as np
import pandas as pd

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from code_NN.nn import NeuralNetwork
from code_NN.math_util import MyMath
from code_misc.utils import MyUtils

verbose = True

def main():
    print("hello world")
    nuts = NeuralNetwork()
    nuts.add_layer(3, 'tanh')
    nuts.add_layer(4, 'tanh')
    nuts.add_layer(5, 'tanh')
    nuts.add_layer(5, 'tanh')
    nuts.add_layer(3, 'tanh')
    nuts._init_weights()



if __name__ == '__main__':
    main()