################################################################################
# MIT License
#
# Copyright (c) 2023 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.hidden_layers = []
        for i, n in enumerate(n_hidden):
          if i == 0:  
            self.hidden_layers.append(LinearModule(in_features=n_inputs, out_features=n))
          else:
            self.hidden_layers.append(LinearModule(in_features=prev_n, out_features=n))

          self.hidden_layers.append(ELUModule())
          prev_n = n
        
        if len(n_hidden) > 0:
          self.output_layer = [LinearModule(in_features=n, out_features=n_classes)]
        else:
          self.output_layer = [LinearModule(in_features=n_inputs, out_features=n_classes)]

        self.output_layer.append(SoftMaxModule())


        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        for i, layer in enumerate(self.hidden_layers):
          if i == 0:
            out = layer.forward(x)
          else:
            out = layer.forward(out)

        if len(self.hidden_layers) > 0:
          out = self.output_layer[0].forward(out)
        else:
          out = self.output_layer[0].forward(x)

        out = self.output_layer[1].forward(out)

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.gradients = []

        self.gradients.append(self.output_layer[1].backward(dout))
        self.gradients.append(self.output_layer[0].backward(self.gradients[-1]))

        for i, layer in enumerate(self.hidden_layers[::-1]):
          self.gradients.append(layer.backward(self.gradients[-1]))

        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        for layer in self.hidden_layers:
          layer.clear_cache()

        self.output_layer[0].clear_cache()
        self.output_layer[1].clear_cache()
        #######################
        # END OF YOUR CODE    #
        #######################
