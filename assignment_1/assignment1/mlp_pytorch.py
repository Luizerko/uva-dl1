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
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm=False):
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
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ELU layer.

        TODO:
        Implement module setup of the network.
        The linear layer have to initialized according to the Kaiming initialization.
        Add the Batch-Normalization _only_ is use_batch_norm is True.
        
        Hint: No softmax layer is needed here. Look at the CrossEntropyLoss module for loss calculation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        super(MLP, self).__init__()

        self.hidden_layers = []
        for i, n in enumerate(n_hidden):
          if i == 0:  
            self.hidden_layers.append(nn.Linear(n_inputs, n))
          else:
            self.hidden_layers.append(nn.Linear(prev_n, n))
          nn.init.kaiming_uniform_(self.hidden_layers[-1].weight)
          self.hidden_layers[-1].bias.data.fill_(0.0)

          if use_batch_norm:
            bn = nn.BatchNorm1d(n)
            nn.init.constant_(bn.weight, 1.0)
            nn.init.constant_(bn.bias, 0.0)
            self.hidden_layers.append(bn)
          
          self.hidden_layers.append(nn.ELU())
          prev_n = n
        
        if len(n_hidden) > 0:
          self.hidden_layers.append(nn.Linear(in_features=n, out_features=n_classes))
        else:
          self.hidden_layers.append(nn.Linear(in_features=n_inputs, out_features=n_classes))
        nn.init.kaiming_uniform_(self.hidden_layers[-1].weight)
        self.hidden_layers[-1].bias.data.fill_(0.0)

        self.hidden_layers.append(nn.Softmax(dim=1))

        self.hidden_layers = nn.ModuleList(self.hidden_layers)

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
            out = layer(x)
          else:
            out = layer(out)
        
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
    
