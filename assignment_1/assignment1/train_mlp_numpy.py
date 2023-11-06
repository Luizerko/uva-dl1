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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    n_classes = predictions.shape[1]

    conf_mat = np.zeros((n_classes, n_classes))
    integer_predictions = np.argmax(predictions, axis=1)
    for i, j in zip(integer_predictions, targets):
      conf_mat[i, j] += 1

    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    
    n_classes = confusion_matrix.shape[0]

    metrics = {}
    metrics['accuracy'] = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    metrics['precision'] = np.zeros(n_classes)
    metrics['recall'] = np.zeros(n_classes)
    metrics['f1_beta'] = np.zeros(n_classes)
    for i in range(n_classes):
      metrics['precision'][i] = confusion_matrix[i, i]/(np.sum(confusion_matrix[i, :]) + 1e-6)
      metrics['recall'][i] = confusion_matrix[i, i]/(np.sum(confusion_matrix[:, i]) + 1e-6)

      precision = metrics['precision'][i]
      recall = metrics['recall'][i]
      metrics['f1_beta'][i] = (1 + pow(beta, 2))*precision*recall/(pow(beta, 2)*precision + recall + + 1e-6)
      

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    metrics = {}
    metrics['accuracy'] = 0
    metrics['precision'] = np.zeros(num_classes)
    metrics['recall'] = np.zeros(num_classes)
    metrics['f1_beta'] = np.zeros(num_classes)

    num_processed_points = 0
    for image_batch, target_batch in data_loader:
      flattened_batch = image_batch.reshape(image_batch.shape[0], -1)
      out = model.forward(flattened_batch)

      target_batch = np.array(target_batch)
      conf_mat = confusion_matrix(out, target_batch)

      num_processed_points += len(flattened_batch)
      
      metrics_batch = confusion_matrix_to_metrics(conf_mat)
      metrics['accuracy'] += len(flattened_batch)*metrics_batch['accuracy']
      metrics['precision'] += len(flattened_batch)*metrics_batch['precision']
      metrics['recall'] += len(flattened_batch)*metrics_batch['accuracy']
      metrics['f1_beta'] += len(flattened_batch)*metrics_batch['f1_beta']

    metrics['accuracy'] /= num_processed_points
    metrics['precision'] /= num_processed_points
    metrics['recall'] /= num_processed_points
    metrics['f1_beta'] /= num_processed_points
      
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics



def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    """
    train_data = cifar10_loader['train'].dataset.dataset.data
    train_targets = np.array(cifar10_loader['train'].dataset.dataset.targets)

    validation_data = cifar10_loader['validation'].dataset.dataset.data
    validation_targets = np.array(cifar10_loader['validation'].dataset.dataset.targets)
    
    test_data = cifar10_loader['test'].dataset.data
    test_targets = np.array(cifar10_loader['test'].dataset.targets)
    """

    train_loader = cifar10_loader['train']
    validation_loader = cifar10_loader['validation']
    test_loader = cifar10_loader['test']

    # TODO: Initialize model and loss module

    n_inputs = 32*32*3
    model = MLP(n_inputs=n_inputs, n_hidden=hidden_dims, n_classes=10)
    loss_module = CrossEntropyModule()

    # TODO: Training loop including validation

    #import ipdb
    #ipdb.set_trace()

    train_loss = []
    val_accuracies = []
    for i in range(epochs):
      for count, (image_batch, target_batch) in enumerate(tqdm(train_loader)):
        flattened_batch = image_batch.reshape(image_batch.shape[0], -1)
        out = model.forward(flattened_batch)

        target_batch = np.array(target_batch)
        train_loss.append(loss_module.forward(out, target_batch))

        dL = loss_module.backward(out, target_batch)
        model.backward(dL)

        for layer_index, layer in enumerate(model.hidden_layers):
          if layer_index % 2 == 0:
            layer.params['weight'] -= lr*layer.grads['weight']
            layer.params['bias'] -= lr*layer.grads['bias']

        model.output_layer[0].params['weight'] -= lr*model.output_layer[0].grads['weight']
        model.output_layer[0].params['bias'] -= lr*model.output_layer[0].grads['bias']
        
        #if count % 50 == 0:
        #  print('Train Loss: {}'.format(train_loss[-1]))
      
      val_accuracies.append(evaluate_model(model, validation_loader, 10)['accuracy'])
      #print('Validation Accuracy: {}'.format(val_accuracies[-1]))
      if len(val_accuracies) > 1 and val_accuracies[-2] > val_accuracies[-1]:
        if np.isclose(val_accuracies[-2], val_accuracies[-1], rtol=3e-2):
          continue
        else:
          break

    # TODO: Test best model
    test_accuracy = evaluate_model(model, test_loader, 10)['accuracy']
    print('Test Accuracy: {}'.format(test_accuracy))
    # TODO: Add any information you might want to save for plotting
    logging_dict = {'train_loss': train_loss, 'validation_accuracy': val_accuracies}
    
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(train_loss)), train_loss, c='r', label='Train Loss')
    plt.legend()
    plt.show()
    plt.plot(np.linspace(0, len(train_loss), len(val_accuracies)), val_accuracies, c='b', label='Validation Accuracy')
    plt.legend()
    plt.show()
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    