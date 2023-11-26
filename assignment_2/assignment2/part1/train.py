################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models

from cifar100_utils import get_train_validation_set, get_test_set


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(weights='DEFAULT')

    # Randomly initialize and modify the model's last layer for CIFAR100.
    new_in_features = model.fc.in_features
    model.fc = nn.Linear(new_in_features, num_classes)
    nn.init.normal_(model.fc.weight, 0.0, 0.01)
    nn.init.constant_(model.fc.bias, 0.0)

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train_data, val_data = get_train_validation_set(data_dir, validation_size=5000, augmentation_name=augmentation_name)

    train_dataloader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, drop_last=False)

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    loss_module = nn.CrossEntropyLoss()

    # Training loop with validation after each epoch. Save the best model.
    train_loss = []
    val_acc = []
    best_val_acc = 0

    model.to(device)
    for epoch in range(epochs):
        model.train()
        for batch_num, (data_batch, target_batch) in enumerate(train_dataloader):
            data_batch = data_batch.to(device)
            target_batch = target_batch.to(device)
            
            optimizer.zero_grad()
            
            output = model(data_batch)
            loss = loss_module(output, target_batch)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            if batch_num != 0 and batch_num % 100 == 0:
                print('Epoch {}: batch number {} with training loss {}'.format(epoch+1, batch_num, train_loss[-1]))

        val_acc.append(evaluate_model(model, val_dataloader, device))
        print('Epoch {}: validation accuracy {}\n'.format(epoch+1, val_acc[-1]))

        if epoch > 0 and val_acc[-2] > val_acc[-1]:
            if val_acc[-2] - val_acc[-1] > 0.03:
                break
            else:
                continue
        else:
            best_val_acc = val_acc[-1]
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict, 'loss': train_loss[-1], 'val_acc': best_val_acc}, checkpoint_name)

    # Load the best model on val accuracy and return it.
    checkpoint = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint['model_state_dict'])

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    with torch.no_grad():
        acc = 0
        data_processed = 0

        for data_batch, target_batch in data_loader:
            data_batch = data_batch.to(device)
            target_batch = target_batch.to(device)

            output = model(data_batch)
            predictions = torch.argmax(output, dim=1)
            
            acc += torch.sum(predictions == target_batch)
            data_processed += len(target_batch)

    accuracy = acc/data_processed

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name, test_noise):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # Load the model
    model = get_model()

    # Get the augmentation to use
    augmentation_name = augmentation_name

    # Train the model
    train_model(model, lr=lr, batch_size=batch_size, epochs=epochs, data_dir=data_dir,
                checkpoint_name='checkpoint.pt', device=device, augmentation_name=augmentation_name)

    # Evaluate the model on the test set
    test_data = get_test_set(data_dir, test_noise)
    test_dataloader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, drop_last=False)

    test_acc = evaluate_model(model, test_dataloader, device)
    print('Test accuracy {}'.format(test_acc))

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')
    parser.add_argument('--test_noise', default=False, action="store_true",
                        help='Whether to test the model on noisy images or not.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
