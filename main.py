#coding: utf-8

import argparse
import model
import data_load
import torch
import train
import sys
from sklearn.model_selection import train_test_split

def train_and_save_example():
    images, labels = data_load.load_dataset()

    trainX, testX, trainY, testY = train_test_split(images, labels, test_size = 0.2)

    trainX = torch.from_numpy(trainX).float()
    trainY = torch.from_numpy(trainY).long()
    testX = torch.from_numpy(testX).float()
    testY = torch.from_numpy(testY).long()

    net = model.ClothesNet()
    train.train(net, trainX, trainY, testX, testY, epochs = 25, batch_size = 4)

    torch.save(net.state_dict(), 'saved_model/model')

def load_and_test_example():
    images, labels = data_load.load_dataset()

    net = model.ClothesNet()
    net.load_state_dict(torch.load('saved_model/model'))
    net.eval()

    images = torch.from_numpy(images).float()
    labels = torch.from_numpy(labels).long()

    accuracy = train.get_accuracy(net, images, labels)
    print('Accuracy: {0:.10f}'.format(accuracy))

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action = 'store_const', const = True, default = False)
    parser.add_argument('--test', action = 'store_const', const = True, default = False)
    return parser

if __name__ == "__main__":
    parser = create_parser()
    namespace = parser.parse_args(sys.argv[1:])

    if namespace.train:
        train_and_save_example()
    elif namespace.test:
        load_and_test_example()