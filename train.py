#coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def get_accuracy(clothesNet, x, y):
    '''
    Вычисление точности на выборке
    '''
    output = clothesNet(x)
    _, predicted = torch.max(output, 1)
    
    return accuracy_score(y, predicted)


def train(clothesNet, trainX, trainY, testX, testY,epochs, batch_size = 4, learning_rate = 5.0e-4):
    '''
    Обучение модели
    
    Arguments:
        clothesNet {ClothesNet} -- Сверточная нейронная сеть
        trainX {torch.tensor} -- Обучающая выборка
        trainY {torch.tensor} -- Обучающая выборка
        testX {torch.tensor} -- Тестовая выборка
        testY {torch.tensor} -- тестовая выборка
        epochs {int} -- Количество эпох
    
    Keyword Arguments:
        batch_size {int} -- Размер минибатча (default: {4})
        learning_rate {float} -- Шаг градиента (default: {5.0e-4})
    '''

    #Целевая функция
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(clothesNet.parameters(), lr = learning_rate, momentum = 0.9)

    n = trainX.size()[0]

    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []

    print("Start training")
    for epoch in range(epochs):

        for i in range(0, n, batch_size):

            inputs = trainX[i : i+batch_size]
            labels = trainY[i : i+batch_size]
            #print(inputs.size())
            optimizer.zero_grad()

            outputs = clothesNet(inputs)

            

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()



        #Подсчет параметров
        train_output = clothesNet(trainX)
        train_loss = criterion(train_output, trainY).item()
        train_losses.append(train_loss)

        test_output = clothesNet(testX)
        test_loss = criterion(test_output, testY).item()
        test_losses.append(test_loss)

        train_accuracy = get_accuracy(clothesNet, trainX, trainY)
        train_accuracies.append(train_accuracy)

        test_accuracy = get_accuracy(clothesNet, testX, testY)
        test_accuracies.append(test_accuracy)

        print('Epoch #{}'.format(epoch+1))
        print('\tTrain loss    : {0:.10f}'.format(train_loss))
        print('\tTest loss     : {0:.10f}'.format(test_loss))
        print('\tTrain accuracy: {0:.10f}'.format(train_accuracy))
        print('\tTest accuracy : {0:.10f}'.format(test_accuracy))
        print('#', '~'*40, '#')
    
    print("Finish training!")
    print("Result accuracy in test set:")
    test_accuracy = get_accuracy(clothesNet, testX, testY)
    print('Test accuracy : {0:.10f}'.format(test_accuracy))

    # График ошибки
    plt.figure()
    plt.plot(np.array(train_losses), label = 'train loss', color ='blue')
    plt.plot(np.array(test_losses), label = 'test loss', color = 'red')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss')
    plt.legend()
    
    # График точности
    plt.figure()
    plt.plot(np.array(train_accuracies), label = 'train accuracy', color ='blue')
    plt.plot(np.array(test_accuracies), label = 'test accuracy', color = 'red')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('accuracy')
    plt.legend()
    plt.show()


