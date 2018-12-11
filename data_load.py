#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt 
import torch
import os


def _load_images(folder: str):
    '''
    Загрузка изображений из папки
    Arguments:
        folder {str} -- [папка/директория]
    
    Returns:
        [np.array, shape 3*32*32] -- [Массив изображений]
    '''
    image_names = os.listdir(folder)
    n = len(image_names)
    images = np.zeros((n,3, 32,32))
    for i, img_name in enumerate(image_names, 0):
        image = plt.imread(folder+img_name, 'JPG')
        image = np.reshape(image, (3,32,32))
        images[i] = image
    return images

def load_dataset():
    '''
    Загрузка датасета
    
    Returns:
        [images, labels, np.array] -- [перемешанные изображения и метки к ним]
    '''

    #загрузили одежду
    clothes_images = _load_images('data/clothes/')
    #загрузили другие картини
    other_images = _load_images('data/others/')
    
    #поместили в один маллсв и заодно присвоили меки
    image_count = clothes_images.shape[0] + other_images.shape[0]
    images = np.zeros((image_count,3, 32,32))
    labels = np.zeros((image_count))
    images[:clothes_images.shape[0]] = clothes_images
    labels[:clothes_images.shape[0]] = 1
    images[clothes_images.shape[0]:] = other_images
    labels[clothes_images.shape[0]:] = 0   

    #перемешали
    indexes = np.arange(image_count)
    np.random.shuffle(indexes)
    images = images[indexes]
    labels = labels[indexes]

    images = (images - 128) / 255

    return images, labels
