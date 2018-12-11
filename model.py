import torch
import torch.nn as nn

import torch.nn.functional as F

def conv3(input_size, output_size, stride = 1):
    return nn.Conv2d(input_size, output_size, kernel_size = 3, stride = stride,
                    padding = 1)

class ClothesNet(nn.Module):
    '''
    Нейронная сеть для классификации изображений
    1 - одежда
    0 - не одежда
    размер входного изображени - 32*32 пикселей
    '''

    def __init__(self):
        super(ClothesNet, self).__init__()
        self.conv1_1 = conv3(3, 32)
        self.conv1_2 = conv3(32,32)
        #max pool
        self.conv2_1 = conv3(32, 64)
        self.conv2_2 = conv3(64, 64)
        #max pool
        self.fc1 = nn.Linear(8*8*64, 100)

        self.fc2 = nn.Linear(100, 2)


    def forward(self, x):
        x = self.conv1_1(x)
        x = F.relu(x)
        
        x = self.conv1_2(x)
        x = F.relu(x)
        
        
        x = F.max_pool2d(x, 2, 2)
       
        x = self.conv2_1(x)
        x = F.relu(x)

        x = self.conv2_2(x)
        x = F.relu(x)

        x = F.max_pool2d(x,  2,  2)

        x = x.view(-1, 8*8*64)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        
        x = F.softmax(x)
        return x

        


        