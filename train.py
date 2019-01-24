import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchsummary import summary

EPOCHS = 100
BATCH_SIZE = 30
LEARNING_RATE = 0.0001
TRAIN_DATA_PATH = "./train/"
TEST_DATA_PATH = "./test/"
image_size = 360
TRANSFORM_IMG = transforms.Compose([
    # transforms.Resize((420,360)),
    transforms.Grayscale(),
    # transforms.RandomCrop(image_size),
    # transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(360),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225] )
    ])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# class Net(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(Net, self).__init__()                    # Inherited from the parent class nn.Module
#         self.fc1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
#         self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
#         self.fc2 = nn.Linear(hidden_size, num_classes) # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)
#
#     def forward(self, x):                              # Forward pass: stacking each layer together
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 18, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(18, 36, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(8640//2, 100)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = self.pool1(x)
        x = F.relu(self.conv2(x), 2)
        x = self.pool2(x)
        x = self.pool2(x)
        x = self.pool2(x)
        x = self.pool2(x)
        x = x.view(-1, 8640//2)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x



if __name__ == '__main__':

    print("Number of train samples: ", len(train_data))
    print("Number of test samples: ", len(test_data))
    print("Detected Classes are: ", train_data.class_to_idx) # classes are detected by folder structure

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = Net().to(device)
    # summary(model, (1,image_size,image_size))
    summary(model, (1,420,360))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()

    # Training and Testing
    for epoch in range(EPOCHS):
        for step, (x, y) in enumerate(train_data_loader):
            # print(step)
            # b_x = Variable(x.view((-1,1,image_size,image_size))).cuda()   # batch x (image)
            b_x = Variable(x).cuda()
            b_y = Variable(y).cuda()   # batch y (target)

            optimizer.zero_grad()
            output = model(b_x)

            loss = loss_func(output, b_y)

            loss.backward()
            optimizer.step()

            # # Test -> this is where I have no clue
            if step % 20 == 0:
                test_x, test_y = next(iter(test_data_loader))
                b_test_x = Variable(test_x).cuda()
                b_test_y = Variable(test_y).cuda()
                test_output = model(b_test_x)
                pred_y = torch.max(test_output, 1)[1]
                accuracy = float(sum(pred_y == b_test_y)) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)
                # print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())
