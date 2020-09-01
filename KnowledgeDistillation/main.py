# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:39:55 2019
@author: Administrator
"""

# coding: utf-8

import torch
from torch.utils.data import DataLoader
import torch.utils.data as Data
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
import argparse
from tensorflow.keras.datasets.mnist import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5, help="default: 0.5")
parser.add_argument("--lr", type=float, default=1e-3, help="default: 1e-3")
parser.add_argument("--log_interval", type=int, default=100, help="default: 100")
parser.add_argument("--temp", type=int, default=2, help="temperature default: 2")
parser.add_argument("--train_batch_size", type=int, default=64, help="default: 64")
parser.add_argument("--test_batch_size", type=int, default=512, help="default: 512")
parser.add_argument("--n_epochs", type=int, default=200, help="default: 200")

args = parser.parse_args()

class anNet(nn.Module):
    def __init__(self):
        super(anNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool1 = nn.MaxPool2d(2, 1)
        self.fc3 = nn.Linear(3750, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.relu(x))
        x = x.view(x.size()[0], -1)
        x = self.fc3(x)
        return x

class anNet_deep(nn.Module):
    def __init__(self):
        super(anNet_deep, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.pooling1 = nn.Sequential(nn.MaxPool2d(2, stride=2))
        self.fc = nn.Sequential(nn.Linear(6272, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pooling1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pooling1(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

img_rows, img_cols  = 28, 28
(X_train, y_train), (X_test, y_test) = load_data()
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

torch_dataset_train = Data.TensorDataset(torch.from_numpy(np.double(X_train)),
                                         torch.from_numpy(np.int64(y_train)))
torch_dataset_test = Data.TensorDataset(torch.from_numpy(np.double(X_test)),
                                        torch.from_numpy(np.int64(y_test)))
trainset = torch.utils.data.DataLoader(
    torch_dataset_train,
    batch_size=args.train_batch_size,
    shuffle=True)

valset = torch.utils.data.DataLoader(
    torch_dataset_test,
    batch_size=args.test_batch_size,
    shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
teach_model = anNet_deep()
teach_model = teach_model.to(device)
teach_model.load_state_dict(torch.load('teach_net_params_0.9895.pkl', map_location=device))

student_model = anNet()
student_model = student_model.to(device)

criterion = nn.CrossEntropyLoss()
criterion2 = nn.KLDivLoss()

optimizer = optim.Adam(student_model.parameters(), lr=args.lr)

def get_loss(student_pred, teacher_pred, y_true, T, alpha):
    loss1 = criterion(student_pred, y_true)
    loss2 = criterion2(F.log_softmax(student_pred / T, dim=1),
                       F.softmax(teacher_pred / T, dim=1)) * T * T
    loss = (1 - alpha) * loss1 + alpha * loss2
    return loss

def train_one_epoch(student_model, teach_model, trainset):
    for step, (x_batch, y_batch) in enumerate(trainset):
        x_batch = x_batch.float().to(device)
        y_batch = y_batch.to(device)
        student_pred = student_model(x_batch)
        teacher_pred = teach_model(x_batch)
        loss = get_loss(student_pred, teacher_pred, y_batch, args.temp, args.alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (torch.argmax(student_pred, dim=1) == y_batch).float().mean()
        if (step + 1) % args.log_interval == 0:
            print("Step: {}/{} trainLoss: {:.4} train Acc: {:.4}".format(step+1, len(trainset),
                                                                         loss.item(), acc))

def validate_one_epoch(student_model, teach_model, valset):
    student_model.eval()
    teach_model.eval()
    loss_epoch = 0.0
    total_num = 0
    correct_epoch = 0
    for step, (x_batch, y_batch) in enumerate(valset):
        x_batch = x_batch.float().to(device)
        y_batch = y_batch.to(device)
        student_pred = student_model(x_batch)
        teacher_pred = teach_model(x_batch)
        loss = get_loss(student_pred, teacher_pred, y_batch, args.temp, args.alpha)
        loss_epoch += loss.item()
        total_num += x_batch.shape[0]
        correct_epoch += (torch.argmax(student_pred, dim=1) == y_batch).float().sum()
    print("Evaluation: val Loss: {:.4} val Acc: {:.4}".format( loss_epoch / len(valset),
                                                               correct_epoch / total_num))

for epoch in range(args.n_epochs):
    print("Epoch: {}/{}".format(epoch + 1, args.n_epochs))
    train_one_epoch(student_model, teach_model, trainset)
    validate_one_epoch(student_model, teach_model, valset)

