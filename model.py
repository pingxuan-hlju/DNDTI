import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConvolution(nn.Module):
    def __init__(self):
        super(DilatedConvolution, self).__init__()

        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=(2, 2),
                      stride=(1, 1),
                      padding=1,
                      dilation=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2)),
        )

        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=(2, 2),
                      stride=(1, 1),
                      padding=1,
                      dilation=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2)),
        )

        self.conv_1_3 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=(2, 2),
                      stride=(1, 1),
                      padding=1,
                      dilation=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2)),
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=(2, 2),
                      stride=(1, 2),
                      padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2)),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(2, 2),
                      stride=(1, 2),
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2)),
        )

        self.fc = nn.Sequential(
            nn.Linear(26624, 2)
        )

    def forward(self, x):
        y_1 = self.conv_1_1(x)
        y_2 = self.conv_1_2(x)
        y_3 = self.conv_1_3(x)
        y = torch.cat([y_1, y_2], -1)
        y = torch.cat([y, y_3], -1)
        y = self.conv_1(y)
        y = self.conv_2(y)
        y = y.view(y.size(0), -1)
        output = self.fc(y)
        return output


class RelationAttention(nn.Module):
    def __init__(self):
        super(RelationAttention, self).__init__()

        self.node = nn.Linear(200, 50)
        nn.init.xavier_normal_(self.node.weight)
        self.h_n_parameters = nn.Parameter(torch.randn(50, 1))
        nn.init.xavier_normal_(self.h_n_parameters)

    def forward(self, x):
        z = x
        temp_nodes = self.node(x)
        temp_nodes = torch.tanh(temp_nodes)
        nodes_score = torch.matmul(temp_nodes, self.h_n_parameters).cuda()
        beta = F.softmax(nodes_score, dim=2)
        att_z = beta * x
        last_z = att_z + z
        last_z = last_z.reshape(-1, 1, 2, 800)
        return last_z


class CnnModule(nn.Module):
    def __init__(self):
        super(CnnModule, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=2,
                stride=(1, 2),
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=2,
                stride=(1, 2),
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        self.fc = nn.Sequential(
            nn.Linear(6400, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output


