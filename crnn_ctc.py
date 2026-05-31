import torch 
import torch.nn as nn

class ResidualBlock(nn.Module):
    # some code is borrowed from the book "Dive into Deep Learning" by Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola
    # this repository on CRNN: https://github.com/meijieru/crnn.pytorch/blob/master/README.md
    # and this tutorial based on tensorflow keras: https://pylessons.com/handwritten-sentence-recognition
    def __init__(self, in_channels, out_channels, activation='relu', skip_conv=True, stride=2, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU() if activation == 'leaky_relu' else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.skip_conv = skip_conv
        if skip_conv:
            self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn_skip = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.skip_conv:
            identity = self.conv_skip(x)
            identity = self.bn_skip(identity)
        out += identity
        out = self.relu(out)
        out = self.dropout(out)
        return out

        # batch_size, channels, height, width = x.size()
        # #out = out.view(batch_size, height * width, channels)
        # out = out.view(batch_size, channels, height * width)
        # # out = out.view(width, batch_size, height * channels)
        #return out

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, activation="leaky_relu", dropout=0.2):
        # input dimension: number of channels of the input image
        # output dimension: length of the vocabulary + 1 (for the CTC blank token)
        super(Model, self).__init__()
        self.layer1 = ResidualBlock(input_dim, 32, activation, True, 1, dropout)
        self.layer2 = ResidualBlock(32, 32, activation, True, 2, dropout)
        self.layer3 = ResidualBlock(32, 32, activation, False, 1, dropout)
        self.layer4 = ResidualBlock(32, 64, activation, True, 2, dropout)
        self.layer5 = ResidualBlock(64, 64, activation, False, 1, dropout)
        self.layer6 = ResidualBlock(64, 128, activation, True, 2, dropout)
        self.layer7 = ResidualBlock(128, 128, activation, True, 1, dropout)
        self.layer8 = ResidualBlock(128, 128, activation, True, 2, dropout)
        self.layer9 = ResidualBlock(128, 128, activation, False, 1, dropout)

        self.blstm1 = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.blstm2 = nn.LSTM(256*2, 64, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(64*2, output_dim + 1)
        # self.blstm1 = nn.LSTM(256, 256, bidirectional=True)
        # self.dropout1 = nn.Dropout(dropout)
        # self.blstm2 = nn.LSTM(256, 64, bidirectional=True)
        # self.dropout2 = nn.Dropout(dropout)
        # self.fc = nn.Linear(128, output_dim + 1)

    def forward(self, x):
        # x = x / 255.0
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)

        x = x.reshape(x.size(0), -1, x.size(1))

        # x = x.reshape(x.size(0), x.size(-3) * x.size(-2), x.size(-1))
        # x = x.reshape(x.size(0), x.size(-3) * x.size(-2), x.size(-1))
        # b, c, h, w = x.size()
        # # print(f'x.size(): {x.size()}') 
        # x = x.view(b, c*h, w)
        # print(f'x.size() after reshaping: {x.size()}')
        #x = x.view(b, h*w, c)

        # the second return from rnn is (h_n, c_n): 
        # tuple containing the final hidden state and final cell state
        # for each layer. h_n and c_n are both tensors of shape 
        # (num_layers * num_directions, batch_size, hidden_size).
        x, _ = self.blstm1(x)
        x = self.dropout1(x)
        x, _ = self.blstm2(x)
        x = self.dropout2(x)
        x = self.fc(x)
        x = torch.log_softmax(x, 2)
        return x
    
