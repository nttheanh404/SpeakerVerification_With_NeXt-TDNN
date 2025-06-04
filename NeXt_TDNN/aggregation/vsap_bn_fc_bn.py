import torch
import torch.nn as nn
import torch.nn.functional as F


class VSAP_BN_FC_BN(nn.Module):
    def __init__(self, channel_size, intermediate_size, embeding_size):
        super(VSAP_BN_FC_BN, self).__init__()
        self.channel_size = channel_size
        self.intermediate_size = intermediate_size
        self.embeding_size = embeding_size

        self.conv_1 = nn.Conv1d(self.channel_size, self.intermediate_size, kernel_size=1)
        self.conv_2 = nn.Conv1d(self.intermediate_size, self.channel_size, kernel_size=1)
        self.tanh = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(self.channel_size)

        self.fc = nn.Linear(self.channel_size, self.embeding_size)
        self.bn2 = nn.BatchNorm1d(self.embeding_size)


    def forward(self, x):
        """
        Args:
            x: (batch_size, channel_size, T)
        Returns:
            x: (batch_size, embeding_size)
        """
        assert x.dim() == 3, "x.dim() must be 3"

        attn = self.conv_2(self.tanh(self.conv_1(x)))
        # self.conv_1(x).shape : (batch_size, intermediate_size, T)
        # self.conv_2(self.tapn(self.conv_1(x))).shape : (batch_size, channel_size, T)
        attn = F.softmax(attn, dim=2) # (batch_size, channel_size, T)

        x = torch.sum(x * attn, dim=2) # (batch_size, channel_size)
        x = self.bn1(x)

        x = self.fc(x) # (batch_size, embeding_size)
        x = self.bn2(x)

        return x

def Aggregation(channel_size, intermediate_size, embeding_size):
    return VSAP_BN_FC_BN(channel_size, intermediate_size, embeding_size)