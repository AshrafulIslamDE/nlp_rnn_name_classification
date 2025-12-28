import torch.nn as nn
import torch

class HandCraftedRNN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_classes:int):
        super(HandCraftedRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # weight_parameter
        self.w_xh=nn.Parameter(torch.randn(input_size,hidden_size)*0.01)
        self.w_hh=nn.Parameter(torch.randn(hidden_size,hidden_size)*0.01)
        self.w_hy=nn.Parameter(torch.randn(hidden_size,num_classes)*0.01)

        # bias parameter
        self.b_y=torch.zeros(num_classes)
        self.b_h=torch.zeros(hidden_size)

    def forward(self, x):
        # if [[0,0,1,0],[1,0,0,0]] seq_len=2, feature_size=4
        batch_size,seq_length,feature_size = x.size(0),x.size(1),x.size(2)
        h_t=torch.zeros(batch_size,self.hidden_size,device=x.device)
        for t in range(seq_length):
            x_t=x[:,t,:]
            x_t_w_xh=torch.matmul(x_t,self.w_xh)
            h_t_w_hh=torch.matmul(h_t,self.w_hh)
            h_t=x_t_w_xh+h_t_w_hh+self.b_h
            h_t= torch.tanh(h_t)
        prediction = torch.matmul(h_t,self.w_hy)+self.b_y
        return prediction


class BuiltINRnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            nonlinearity="tanh"
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        # Pack padded batch
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),          # must be on CPU
            batch_first=True,
            enforce_sorted=False
        )

        _, h_n = self.rnn(packed)

        # h_n: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]

        return self.fc(last_hidden)









