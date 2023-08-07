import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

class Bilinear(nn.Module):
    def __init__(self, input1_size, input2_size, output_size, bias=True):
        super(Bilinear, self).__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.output_size = output_size

        self.weight = nn.Parameter(torch.Tensor(input1_size, input2_size, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input1, input2):
        input1_size = list(input1.size())
        input2_size = list(input2.size())

        intermediate = torch.mm(input1.view(-1, input1_size[-1]), self.weight.view(-1, self.input2_size * self.output_size),)

        input2 = input2.transpose(1, 2)
        output = intermediate.view(input1_size[0], input1_size[1] * self.output_size, input2_size[2]).bmm(input2)

        output = output.view(input1_size[0], input1_size[1], self.output_size, input2_size[1]).transpose(2, 3)

        if self.bias is not None:
            output = output + self.bias

        return output

class Biaffine(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bias=True, bias_init=None):
        super(Biaffine, self).__init__()

        self.trans_1 = nn.Sequential(*[nn.Linear(input_dim, hidden_dim, bias=True)])
        self.trans_2 = nn.Sequential(*[nn.Linear(input_dim, hidden_dim, bias=True)])

        self.linear_1 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.linear_2 = nn.Linear(hidden_dim, output_dim, bias=False)

        self.bilinear = Bilinear(hidden_dim, hidden_dim, output_dim, bias=bias)
        if bias_init is not None:
            self.bilinear.bias.data = bias_init

        self.trans_1.apply(init_weights)
        self.trans_2.apply(init_weights)
        self.linear_1.apply(init_weights)
        self.linear_2.apply(init_weights)


    def forward(self, x, y, sent_num=None):
        x, y = self.trans_1(x), self.trans_2(y)
        res = self.bilinear(x, y) + self.linear_1(x).unsqueeze(2) + self.linear_2(y).unsqueeze(1)
        if sent_num:
            return res[:, :sent_num, :sent_num]
        else:
            return res