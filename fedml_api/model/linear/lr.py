import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # outputs = torch.sigmoid(self.linear(x))
        x = x.flatten(start_dim = 1)
        outputs = self.linear(x)
        # outputs = self.linear(x.flatten(start_dim = 1))
        return outputs
