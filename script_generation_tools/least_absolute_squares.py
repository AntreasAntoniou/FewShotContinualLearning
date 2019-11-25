import torch
import torch.nn as nn

x = torch.randn(100)

class AbsoluteDifferenceLayer(nn.Module):
    def __init__(self, alpha=None):
        super(AbsoluteDifferenceLayer, self).__init__()
        if alpha is None:
            self.alpha = nn.Parameter(torch.randn(1), requires_grad=True)
        else:
            self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=True)

    def forward(self, x):
        return torch.abs(x - self.alpha)

layer = AbsoluteDifferenceLayer()

optimizer = torch.optim.Adam(params=layer.parameters(), lr=0.0001)

print('Random init')
out = layer.forward(x)
loss = torch.sum(out)
for i in range(10000):
    print('final stats: ', float(loss), 'final alpha', float(layer.alpha),
          'median', float(x.median()))
    optimizer.zero_grad()
    layer.zero_grad()
    out = layer.forward(x)
    loss = torch.sum(out)
    loss.backward()
    optimizer.step()

