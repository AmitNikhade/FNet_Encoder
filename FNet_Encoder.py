import torch
from torch import nn
from torch.nn import functional as F



class ff(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        
        return self.net(x)
class FNetLayer(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.feedForward = ff(dim, hidden_dim, dropout)
    def forward(self, x):
        residual = x
        x = torch.fft.fft2(x, dim=(-1, -2)).real
        x = self.norm(x+residual)
        x = self.feedForward(x)
        x = self.norm(x+residual)
        return x
    
class FNet(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, layers):
        super().__init__()
  
        self.Encoder = FNetLayer(dim, hidden_dim, dropout)
        self._layers_e = nn.ModuleList()
        for i in range(layers):
            layer = self.Encoder 
            self._layers_e.append(layer)
            
    def forward(self, x):
        for e in self._layers_e:
            x = e.forward(x)
        return x
        
model = FNet(dim=256, hidden_dim=512, dropout=.5, layers=2)
print(model)
x = torch.randint(1, 20, size=(20, 256))

output = model(x)
print(output)