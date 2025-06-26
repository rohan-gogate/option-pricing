import torch 
import torch.nn as nn

class OptionPricingModel(nn.Module):
    def __init__(self, input_dim=6, output_dim=2, hidden_dim=64):
        super(OptionPricingModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, output_dim)  
        )

        def forward(self, x):
            return self.net(x)