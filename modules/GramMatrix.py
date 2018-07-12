import torch
import torch.nn as nn

class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()
        features = input.view(a, b, c * d)
        G = torch.bmm(features, features.permute(0, 2, 1))
        #G = G.view(a, b, -1)

        return G.div(a * b * c * d)
