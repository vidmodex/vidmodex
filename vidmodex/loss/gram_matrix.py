import torch as th
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image


def gram_matrix(input):
    a ,b ,c, d = input.size()

    features = input.view(a*b, c*d)

    G = th.mm(features, features.t())

    return G.div(a*b*c*d)

class GramLoss(nn.Module):

    def __init__(self, device):
        super(GramLoss, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True).features.to(device=device).eval()
        
    def forward(self, inp, ref):
        inp_features = self.model(inp)
        ref_features = self.model(ref).detach()
        ginp = gram_matrix(inp_features)
        gref = gram_matrix(ref_features)
        return F.mse_loss(ginp, gref)


