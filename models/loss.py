import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

# class mse_loss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.loss_fn = nn.MSELoss()
#     def forward(self, output, target):
#         return self.loss_fn(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)
def contrastive_loss(output, target,direction=0):
    output = output.squeeze()
    target = target.squeeze()

    euclidean_distance = F.pairwise_distance(output, target, keepdim = True)
    loss_contrastive = torch.mean((1-direction) * torch.pow(euclidean_distance, 2) +
                                  (direction) * torch.pow(torch.clamp(2 - euclidean_distance, min=0.0), 2))
    return loss_contrastive
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class RaceTransformCycleDiffusionLossModel(nn.Module):
    def __init__(self,device='cuda'):
        super().__init__()
        resnet18 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18')
        path = '/home/st392/fsl_groups/grp_nlp/compute/RFW/models/20231121-131306_%f/model_9_88.18226722647567.pt'
        resnet18.load_state_dict(torch.load(path))
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        #freeze the resnet18
        for param in self.resnet18.parameters():
            param.requires_grad = False
        self.resnet18.to(device)
        
    def forward(self, output, target, direction=0):
        #print if ouput has gradient
        outputEmbedding = self.resnet18(output)
        targetEmbedding = self.resnet18(target)
        loss = contrastive_loss(outputEmbedding, targetEmbedding,direction=direction)
        return loss
if __name__ == "__main__":
    output = torch.randn(4, 2048, 1, 1)
    target = torch.randn(4, 2048, 1, 1)

    loss = contrastive_loss(output, target)
