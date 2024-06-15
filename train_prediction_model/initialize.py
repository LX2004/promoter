import torch

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,(torch.nn.Conv2d,torch.nn.ConvTranspose2d,torch.nn.Linear,torch.nn.BatchNorm2d)):
            torch.nn.init.normal_(m.weight.data,0.0,0.01)
            torch.nn.init.normal_(m.bias.data,0.0,0.01)
