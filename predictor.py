import torch
from torchvision import models
import torch.nn as nn
from utils import *
from dataloader import get_rust_loader, get_annotated_rust_loader
from unet import UNet

import imageio

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_resnet_predictor(config):
    model_path = config['model_path']
    model = models.resnet18(pretrained=True)
    #freeze all params
    for params in model.parameters():
        params.requires_grad_ = False

    #add a new final layer
    nr_filters = model.fc.in_features  #number of input features of last layer
    model.fc = nn.Linear(nr_filters, 1)
    model = model.to(device)

    load_checkpoint(model_path, model, None)
    dataloader = get_rust_loader(config['loader'], True)

    return ModelPredictor(model, dataloader, **config)

def create_unet_predictor(config):
    model_path = config['model_path']
    in_channels = config.get('in_channels', 3)
    out_channels = config.get('out_channels', 1)
    model = UNet(in_channels, out_channels)
    model = model.to(device)

    load_checkpoint(model_path, model, None)
    dataloader = get_annotated_rust_loader(config['test_loader'])

    return ModelPredictor(model, dataloader, **config)

class ModelPredictor:
    def __init__(self, model, dataloader, output_dir, threshold = 0.0, **kwargs):
        self.model = model
        self.dataloader = dataloader
        self.threshold = threshold
        self.output_dir = output_dir
    def predict(self):
        idx = 0
        Sigmoid = nn.Sigmoid()
        with torch.set_grad_enabled(False):   
            for inputs in self.dataloader:
                # track history if only in train
                outputs = [ Sigmoid(self.model(input_.unsqueeze(0).to(device))).squeeze(0)  for input_ in inputs ]
                preds = [  output > self.threshold for output in outputs ]
                if preds[0].ndim == 3:
                    for pid in range(len(preds)):
                        pred = preds[pid].cpu().numpy()
                        pred = np.transpose(pred * 255, (1, 2, 0)).astype('uint8')
                        output = outputs[pid].cpu().numpy()
                        output = np.transpose(output * 255, (1, 2, 0)).astype('uint8')
                        imageio.imwrite(self.output_dir + '/' + str(idx) +'_mask.png', pred)
                        imageio.imwrite(self.output_dir + '/' + str(idx) +'_prob.png', output)
                        idx += 1

