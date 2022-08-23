import torch
from torchvision import datasets, models, transforms
from torch import optim
import torch.nn as nn
from utils import *
from dataloader import get_rust_loader, get_annotated_rust_loader
from dice_loss import BCEDiceLoss
from unet import UNet
device = "cuda" if torch.cuda.is_available() else "cpu"

def create_resnet_trainer(config):
    # binary classification
    model = models.resnet18(pretrained=True)
    #freeze all params
    for params in model.parameters():
        params.requires_grad_ = False

    #add a new final layer
    nr_filters = model.fc.in_features  #number of input features of last layer
    model.fc = nn.Linear(nr_filters, 1)
    model = model.to(device)
    learning_rate = config.get('learning_rate', 0.01)
    weight_decay = config.get('weight_decay', 1e-5)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    loss_criterion = nn.BCEWithLogitsLoss()
    train_dataloader, train_dataset_size, val_dataloader, val_dataset_size = get_rust_loader(config['loader'])
    return ModelTrainer(model, optimizer, lr_scheduler, loss_criterion, train_dataloader, val_dataloader, train_dataset_size, val_dataset_size, **config)
    
def create_unet_trainer(config):
    in_channels = config.get('in_channels', 3)
    out_channels = config.get('out_channels', 1)
    model = UNet(in_channels, out_channels)
    model = model.to(device)
    learning_rate = config.get('learning_rate', 0.01)
    weight_decay = config.get('weight_decay', 1e-5)
    alpha = config.get('alpha', 1.0)
    beta = config.get('beta', 1.0)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    loss_criterion = BCEDiceLoss(alpha, beta)
    train_dataloader, train_dataset_size = get_annotated_rust_loader(config['train_loader'])
    val_dataloader, val_dataset_size = get_annotated_rust_loader(config['val_loader'])
    return ModelTrainer(model, optimizer, lr_scheduler, loss_criterion, train_dataloader, val_dataloader, train_dataset_size, val_dataset_size, **config)

class ModelTrainer:
    def __init__(self, model, optimizer, lr_scheduler, loss_criterion, train_dataloader, val_dataloader, train_dataset_size, val_dataset_size, checkpoint_dir, max_epochs = 100, pretrained = None, threshold = 0.0, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.criterion = loss_criterion
        self.max_epochs = max_epochs
        self.dataloaders = {'train': train_dataloader, 'val':val_dataloader}
        self.dataset_sizes = {'train': train_dataset_size, 'val':val_dataset_size}
        if pretrained is not None:
            load_checkpoint(pretrained, self.model, self.optimizer)
        self.best_acc = 0.0
        self.checkpoint_dir = checkpoint_dir
        self.threshold = threshold
    def train(self):
        Sigmoid = nn.Sigmoid()
        for epoch in range(self.max_epochs):
            print(f'Epoch {epoch}/{self.max_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:

                running_loss = 0.0
                running_corrects = 0
                running_overall = 0
                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    # print(inputs)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):          
                              
                        outputs = torch.cat(tuple( self.model(input_.unsqueeze(0).to(device)).flatten() for input_ in inputs ))
                        labels = torch.cat(tuple( label.to(device).flatten()  for label in labels ))
                        # print(outputs, labels)  
                        loss = self.criterion(outputs, labels.type(outputs.dtype))
                        preds = (Sigmoid(outputs) > self.threshold).long()

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * len(inputs)
                    running_corrects += torch.sum(preds == labels.data)
                    running_overall += len(preds)
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / running_overall

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                if phase == 'train':
                    self.scheduler.step(epoch_acc)
                # deep copy the model
                if phase == 'val':
                    is_best = epoch_acc > self.best_acc
                    save_checkpoint({
                        'model_state_dict': self.model.state_dict(), 
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        }, is_best, self.checkpoint_dir)
                    if is_best:
                        self.best_acc = epoch_acc
