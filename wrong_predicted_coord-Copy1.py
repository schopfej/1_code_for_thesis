# code: From pytorch tutorial
# modified by Jonathan

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import wandb
import torch.nn.functional as F
import torchvision.transforms.functional as TF

plt.ion()   # interactive mode

wandb.init(project= "radius_real_images")
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epo', default=1, type=int, help='epochs')
parser.add_argument('--mom', default=0.8, type=float, help='momentum')
parser.add_argument('--batch', default=30, type=int, help='batch size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--path', default="/home/jonathan/Videos/Patient_Images_Original/new_dataset_13_try_Wednesd_twelve_Feb_2020/", type=str, help='PATH to Folder Train and Val')


args = parser.parse_args()
config = wandb.config
config.batch_size = args.batch
config.lr = args.lr
config.momentum = args.mom
config.epochs = args.epo



##############################################################
### keep image name, that we know the coordinate.... #########
##############################################################



#################################################################
############ costom data loader, so that we keep the image name (coordinate)
#################################################################


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path




class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        #self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4)
        #self.conv4 = nn.Conv2d(in_channels=24, out_channels=40, kernel_size=4)

 
        self.fc1 = nn.Linear(in_features=12 * 53 * 53, out_features= 200)
        self.fc2 = nn.Linear(in_features=200, out_features=100)
        self.out = nn.Linear(in_features=100, out_features=2)

    def forward(self, t):
        # implement the forward pass
        # (1) input layer
        t = t
        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # (4) hidden conv layer
        #t = self.conv3(t)
        #t  = F.relu(t)
        #t = F.max_pool2d(t, kernel_size=2, stride=2)
        # (5) hidden conv layer
        #t = self.conv4(t)
        #t = F.relu(t)
        #t = F.max_pool2d(t, kernel_size=2, stride=2)
        # (6) hidden linear layer
        t = t.reshape(-1, 12 * 53 * 53)
        t = self.fc1(t)
        t = F.relu(t)

        # (7) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        # t = F.softmax(t, dim=1)
        return t





#ad Data
# ---------
#
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(230),
        transforms.Resize(224),
        transforms.ColorJitter(brightness=0.8),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomPerspective(),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #AddGaussianNoise(0., 0.3),
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(230),
        transforms.Resize(224),
    #transforms.RandomPerspective(),transforms.RandomPerspective(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = args.path
image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


device = torch.device("cpu")

######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
#inputs, classes = next(iter(dataloaders['train']))
# Make a grid from batch
#out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])


######################################################################
# Training the model

# -  Scheduling the learning rate
# -  Saving the best model



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, path in dataloaders[phase]:
                #print(path)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()




            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'train':
                wandb.log({"Train Acc": epoch_acc,
                           "Train Loss": epoch_loss})
            else:
                wandb.log({
                    "Test Acc": epoch_acc,
                           "Test Loss": epoch_loss})



            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'checkpoint_86acc.pth')
        #wandb.log({ "Example": visualize_model(model_conv)})




    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights

    wandb.log({
               '{} Accuracy:'.format(
                phase): epoch_acc,
               '{} Loss:'.format(
                phase): epoch_loss})
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# Visualizing the model predictions

















def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels, path) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)





'''
######################################################################
# Finetuning the convnet


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)
wandb.watch(model_ft)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.mom)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
#

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=args.epo)

######################################################################
#

visualize_model(model_ft)


######################################################################
# ConvNet as fixed feature extractor
# ----------------------------------
#
# Here, we need to freeze all the network except the final layer. We need
# to set ``requires_grad == False`` to freeze the parameters so that the
# gradients are not computed in ``backward()``.
#
# You can read more about this in the documentation
# `here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
#

'''

model_conv = models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False


#model = models.resnet18(pretrained=True)
#num_ftrs = model_conv.fc.in_features
#model_conv.fc = nn.Sequential(
#    nn.Dropout(0.2))

#every parameter has an attribute called requires_grad which is by default True. True means it will be backpropagrated
# and hence to freeze a layer you need to set requires_grad to False for all parameters of a layer. This can be done like this

#model = models.resnet18(pretrained=True)
num_ftrs = model_conv.fc.in_features

#ct = 0
#for child in model_conv.children():
#	ct += 1
#	if ct < 3:
#    		for param in child.parameters():
#        		param.requires_grad = False

'''
for name, child in model.named_children():
   if name in ['layer3', 'layer4']:
       print(name + ' is unfrozen')
       for param in child.parameters():
           param.requires_grad = True
   else:
       print(name + ' is frozen')
       for param in child.parameters():
           param.requires_grad = False
'''

#This freezes layers 1-6 in the total 10 layers of Resnet50. Hope this helps!

#optimizer_conv = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0006, momentum=0.9)

# Parameters of newly constructed modules have requires_grad=True by default

model_conv.fc = nn.Linear(num_ftrs,2)
model_conv = model_conv.to(device)

#model = models.resnet18(pretrained=True)
#############################################################################
######################## Adding dropout #####################################
#############################################################################

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.parameters(), lr=args.lr, momentum=args.mom)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=20, gamma=0.0001)


######################################################################
# Train and evaluate


model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=3)

######################################################################
#

import torchvision.transforms.functional as TF



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, path in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Store wrongly predicted images
            wrong_idx = (pred != target.view_as(pred)).nonzero()[:, 0]
            wrong_samples = data[wrong_idx]
            wrong_preds = pred[wrong_idx]
            wrong_name = np.array(path)[wrong_idx]
            actual_preds = target.view_as(pred)[wrong_idx]

            for i in range(len(wrong_idx)):
                sample = wrong_samples[i]
                wrong_pred = wrong_preds[i]
                actual_pred = actual_preds[i]
                file_name = os.path.basename(os.path.normpath(wrong_name[i]))
                # Undo normalization
                sample = sample * 0.3081
                sample = sample + 0.1307
                sample = sample * 255.
                sample = sample.byte().cpu()
                img = TF.to_pil_image(sample)
                img.save('name{}_wrong_idx{}_pred{}_actual{}.png'.format(
                    file_name, wrong_idx[i], wrong_pred.item(), actual_pred.item()))

#test(args, model_conv, 'cpu', dataloaders['val'])

test(model_conv,device,dataloaders['val'])

#visualize_model(model_conv)





