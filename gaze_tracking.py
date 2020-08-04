#from pytorch tutorial 
#modified by Jonathan

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
import resnet
import lr_scheduler

plt.ion()   # interactive mode

wandb.init(project= "gaze_resnet_freezed")
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epo', default=10, type=int, help='epochs')
parser.add_argument('--mom', default=0.8, type=float, help='momentum')
parser.add_argument('--batch', default=30, type=int, help='batch size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--path', default="/home/ubelix/artorg/js14j080", type=str, help='PATH to Folder Train and Val')


args = parser.parse_args()
config = wandb.config
config.batch_size = args.batch
config.lr = args.lr
config.momentum = args.mom
config.epochs = args.epo



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



######################################################################
# Load Data
# ---------
#
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(240),
        transforms.Resize(224),
        transforms.ColorJitter(brightness=0.8),
        #transforms.RandomVerticalFlip(p=0.5),
        #transforms.RandomHorizontalFlip(p=0.5),
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
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            for inputs, labels in dataloaders[phase]:
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
           # if phase == 'train':
           #     scheduler.step(loss)
            if phase == 'val':
                scheduler.step(loss,epoch)




            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'train':
                wandb.log({"Train Acc": epoch_acc,
                           "Train Loss": epoch_loss})
            else:
                wandb.log({
                    "Validation Acc": epoch_acc,
                           "Validation Loss": epoch_loss})



            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'checkpoint_freezed_gaze.pth')
		#wandb.log({ "Example": visualize_model(model_conv)})


        print()

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
        for i, (inputs, labels) in enumerate(dataloaders['val']):
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




model_conv = models.resnet50(pretrained=True)
#model_conv = models.resnet(pretrained=True)  
num_ftrs = model_conv.fc.in_features
#ct = 0

#for param in model_conv.parameters():
#    param.requires_grad = False


for child in model_conv.children():
   ct += 1
   if ct < 9:
           for param in child.parameters():
               param.requires_grad = False
model_conv.fc = nn.Linear(num_ftrs,2)
model_conv = model_conv.to(device)
print(model_conv) 

#model = models.resnet18(pretrained=True)
#############################################################################
######################## Adding dropout #####################################
#############################################################################

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.parameters(), lr=args.lr, momentum=args.mom )

exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_conv, mode='min', factor=0.1, min_lr=0)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.01)


######################################################################
# Train and evaluate


model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=args.epo)

######################################################################
#


wandb.log({
            "Example": visualize_model(model_conv)})







#files = listdir("/home/ubelix/artorg/js14j080/cat_dog/data/train/")
#f = random.choice(files)
#img = Image.open("/home/ubelix/artorg/js14j080/cat_dog/data/train/" + f)
#subprocess.Popen("mkdir /home/ubelix/artorg/js14j080/cat_dog/data/test_result", shell=True)
#img.save('/home/ubelix/artorg/js14j080/ca


