"""
Neural Style Transfer
This implementation used codes from https://github.com/pytorch/tutorials/tree/main/advanced_source
Authors: 
Alexis Jacq: https://alexis-jacq.github.io
Winston Herring: https://github.com/winston6_

Original paper: https://arxiv.org/abs/1508.06576
Authors: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor (3D matrix)


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


unloader = transforms.ToPILImage()  # reconvert into PIL image
plt.ion()


def toPIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        # to caculate how much content loss between input tensors and target tensor
        self.loss = F.mse_loss(input, self.target)
        return input


# Take input of a feature map of a layer and return gram matrix
#  The gram matrix represents the correlations between the different feature maps of a given layer
def gram_matrix(input):
    batch_size, channel, width, height = input.size()
    # extract the features of a layers
    features = input.view(batch_size * channel, width * height)
    # compute the gram product, t(): transpose
    G = torch.mm(features, features.t())
    # normalize the values of the gram matrix by dividing by the number of element in each feature maps.
    return G.div(batch_size * channel * width * height)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# get the model which is a subsets that contains all necessary layers of the cnn for neural style transfer purpose, discard other unecessary layers.
def get_nst_model_and_losses(cnn, normalization_mean, normalization_std,
                             style_img, content_img,
                             content_layers=content_layers_default,
                             style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(
        normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    # Loop through all the layers
    i = 0  # increment every time we see a conv layer
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(
                layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # calculating content loss for a content layer:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # calculating style loss for style layers
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img):
    num_steps = 300
    style_weight = 1000000
    content_weight = 0
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_nst_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]

    while run[0] <= num_steps:
        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            # Loss formular: content_weight * content_losses + style_weight * style_losses
            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1

            check_point = 50
            if run[0] % check_point == 0:
                print("Thread1 - Run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score
        # applies the loss and reduces it to minimize the loss of content and style on the image
        # The optimizer tries to adjust the parameters in such a way that the loss function is minimized

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    print(num_steps)
    return input_img


def toTensor(pil_image):
    tensor_image = loader(pil_image).unsqueeze(0)
    return tensor_image.to(device, torch.float)


def getNSTimage(content_image, style_image):
    content_image = content_image.resize((128, 128))
    style_image = style_image.resize((128, 128))

    tensor_content_img = toTensor(content_image)
    tensor_style_img = toTensor(style_image)

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    input_img = tensor_content_img.clone()

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                tensor_content_img, tensor_style_img, input_img)

    return toPIL(output)


def run_style_transfer_with_weight(cnn, normalization_mean, normalization_std, content_img, style_img, input_img, weight):
    num_steps = 300

    content_weight = weight
    style_weight = 1000000-weight
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_nst_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]

    while run[0] <= num_steps:
        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1

            check_point = 50
            if run[0] % check_point == 0:
                print("Thread2 - Run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    print(num_steps)
    return input_img


def getNSTimageWithWeight(content_image, style_image, weight):
    

    content_image = content_image.resize((128, 128))
    style_image = style_image.resize((128, 128))

    tensor_content_img = toTensor(content_image)
    tensor_style_img = toTensor(style_image)

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    input_img = tensor_content_img.clone()

    output = run_style_transfer_with_weight(cnn, cnn_normalization_mean, cnn_normalization_std,
                                tensor_content_img, tensor_style_img, input_img, weight)

    return toPIL(output)
