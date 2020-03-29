import torchvision
import torch
from torch import nn, optim
from torchvision import datasets, transforms,models
from collections import OrderedDict
import numpy as np
from PIL import Image
import json
import argparse

def load_checkpoint(path_to_file, device):
    ''' Load checkpoint from path_to_file'''
    #Specify map location to load checkpoint in cpu or in gpu
    if device == 'gpu':
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    #load checkpoint
    checkpoint = torch.load('checkpoint.pth', map_location=map_location)
    #rebuild_model
    model_name = checkpoint['name']
    model = eval(f'models.{model_name}(pretrained=True)')
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict(checkpoint['sequence']))
    model.classifier[-1] = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(params= [p for p in model.parameters() if p.requires_grad])
    optimizer.state_dict = checkpoint['optimizer_state_dict']
    return model, optimizer


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    #Resize the image
    size = (int(im.size[0] / (1.953)) , int(im.size[1] / 1.953))
    im = im.resize(size)
    #Center crop the image
    left = (im.size[0] - 224)/2
    top = (im.size[1] - 224)/2
    right = (im.size[0] + 224)/2
    bottom = (im.size[1] + 224)/2
    im = im.crop((left, top, right, bottom))
    #Normalize image
    np_im = np.array(im) / 255.0
    means = np.array([0.485, 0.456, 0.406])
    sds = np.array([0.229, 0.224, 0.225])
    np_im = (np_im - means) / sds
    #Transpose data
    np_im = np_im.transpose(2, 0, 1)
    return torch.from_numpy(np_im).float()

def predict(image_path, model, device, topk=1, cat_to_name = None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = image.to(device)
    model.to(device)
    class_to_idx = model.class_to_idx
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    logps1 = model.forward(image.unsqueeze_(0))
    ps1 = torch.exp(logps1)
    top_ps, top_class = ps1.topk(topk, dim = 1)
    top_ps = top_ps.tolist()[0]
    top_class = [idx_to_class[x] for x in top_class.tolist()[0]]
    if cat_to_name:
        with open(cat_to_name, 'r') as f:
            cat_to_name = json.load(f)
        top_names = [cat_to_name[x] for x in top_class]
        return top_ps, top_names
    else:
        return top_ps, top_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='path to image to predict', type = str)
    parser.add_argument('checkpoint', help='path to checkpoint file', type = str)
    parser.add_argument('--top_k', '-tk', help = 'top k classes and probabilities to show', default='1')
    parser.add_argument('--category_names', '-cn', help = 'Use a mapping of categories to real name', default=None)
    parser.add_argument('--gpu', help = 'use gpu for calculations', action='store_true')
    
    
    args = parser.parse_args()

    device = 'cuda' if args.gpu else 'cpu'
    top_k = eval(args.top_k)
    model, optimizer = load_checkpoint(args.checkpoint, device)
    top_ps, top_class = predict(args.image_path, model, device, topk=top_k, cat_to_name = args.category_names)
    top_ps = [round(x, 2) for x in top_ps]
    print(f'Top classes : {top_class}. Top probabilities : {top_ps}')

if __name__ == "__main__":
    main()
