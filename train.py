import torchvision
from torch import nn, optim
import torch
from torchvision import datasets, transforms,models
from collections import OrderedDict
import argparse


def load_data(data_dir):
    ''' Loads the data to train, validate and test network from data_directory
        INPUT:
            data_dir (str): path to the data directory
        OUTPUT:
            dataloaders (dict): dict of dataloaders for train, validation and test data
            image_datasets (dict): image datasets for train, validation and test data
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = {'train': transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])]),
                   'valid_and_test': transforms.Compose([transforms.Resize(256),
                                                         transforms.CenterCrop(224),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])])}
    image_datasets =  {'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
                   'valid': datasets.ImageFolder(valid_dir, transform = data_transforms['valid_and_test']),
                   'test': datasets.ImageFolder(test_dir, transform = data_transforms['valid_and_test'])           
                  }
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
              'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
              'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)}
    return dataloaders, image_datasets

def choose_architecture(model, hidden_units):
    '''choose the architecture of the model to train
    INPUTS:
      model (string): string defining torch pre-trained model. Options: vgg11, vgg13, vgg16, vgg19
      hidden_units (int): number of hidden layers in the model. If a list of integers is provided,
      each integer will specify the number of hidden units in each hidden layer. For this model, the first hidden layer must have 4096 units and the last must have 102 units. If these numbers are not provided in the list they are added.
    OUTPUTS:
      model: torch trained model with the specified architecture.
      sequence: sequence used to build the classifier
      name: name of the chosen model
    '''
    name = model
    available_models = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
    if model in available_models:
        model = eval(f'models.{model}(pretrained=True)')
        for param in model.parameters():
            param.requires_grad = False
        if type(hidden_units) == int:
            hidden_units = [4096, hidden_units, 102]
        else:
            if 4096 not in hidden_units:
                hidden_units.insert(0, 4096)
            if 102 not in hidden_units:
                hidden_units.append(102)
        layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
        hidden_layers = [nn.Linear(h1, h2) for h1,h2 in layer_sizes]
        sequence = []
        for i in range(len(hidden_layers) - 1):
              sequence.extend([(f'fc{i}', hidden_layers[i]),
                              (f'relu{i}', nn.ReLU()),
                              (f'dropout{i}', nn.Dropout(p=0.4))])
        sequence.extend([('last_fc', hidden_layers[-1]), ('output', nn.LogSoftmax(dim =1))])
        classifier = nn.Sequential(OrderedDict(sequence))
        model.classifier[-1] = classifier
        return model, sequence, name
    else:
        print("Choose a model from this list: ['vgg11', 'vgg13', 'vgg16', 'vgg19']")

def train_model(model, epochs, train_loader, valid_loader, lr, device):
    '''
    This function trains a network with an NLLLoss criterion and Adam optimizer
    INPUT:
        model(torch model)
        epochs(int): number of epochs to train
        train_loader: data loaader for train data
        valid_loader: data loader for validation data
        device (str): 'cpu' or 'gpu'
        lr(float): learning rate
    OUTPUT:
      trained model, and optimizer
    '''
    print('Training model..')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(params= [p for p in model.parameters() if p.requires_grad], lr=lr)
    #use cuda if available
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)    
    print_every = 32
    running_loss = 0
    steps = 0
    for epoch in range(epochs):
        # Train loop
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
    
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()          
            # Validation loop
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {test_loss/len(valid_loader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()
    return model, optimizer


def save_checkpoint(model, sequence, optimizer, image_datasets, save_path, epochs, name):
    """ Save checkpoint file with model most important parameters
      INPUTS:
        model: torch model
        sequence: sequence used to modify the last layer of model
        optimizer: torch optimizer
        dataloader: dataloader for data
        save_path: path to save file including the name and extension of the file
    """
    checkpoint = {'class_to_idx': image_datasets['train'].class_to_idx,
              'state_dict': model.state_dict(),
              'number_of_epochs': epochs,
              'optimizer_state_dict': optimizer.state_dict,
              'sequence': sequence,
              'name': name}
    torch.save(checkpoint, save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', help='the directory containing the training data', type = str)
    parser.add_argument('--save_dir', '-sd', help = 'the directory to save checkpoints', default=None)
    parser.add_argument('--learning_rate', '-lr', help = 'set learning rate', default= '0.001')
    parser.add_argument('--hidden_units', '-hu', help = 'set network\'s hidden units. You can specify either a single number "300", or a list "[1000, 500]"', default= '500')
    parser.add_argument('--gpu', help = 'use gpu for  calculations', action='store_true')
    parser.add_argument('--arch', help= "model from the list ['vgg11', 'vgg13', 'vgg16', 'vgg19']", default= 'vgg11')
    parser.add_argument('--epochs', '-e', help='set number of epochs to train', default='1')
    
    args = parser.parse_args()

    device = 'cuda' if args.gpu else 'cpu'
    learning_rate = eval(args.learning_rate)
    hidden_units = eval(args.hidden_units)
    epochs = eval(args.epochs)
    dataloaders, image_datasets = load_data(args.data_directory)
    model, sequence, name = choose_architecture(args.arch, hidden_units)
    model, optimizer = train_model(model, epochs, dataloaders['train'], dataloaders['valid'], learning_rate, device)
    if args.save_dir:
        print(f'Saving checkpoint to {args.save_dir}')
        save_checkpoint(model, sequence, optimizer, image_datasets, args.save_dir, epochs, name)
    
if __name__ == "__main__":
    main()


