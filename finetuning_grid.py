import torch
import torch.optim as optim
from torch.nn import DataParallel
import csv
import itertools
import argparse
import random
from models.model import ImageNetModel, CLIPModel

import medmnist
from medmnist import INFO, Evaluator
from dataset.MIMIC_dataset import RSNADataset
import torchvision.transforms as transforms

import torchvision.models as models
from sklearn.metrics import f1_score, roc_auc_score
import sys
import random
import clip
from tqdm import tqdm
import PIL
import torch.utils.data as data

import torch.nn as nn
import os
import numpy as np
from timm.models import vit_base_patch16_224
from torchvision import datasets, transforms
from models.model import ImageNetModel, CLIPModel, ViTModel, CheXpertModel
import torchxrayvision as xrv


def load_pretrained_model():
    if args.initialisation == 'clip':
        base_model, preprocess = clip.load(load_model, 'cuda', jit=False)
        full_model = CLIPModel(base_model, num_classes=n_classes)
    elif args.initialisation == 'imagenet' or args.initialisation == 'chexpert':
        if args.use_pretrained:
            print('USING PRETRAINED WEIGHTS')
            if args.model == 'resnet18':
                full_model =  getattr(models, load_model)(pretrained=True)
                full_model.fc = nn.Linear(full_model.fc.in_features, 2)
            elif args.model == 'densenet121-res224-chex':
                full_model = CheXpertModel(xrv.models.DenseNet(weights="densenet121-res224-chex"),2) # CheXpert (Stanford)
                # full_model.classifier = nn.Linear(full_model.classifier.in_features, args.n_classes)

            elif args.model=='vit_b_16':
                print('in thisssssssssss')
                #image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
                # model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
                # print(model)
                # model_name = 'vit_base_patch16_224'  # Example model, you can use other available ViT models
                full_model = vit_base_patch16_224(pretrained=True)
                full_model.head = nn.Linear(full_model.head.in_features, 2)
                # model =  getattr(models, load_model)(pretrained=True)
        else:
            model =  getattr(models, load_model)(pretrained=False)
        # model = torch.nn.Sequential(*list(model.children())[:-1])
        # print('modellllll after', model)
        # full_model = ViTModel(model, in_channels = 768, num_classes=n_classes)
        # print('finallll modellll', full_model)
    checkpoint = torch.load(args.load_model_weights)  # Provide the path to your weights file
    full_model.load_state_dict(checkpoint['model_state_dict'])
    print(full_model)

    return full_model



def train_and_validate_model(model, criterion, optimizer, train_loader, val_loader, epochs, model_id, device):
    model.to(device)
    best_auc = 0

    for epoch in range(epochs):
        train_correct = 0
        train_total = 0
        val_correct = 0
        val_total = 0
        train_predictions = []
        train_targets = []
        val_predictions = []
        val_targets = []
        train_loss = 0
        val_loss = 0

        train_f1_predictions = []
        val_f1_predictions = []

        model.train()
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            m = nn.Softmax(dim=1)
            predicted = m(outputs)

            if args.task == 'multi-label':
                targets = targets.squeeze().float()
                loss = criterion(outputs.squeeze(), targets)
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs.squeeze(), targets)
            # print(targets, outputs)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

            _, t = torch.max(outputs.data, 1)
            # print(predicted)
            train_total += targets.size(0)
            train_correct += (t == targets).sum().item()
            if args.task == 'binary-class':
                predicted = predicted[:,-1]

            # outputs, predicted, t = outputs.to("cpu"), predicted.to("cpu"), t.to("cpu")

            train_predictions.extend(predicted.detach().cpu().numpy())
            train_f1_predictions.extend(t.detach().cpu().numpy())
            train_targets.extend(targets.detach().cpu().numpy())
        train_accuracy = 100 * train_correct / train_total

        if args.task != 'binary-class':
            train_f1 = f1_score(train_targets, train_f1_predictions, average = 'macro')
            train_auc = roc_auc_score(train_targets, train_predictions,average = 'macro', multi_class = 'ovr')
        else:
            train_f1 = f1_score(train_targets, train_f1_predictions)
            train_auc = roc_auc_score(train_targets, train_predictions)

        train_loss = train_loss / len(train_loader.dataset)

        print(f"Epoch {epoch+1}, Training Accuracy: {train_accuracy:.2f}%, F1-Score: {train_f1:.4f}, AUC: {train_auc:.4f}, Loss: {train_loss:.4f}")

        model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                # forward pass
                outputs = model(inputs)
                m = nn.Softmax(dim=1)
                predicted = m(outputs)

                if args.task == 'multi-label':
                    targets = targets.squeeze().float()
                    # print(targets.shape, outputs.squeeze().shape)
                    loss = criterion(outputs.squeeze(), targets)
                else:
                    targets = targets.squeeze().long()
                    loss = criterion(outputs.squeeze(), targets)


                _, t = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (t == targets).sum().item()

                val_loss += loss.item() * inputs.size(0)
                if args.task == 'binary-class':
                    predicted = predicted[:,-1]
                # outputs, predicted, t = outputs.to("cpu"), predicted.to("cpu"), t.to("cpu")

                val_predictions.extend(predicted.detach().cpu().numpy())
                val_targets.extend(targets.detach().cpu().numpy())
                val_f1_predictions.extend(t.detach().cpu().numpy())

            val_accuracy = 100 * val_correct / val_total
            if args.task !='binary-class':
                val_f1 = f1_score(val_targets, val_f1_predictions, average = 'macro')
                val_auc = roc_auc_score(val_targets, val_predictions, average = 'macro', multi_class = 'ovr')
            else:
                val_f1 = f1_score(val_targets, val_f1_predictions)
                val_auc = roc_auc_score(val_targets, val_predictions)
            val_loss = val_loss / len(val_loader.dataset)
            # Print or log validation metrics
            print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.2f}%, F1-Score: {val_f1:.4f}, AUC: {val_auc:.4f}, Loss: {val_loss:.4f}")

            # Save the model if it has the best F1-score
            if val_auc > best_auc:
                best_auc = val_auc
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'hyperparameters': {'learning_rate': lr,'batch_size': BATCH_SIZE, 'epochs':epochs, 'best_epoch': epoch},
                    'results':{'train_loss':train_loss,'val_loss':val_loss,'train_acc':train_accuracy,'val_acc':val_accuracy,'train_F1':train_f1,'val_F1':val_f1, 'train_AUC':train_auc,'val_AUC':val_auc},
                    }
                model_path = f'model_{model_id}.pth'
                torch.save(checkpoint,os.path.join(args.output_model_path, model_path))


def evaluate_model(model, test_loader,device):

    test_correct = 0
    test_total = 0
    test_predictions = []
    test_targets = []
    test_f1_predictions = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            m = nn.Softmax(dim=1)            
            predicted = m(outputs)
            # _, predicted = torch.max(outputs.data, 1)
            targets = targets.squeeze().long()
            test_total += targets.size(0)
            _,t = torch.max(outputs.data, 1)
            test_correct += (t == targets).sum().item()
            # print(predicted)
            if args.task == 'binary-class':
                predicted = predicted[:,-1]

            test_predictions.extend(predicted.detach().cpu().numpy())
            test_f1_predictions.extend(t.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
            # print(test_correct, test_total)
        test_accuracy = 100 * test_correct / test_total
        if args.task !='binary-class':
            test_f1 = f1_score(test_targets, test_f1_predictions, average = 'macro')
            test_auc = roc_auc_score(test_targets, test_predictions, average = 'macro', multi_class = 'ovr')
        else:
            test_f1 = f1_score(test_targets, test_f1_predictions)
            test_auc = roc_auc_score(test_targets, test_predictions)
        # Print or log validation metrics
        print(f"Test Accuracy: {test_accuracy:.2f}%, F1-Score: {test_f1:.4f}, AUC: {test_auc:.4f}")
        return test_accuracy, test_f1, test_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model', default='resnet18', type=str, help='pretrained model')
    parser.add_argument('--task', default='binary-class', type=str, help='available things {multi-label, multi-class, binary-class}')
    parser.add_argument('--csv', default='/home/santosh.sanjeev/model-soups/my_soups/metadata/RSNA_final_df.csv', type=str, help='Data directory')
    parser.add_argument('--data_dir', default='/home/santosh.sanjeev/rsna_18/train/', type=str, help='csv file containing the stats')
    parser.add_argument('--n_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--initialisation', default='imagenet', type=str, help='weight initialisation')
    parser.add_argument('--dataset', default='RSNA', type=str, help='which dataset')
    parser.add_argument('--norm', default=0.5, type=float, help='which norm')
    parser.add_argument('--use_pretrained', action='store_true')

    parser.add_argument('--epoch_list', default=[15, 20, 25], type=list, help='set of epochs')
    parser.add_argument('--lr_list', default=[1e-4, 1e-5, 1e-6, 1e-7], type=list, help='set of lrs')
    parser.add_argument('--optimizer_list', default=["SGD", "AdamW", "RMSprop"], type=list, help='set of optimizers')
    parser.add_argument('--load_model_weights', default='/home/santosh.sanjeev/soups/my_soups/checkpoints/full_finetuning/pneumoniamnist/initial_full_finetuning_model.pth', type=str, help='which system you are using')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--output_model_path', default='/home/santosh.sanjeev/model-soups/my_soups/checkpoints/grid_models/', type=str, help='model path')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set seed for PyTorch
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    if args.dataset[-5:] == 'mnist':
        data_flag = args.dataset
        download = True
        info = INFO[data_flag]
        task = info['task']            
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
    else:
        n_classes = args.n_classes
    load_model = args.model

    learning_rates_list = args.lr_list
    epochs_list = args.epoch_list
    optimizers_list = args.optimizer_list
    BATCH_SIZE = args.batch_size



    print(learning_rates_list, epochs_list, optimizers_list)

    if args.norm!=0.5:
        print('USING IMAGENET NORM')
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
    else:
        print('NOT USING IMAGENET NORM')
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize = transforms.Normalize(mean=mean, std=std)

    # preprocessing
    data_transform = transforms.Compose([
        # transforms.RandomRotation(degrees=15, fill=(0,0,0)),  # Random rotation with 15 degrees
        # transforms.RandomVerticalFlip(p=0.5),  # Random vertical flip with 50% probability
        # transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
        transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':
        print('CIFARRRRRRRRRRRRRRRRR')
        # # Define transformations to be applied to the images
        # transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        #     transforms.RandomCrop(32, padding=4),  # Randomly crop the image
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
        # ])

        # Load CIFAR-10 dataset
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=data_transform, download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=data_transform, download=True)

        # Split the training dataset into training and validation sets
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    elif args.dataset == 'RSNA':
        train_dataset = RSNADataset(csv_file = args.csv, data_folder = args.data_dir, split='train', transform=data_transform)
        val_dataset = RSNADataset(csv_file = args.csv, data_folder = args.data_dir, split='val', transform=data_transform)
        test_dataset = RSNADataset(csv_file = args.csv, data_folder = args.data_dir, split='test', transform=data_transform)

        train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=True)
        test_dataset = DataClass(split='test', transform=data_transform, download=download,  as_rgb=True)
        val_dataset = DataClass(split='val', transform=data_transform, download=download,  as_rgb=True)

        train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    
    selected_parameters = random.sample([(lr, epoch, optimizer) for lr in learning_rates_list for epoch in epochs_list for optimizer in optimizers_list], 30)

    results = []
    model_id = 0
    with open(os.path.join(args.output_model_path,'results.csv'), mode='w', newline='') as csv_file:
        fieldnames = ['Model ID', 'Learning Rate', 'Epochs', 'Optimizer', 'Test Accuracy', 'F1 Score', 'AUC']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Train models one after the other
        for lr, epoch, optimizer_name in selected_parameters:
            print(f'\nTraining with LR: {lr}, Epochs: {epoch}, Optimizer: {optimizer_name}, Model ID: {model_id}')

            model = load_pretrained_model()
            criterion = torch.nn.CrossEntropyLoss()

            if optimizer_name == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=lr)
            elif optimizer_name == 'AdamW':
                optimizer = optim.AdamW(model.parameters(), lr=lr)
            elif optimizer_name == 'RMSprop':
                optimizer = optim.RMSprop(model.parameters(), lr=lr)
            else:
                raise ValueError(f'Unknown optimizer: {optimizer_name}')


            train_and_validate_model(model, criterion, optimizer, train_loader, val_loader, epoch, model_id, device = 'cuda')

            # model_path = f'model_{model_id}.pt'
            # torch.save(model.state_dict(), model_path)
            # model.to('cpu')
            test_accuracy, f1, auc_score = evaluate_model(model, test_loader, device = 'cuda')  # Define evaluate_model function

            results.append({'Model ID': model_id, 'Learning Rate': lr, 'Epochs': epoch, 'Optimizer': optimizer_name, 'Test Accuracy': test_accuracy, 'F1 Score': f1, 'AUC': auc_score})
            writer.writerow({'Model ID': model_id, 'Learning Rate': lr, 'Epochs': epoch, 'Optimizer': optimizer_name, 'Test Accuracy': test_accuracy, 'F1 Score': f1, 'AUC': auc_score})

            torch.cuda.empty_cache()

            model_id += 1

    print("Training completed and results saved.")
