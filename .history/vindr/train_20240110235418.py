from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import argparse
import PIL
import time
from dataset.MIMIC_dataset import MIMIC_Dataset, XRayCenterCrop, XRayResizer, VinDr_Dataset
# from models.model import ImageNetModel, CLIPModel, ViTModel, CheXpertModel
import models.MIMIC_model as local_model
from transformers import AutoImageProcessor, ViTForImageClassification
import torchvision.models as models
from sklearn.metrics import f1_score, roc_auc_score
import sys
import random
import torchvision
from timm.models import vit_base_patch16_224
from torchvision import datasets, transforms
import os
from scripts import train_utils
import torchvision as xrv
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default="", help='')
    parser.add_argument('--shuffle', type=bool, default=True, help='')
    parser.add_argument('--featurereg', type=bool, default=False, help='')
    parser.add_argument('--weightreg', type=bool, default=False, help='')
    parser.add_argument('--data_aug', type=bool, default=True, help='')
    parser.add_argument('--data_aug_rot', type=int, default=45, help='')
    parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
    parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')
    parser.add_argument('--label_concat', type=bool, default=False, help='')
    parser.add_argument('--label_concat_reg', type=bool, default=False, help='')
    parser.add_argument('--labelunion', type=bool, default=False, help='')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model', default='vit_b_16', type=str, help='pretrained model')
    parser.add_argument('--task', default='binary-class', type=str, help='available things {multi-label, multi-class, binary-class}')
    parser.add_argument('--csv', default='/home/santosh.sanjeev/model-soups/my_soups/metadata/RSNA_final_df.csv', type=str, help='Data directory')
    parser.add_argument('--data_dir', default='/home/santosh.sanjeev/rsna_18/train/', type=str, help='csv file containing the stats')
    parser.add_argument('--output_dir', default='/home/santoshsanjeev/MIMIC_classification/own_prep/checkpoints/', type=str, help='csv file containing the stats')
    parser.add_argument('--n_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--initialisation', default='imagenet', type=str, help='weight initialisation')
    parser.add_argument('--dataset', default='pneumoniamnist', type=str, help='which dataset')
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--lp_ft', default='LP', type=str, help='which type of finetuning')
    
    parser.add_argument('--device', default='cuda', type=str, help='which device')
    parser.add_argument('--norm', default=0.5, type=float, help='which norm')

    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--n_epochs', default=2, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--taskweights', type=bool, default=True, help='')

    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--output_model_path', default='/home/santosh.sanjeev/model-soups/my_soups/checkpoints/full_finetuning/pneumoniamnist/initial_full_finetuning_model.pth', type=str, help='model path')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set seed for PyTorch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    NUM_EPOCHS = args.n_epochs
    BATCH_SIZE = args.batch_size
    lr = args.lr
    load_model = args.model
    data_aug = None

    # print(args.use_pretrained)
    
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
    # data_transform = transforms.Compose([XRayCenterCrop(), XRayResizer(512)])

    data_aug = None
    if args.data_aug:
        data_aug = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomAffine(args.data_aug_rot, 
                                                translate=(args.data_aug_trans, args.data_aug_trans), 
                                                scale=(1.0-args.data_aug_scale, 1.0+args.data_aug_scale)),
            torchvision.transforms.ToTensor(),
            # normalize
        ])
        print(data_aug)
    im_path = "/share/ssddata/mimic_pt"#"/nfs/users/ext_ibrahim.almakky/mimic_pt" #'/nfs/users/ext_ibrahim.almakky/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/files/' 

    if args.dataset_name == "mimic":
        train_dataset = MIMIC_Dataset(imgpath=im_path, 
                                    csvpath=args.data_dir + "mimic-cxr-2.0.0-chexpert.csv.gz",
                                    metacsvpath=args.data_dir + "mimic-cxr-2.0.0-metadata.csv.gz",
                                    splitpath = args.data_dir + "mimic-cxr-2.0.0-split.csv.gz", split = 'train',
                                    transform=None, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    
        val_dataset = MIMIC_Dataset(imgpath=im_path, 
                                    csvpath=args.data_dir + "mimic-cxr-2.0.0-chexpert.csv.gz",
                                    metacsvpath=args.data_dir + "mimic-cxr-2.0.0-metadata.csv.gz",
                                    splitpath = args.data_dir + "mimic-cxr-2.0.0-split.csv.gz", split = 'validate',
                                    transform=None, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
        test_dataset = MIMIC_Dataset(imgpath=im_path, 
                                    csvpath=args.data_dir + "mimic-cxr-2.0.0-chexpert.csv.gz",
                                    metacsvpath=args.data_dir + "mimic-cxr-2.0.0-metadata.csv.gz",
                                    splitpath = args.data_dir + "mimic-cxr-2.0.0-split.csv.gz", split = 'test',
                                    transform=None, data_aug=data_aug, unique_patients=False, views=["PA","AP"])

    elif args.dataset_name == "vindr":
        im_path = '/home/aneeshashmi/generative/xray_classification/vindr/annotations/vindr_rsna/' # None for GT
        train_dataset = VinDr_Dataset(
            imgpath=im_path, 
            csvpath=args.data_dir + "train.csv", # val.csv has vindr + rsna test set
            transform=None, data_aug=data_aug, 
            # unique_patients=False, 
            # views=["AP"], 
            # healthy = args.healthy
            )

        val_dataset = VinDr_Dataset(
                imgpath=im_path, 
                csvpath=args.data_dir + "val.csv", # val.csv has vindr + rsna test set
                transform=None, data_aug=data_aug, 
                # unique_patients=False, 
                # views=["AP"], 
                # healthy = args.healthy
            )

        test_dataset = VinDr_Dataset(
            imgpath=im_path, 
            csvpath=args.data_dir + "test.csv", # val.csv has vindr + rsna test set
            transform=None, data_aug=data_aug, 
            # unique_patients=False, 
            # views=["AP"], 
            # healthy = args.healthy
            )


    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = args.num_workers)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = args.num_workers)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = args.num_workers)
    
    print('MIMIC')
    print(len(train_dataset))
    print("===================")
    print(len(val_dataset))
    print("===================")
    print(len(test_dataset))
    

    print("train_dataset.labels.shape", train_dataset.labels.shape)
    print("test_dataset.labels.shape", test_dataset.labels.shape)
    # exit()
    # create models
    if "densenet" in args.model:
        # model = local_model.DenseNet(num_classes=train_dataset.labels.shape[1], in_channels=1, 
        #                         **local_model.get_densenet_params(args.model))
    
        model = models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features


        model.classifier = nn.Sequential(nn.Linear(num_ftrs, args.n_classes))
    elif "resnet101" in args.model:

        model = models.resnet101(num_classes=train_dataset.labels.shape[1], pretrained=False)
        #patch for single channel
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
    elif "resnet50" in args.model:
        model = models.resnet50(num_classes=train_dataset.labels.shape[1], pretrained=False)
        #patch for single channel
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
    elif "shufflenet_v2_x2_0" in args.model:
        model = models.shufflenet_v2_x2_0(num_classes=train_dataset.labels.shape[1], pretrained=False)
        #patch for single channel
        model.conv1[0] = torch.nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
    elif "squeezenet1_1" in args.model:
        model = models.squeezenet1_1(num_classes=train_dataset.labels.shape[1], pretrained=False)
        #patch for single channel
        model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
    else:
        raise Exception("no model")
    
    # criterion = nn.CrossEntropyLoss()    
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr,weight_decay=1e-6)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)


    print(model)
    train_utils.train(model, train_dataset, train_loader, val_loader, device, args)
