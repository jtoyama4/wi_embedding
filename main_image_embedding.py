import argparse
import os
import shutil
import time
import nets
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision

from tensorboardX import SummaryWriter

from utils import *
from datasets.cifar_10 import NatCIFAR10
from core.dataloader import DataLoader

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

writer = SummaryWriter(log_dir="./logs")

def main():
    parser = argparse.ArgumentParser(description='PyTorch NAT Training with ImageNet')
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--data_kind', default="CIFAR10",
                        help='Kind of dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--im_gradients', default=True,
                        help='use gradients of image as input')
    parser.add_argument('--im_grayscale', default=True,
                        help='use grayscale image')
    parser.add_argument('--train_decoder', default=False, action="store_true",
                        help='use grayscale image')
    
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    """
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    """

    if args.data_kind=="CIFAR10":
        im_shape: Tuple[int] = (32, 32)
        num_classes = 10

    encoder = nets.get_alexnet_encoder(args.pretrained, args.im_grayscale, args.im_gradients)

    if not args.distributed:
        encoder = torch.nn.DataParallel(encoder).cuda()
    else:
        encoder.cuda()
        encoder = torch.nn.parallel.DistributedDataParallel(encoder)

    decoder = nets.get_alexnet_decoder(encoder.module.get_output_shape(im_shape), num_classes)

    if not args.distributed:
        decoder = torch.nn.DataParallel(decoder).cuda()
    else:
        decoder.cuda()
        decoder = torch.nn.parallel.DistributedDataParallel(decoder)

    print(encoder)
    print(decoder)

    # Data loading code
    if args.data_kind == "CIFAR10":
        print("CIFAR 10")
        
        transforms_list = []

        if args.im_grayscale:
            transforms_list.append(transforms.Lambda(lambda img: img.convert('L')))

        # transforms_list.append(transforms.RandomResizedCrop(224))
        transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.ToTensor())
        
        train_transform = transforms.Compose(transforms_list)

        transforms_list = []

        if args.im_grayscale:
            transforms_list.append(transforms.Lambda(lambda img: img.convert('L')))

        # transforms_list.append(transforms.RandomResizedCrop(224))
        transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.ToTensor())

        val_transform = transforms.Compose(transforms_list)

        train_dataset = NatCIFAR10(root='./data',
                                   train=True,
                                   z_dims=encoder.module.get_output_shape(input_shape=im_shape),
                                   download=True,
                                   transform=train_transform)
        
        val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=val_transform)

    elif args.data_kind == "ImageNet":
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        
        train_dataset = datasets.ImageFolder(
            traindir,train_transform)

        val_dataset = datasets.ImageFolder(
            valdir, val_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    

    # Loss function and optimizer
    # Freeze first layer (Computing the image gradient)
    params_for_optim = encoder.module.features.parameters()
    
    encoder_loss_fn = nn.MSELoss().cuda()
    decoder_loss_fn = nn.CrossEntropyLoss().cuda()
    
    encoder_optim = torch.optim.Adam(params_for_optim, lr=args.lr, weight_decay=args.weight_decay)
    decoder_optim = torch.optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0

    checkpoint = None

    if checkpoint is not None:
        best_acc = checkpoint['best_acc']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        encoder_optim.load_state_dict(checkpoint['encoder_optim'])
        decoder_optim.load_state_dict(checkpoint['decoder_optim'])
        print("=> Successfully restored All model parameters. Restarting from epoch: {}".format(args.current_epoch))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, encoder, decoder, encoder_loss_fn, decoder_loss_fn, encoder_optim, decoder_optim, epoch, train_dataset, args)


def train(train_loader, encoder, decoder, encoder_loss_fn, decoder_loss_fn, encoder_optim, decoder_optim, epoch, train_dataset, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    update_targets = bool((epoch+1) % 3 == 0)
    if args.train_decoder:
        update_decoder = bool((epoch+1) % 10 == 0)
    else:
        update_decoder = False

    # switch to train mode
    encoder.train()

    end = time.time()
    for batch_idx, (idx, x, y, nat) in enumerate(tqdm(train_loader, 0), 1):
        # measure data loading time
        data_time.update(time.time() - end)
        
        e_targets = nat.numpy()
        x, y, nat = x.cuda(), y.cuda(), nat.cuda()
        
        # x = Variable(x)
        outputs = encoder(x)

        # every few iterations greedy re-assign targets.
        if update_targets:
            e_out = outputs.cpu().data.numpy()
            new_targets = calc_optimal_target_permutation(e_out, e_targets)
            # update.
            train_dataset.update_targets(idx, new_targets)
            nat = torch.FloatTensor(new_targets)
            nat = nat.cuda()
        
        
        encoder_loss = encoder_loss_fn(outputs, nat)
        
        # compute gradient and do SGD step
        encoder_optim.zero_grad()
        encoder_loss.backward(retain_graph=True)
        encoder_optim.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 100 == 0:
            writer.add_scalar('encoder_loss', encoder_loss.data, int((epoch+1)*(batch_idx/100)))

        if update_decoder:
            decoder_optim.zero_grad()
            y_pred = decoder(outputs)
            decoder_loss = decoder_loss_fn(y_pred, y)
            decoder_loss.backward()
            decoder_optim.step()
            
            if batch_idx % 100 == 0:
                idx_step = int(((epoch+1)/10)*(batch_idx/100))
                writer.add_scalar('decoder_loss', decoder_loss.data, idx_step)

    # Writer weight + gradient histogram for each epoch
    for name, param in encoder.named_parameters():
        name = 'encoder/'+name.replace('.', '/')
        writer.add_histogram(name, param.clone().cpu().data.numpy(), (epoch+1))
        if param.grad is not None:
            writer.add_histogram(name+'/grad', param.grad.clone().cpu().data.numpy(), (epoch+1))

    
            
if __name__ == "__main__":
    main()        
