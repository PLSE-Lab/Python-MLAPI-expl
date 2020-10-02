# %% [code]
import sys
kaggle_path_prefix = "/kaggle/input/unetmasklite/Pytorch-UNet-objdetect-xuyuewei/"
sys.path.append(kaggle_path_prefix)

import argparse
import os
import numpy as np
from tqdm import tqdm
from data_loader import PreDataLoaderInstance
from unet_model import UNet
from torch import optim
import torch
import time
import Loss_funcs
import cv2 as cv
from torch.utils.data import DataLoader
from lr_scheduler import LR_Scheduler


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Using cuda
        if args.cuda:
            print("Using gpu ...")
            print("\n")
            device = torch.device('cuda')
        else:
            print("Using cpu ...")
            print("\n")
            device = torch.device('cpu')

        # Define Dataloader
        train_set = PreDataLoaderInstance(args, args.train_data_path, flag='train')
        val_set = PreDataLoaderInstance(args, args.val_data_path, flag='val')

        self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=4)

        # Define network
        model = UNet(output_planes=1)
        self.model = model.to(device)
        # Define loss
        self.criterion = Loss_funcs.BCELoss

        # Define Optimizer
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)

        # Define Evaluator
        self.evaluator = Loss_funcs.Evaluator(2)

        # Resuming checkpoint
        self.best_pred = 0.0

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        # Resuming checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            # checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            self.model.load_state_dict(checkpoint['state_dict'])

            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            print('\n')

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

        s_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

        self.log = open(os.path.join(self.args.checkpoint_path, 'unet_train_log' + s_time + '.txt'), 'w')

    def training(self, epoch):
        train_loss = 0.0

        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            # print("input size:", image.size())
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target, cuda=self.args.cuda, batch_average=True)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            self.log.write('train_total_loss_iter--'+str(loss.item())+'--' + str(i + num_img_tr * epoch))
            self.log.write('\n')
        self.log.write('train_total_loss_epoch--'+str(train_loss)+'--' + str(epoch))
        self.log.write('\n')

        # print("output shape:", tmp_output.shape)

        print('[Train Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Train Loss: %.3f' % train_loss)

        if self.args.no_val and epoch % self.args.eval_interval == (self.args.eval_interval - 1):
            # save checkpoint at eval_interval
            print('save checkpoint at eval_interval')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, os.path.join(self.args.checkpoint_path, 'checkpoint', str(epoch + 1).zfill(4)+'ckpt.pth'))

        self.train_output = output
        self.train_target = target

    def validation(self, epoch):
        self.evaluator.reset()
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        class_loss = 0

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output = self.model(image)

            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            pred = (pred > 0.5).astype(np.int16)
            tar = target.cpu().numpy().astype(np.int16)
            class_loss += np.sum(np.absolute(tar - pred))
            self.evaluator.add_batch(target.cpu().numpy(), pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print('[Validation Epoch: %d, numImages: %5d]' % (epoch, i * self.args.val_batch_size + image.data.shape[0]))
        print("Validation Class_Loss: ", class_loss)
        print("Validation Acc:{}, Acc_class: {}, mIoU: {}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Validation Loss: %.3f' % test_loss)
        print('\n')

        self.log.write('test_loss-epoch ' + str(test_loss) + '--' + str(epoch))
        self.log.write('\n')
        self.log.write('Class_Loss-epoch ' + str(class_loss) + '--' + str(epoch))
        self.log.write('\n')
        self.log.write("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU)+'--' + str(epoch))
        self.log.write('\n')

        # if epoch == 0:
        #     self.min_test_loss = test_loss
        # if test_loss < self.min_test_loss:
        if mIoU > self.best_pred:
            for i in range(len(output[-1])):
                tmp_output = (output[-1][i].detach().cpu().numpy() * 255).astype(np.uint8)
                tmp_target = (target[-1][i].detach().cpu().numpy() * 255).astype(np.uint8)
                tr_output = (self.train_output[-1][i].detach().cpu().numpy() * 255).astype(np.uint8)
                tr_target = (self.train_target[-1][i].detach().cpu().numpy() * 255).astype(np.uint8)

                cv.imwrite(os.path.join(self.args.checkpoint_path,
                                        'unet_val_mask_output' + str(i) + '_class_' + str(epoch).zfill(
                                            4) + '.png'),
                           tmp_output)
                cv.imwrite(os.path.join(self.args.checkpoint_path,
                                        'unet_val_mask_target' + str(i) + '_class_' + str(epoch).zfill(
                                            4) + '.png'),
                           tmp_target)
                cv.imwrite(os.path.join(self.args.checkpoint_path,
                                        'unet_train_mask_output' + str(i) + '_class_' + str(epoch).zfill(
                                            4) + '.png'),
                           tr_output)
                cv.imwrite(os.path.join(self.args.checkpoint_path,
                                        'unet_train_mask_target' + str(i) + '_class_' + str(epoch).zfill(
                                            4) + '.png'),
                           tr_target)

            print("best_pred improved ...saving model checkpoint")
            print('\n')
            self.best_pred = mIoU
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, os.path.join(self.args.checkpoint_path, str(epoch+1).zfill(4)+'best_pred_ckpt.pth'))


def main():
    parser = argparse.ArgumentParser(description="PyTorch Unet Training")
    parser.add_argument('--train_data_path', type=str, default="/kaggle/input/unetmasklite/Pytorch-UNet-objdetect-xuyuewei/data/object_images",
                        help='train images data set path')
    parser.add_argument('--val_data_path', type=str, default="/kaggle/input/unetmasklite/Pytorch-UNet-objdetect-xuyuewei/data/object_images/object_images_val",
                        help='validation images data set path')
    parser.add_argument('--checkpoint_path', default="",
                        help='location to save the train log')
    parser.add_argument('--target_size', type=int, default=(320, 240),
                        help='target image size')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=600, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch_size', type=int, default=10,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--val_batch_size', type=int, default=7,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no_val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        print('Cuda info:', torch.cuda.get_device_name(0))
        print('Cuda Capability', torch.cuda.get_device_capability(device=0))
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epochs:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        time_start = time.time()
        trainer.training(epoch)
        time_end = time.time()
        print('epoch:', str(epoch), ' cost time:', time_end - time_start, 's')
        print('\n')
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.log.close()


if __name__ == "__main__":
   main()

