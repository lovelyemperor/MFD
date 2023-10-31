import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from wideresnet import WideResNet
from densenet import DenseNet3
# from resNet import ResNet
# from hzkresnet import ResNet, Bottleneck
# from ResNext import ResNet, Bottleneck
# from Resnet18 import ResNet, BasicBlock
# from GausiResnet import ResNet, Bottleneck, Bottleneck3, BasicBlock
from manyblocknet import ResNet, Bottleneck, Bottleneck3, BasicBlock
# from GausiResnet_Copy import ResNet, Bottleneck
# from SubTwoLoss import ResNet, Bottleneck
# from resNet import ResNet, Bottleneck
# from RESNET50 import ResNet, Bottleneck
# from pruning import cnn
from removeFCnet import rResNet, rBottleneck, rBottleneck1
# from NewLossAddRemovefc import rResNet, rBottleneck, rBottleneck1
from MobileNet import MobileNetV2
from VGG import vgg16_bn
from NASNet import NasNetA
from Densenet1 import DenseNet
import conf.config as conf
import utils
import argparse
from data.dataLoader import *
from torchsummary import summary
from torchstat import stat
from models.modeling import VisionTransformer, CONFIGS
torchvision.models.mobilenet_v2()
torch.nn.MSELoss(reduction='sum')
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
parser = argparse.ArgumentParser(
    description='WideResnet Training With Pytorch')
parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR10', 'CIFAR100', 'MNIST'],
                    type=str, help='CIFAR10, CIFAR100 or MNIST')
parser.add_argument('--dataset_root', default='',
                    help='Dataset root directory path')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--save_folder', default='./models/CIFAR100/7_5/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--save_txt', default='train_op',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

# if torch.cuda.is_available():
#     if args.cuda:
#         torch.set_default_tensor_type('torch.cuda.FloatTensor')
#     if not args.cuda:
#         print("WARNING: It looks like you have a CUDA device, but aren't " +
#               "using CUDA.\nRun with --cuda for optimal training speed.")
#         torch.set_default_tensor_type('torch.FloatTensor')
# else:
#     torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
def train():
    if args.dataset == 'CIFAR10':
        # if args.dataset_root != CIFAR10_ROOT:
        #     if not os.path.exists(CIFAR10_ROOT):
        #         parser.error('Must specify dataset_root if specifying dataset')
        #     print("WARNING: Using default CIFAR10 dataset_root because " +
        #           "--dataset_root was not specified.")
        #     args.dataset_root = CIFAR10_ROOT
        train_data = CIFAR10_train_data
        test_data = CIFAR10_test_data
        cfg = conf.CIFAR10

    elif args.dataset == 'CIFAR100':
        # if args.dataset_root != CIFAR100_ROOT:
        #     if not os.path.exists(CIFAR100_ROOT):
        #         parser.error('Must specify dataset_root if specifying dataset')
        #     print("WARNING: Using CIFAR100 dataset_root because " +
        #           "--dataset was CIFAR100.")
        #     args.dataset_root = CIFAR100_ROOT
        train_data = CIFAR100_train_data
        test_data = CIFAR100_test_data
        cfg = conf.CIFAR100

    elif args.dataset == 'MNIST':
        # if args.dataset_root != MNIST_ROOT:
        #     if not os.path.exists(MNIST_ROOT):
        #         parser.error('Must specify dataset_root if specifying dataset')
        #     print("WARNING: Using MNIST dataset_root because " +
        #           "--dataset was MNIST.")
        #     args.dataset_root = MNIST_ROOT
        # train_data = MNIST_train_data
        # test_data = MNIST_test_data
        cfg = conf.MNIST
    elif args.dataset == 'TinyImageNet':
        # if args.dataset_root != MNIST_ROOT:
        #     if not os.path.exists(MNIST_ROOT):
        #         parser.error('Must specify dataset_root if specifying dataset')
        #     print("WARNING: Using MNIST dataset_root because " +
        #           "--dataset was MNIST.")
        #     args.dataset_root = MNIST_ROOT
        train_data = TinyImageNet_train_data
        test_data = TinyImageNet_test_data
        cfg = conf.TinyImageNet
    elif args.dataset == 'miniImageNet':
        # if args.dataset_root != MNIST_ROOT:
        #     if not os.path.exists(MNIST_ROOT):
        #         parser.error('Must specify dataset_root if specifying dataset')
        #     print("WARNING: Using MNIST dataset_root because " +
        #           "--dataset was MNIST.")
        #     args.dataset_root = MNIST_ROOT
        train_data = miniImageNet_train_data
        test_data = miniImageNet_test_data
        cfg = conf.miniImageNet
    elif args.dataset == 'Facescrubs':
        train_data = Facescrubs_train_data
        test_data = Facescrubs_test_data
        cfg = conf.Facescrubs
    elif args.dataset == 'Imagenet1000':
        train_data = Imagenet1000_train_data
        test_data = Imagenet1000_test_data
        cfg = conf.Imagenet1000
    else:
        print("dataset doesn't exist!")
        exit(0)

    # bulid wideresnet
    utils.PEDCC_PATH = cfg['PEDCC_Type']  # 修改使用的PEDCC文件
    utils.PEDCCgT_PATH = cfg['PEDCCgT_Type']
    # cnn = WideResNet(depth=28, num_classes=cfg['num_classes'], widen_factor=10, feataure_size=cfg['feature_size'])
    # cnn = DenseNet3(depth=100, num_classes=cfg['num_classes'], feature_size=cfg['feature_size'])
    # cnn1 = rResNet(rBottleneck, rBottleneck1, [3, 4, 6, 3], num_classes=cfg['num_classes'], feature_size=cfg['feature_size'])
    # cnn1 = rResNet(rBottleneck, rBottleneck1, [3, 4, 6, 3], num_classes=cfg['num_classes'],
    #                    feature_size=cfg['feature_size'])
    # cnn1 = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=cfg['num_classes'])
    # cnn1 = torch.load("models/TinyImageNet/6_20/_pruned.pkl")
    # 初始化网络参数:
    # cnn1 = init_weights(cnn1)
    # cnn1 = cnn1.module
    # model_path = args.save_folder + args.save_txt + 'pruned.pth'
    # cnn1.load_state_dict(torch.load(model_path))
    # cnn1 = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=cfg['num_classes'])
    # cnn1 = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=cfg['num_classes'], groups=32, width_per_group=4)
    # cnn1 = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=cfg['num_classes'])
    # cnn1 = VisionTransformer(CON FIGS["ViT-B_16"], img_size=32, zero_head=True, num_classes=cfg['num_classes'])
    # cnn1.load_from(np.load("checkpoint/imagenet21k_ViT-B_16.npz"))
    # cnn1 = torchvision.models.resnet50()
    # cnn1 = MobileNetV2(n_class=cfg['num_classes'], input_size=64, width_mult=1.)
    cnn1 = NasNetA(4, 2, 44, 44, class_num=cfg['num_classes'])
    # cnn1 = DenseNet(num_classes=cfg['num_classes'])
    # cnn1 = vgg16_bn(num_class=cfg['num_classes'])
    # cnn1.load_state_dict(torch.load('./models/CIFAR10/softmax+cifar100+resnet50.pth'))
    # PEDCC_loss
    criterion = utils.AMSoftmax(1, 0.05, is_amp=False)#7.5, 0.35    15, 0.5
    criterion1 = nn.MSELoss()
    # criterion2 = nn.NLLLoss()/home/data/ZXW/LpaddL1zghDensenNet/center_pedcc/GausiNew/100_512_s.pkl
    # /home/data/ZXW/LpaddL1zghDensenNet/center_pedcc/GausiNew/100_512T_s.pkl
    criterion2 = nn.CrossEntropyLoss()

    # train_loader = data.DataLoader(dataset, args.batch_size,num_workers=args.num_workers,shuffle=True, collate_fn=detection_collate, pin_memory=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=cfg['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
    # train_loader1 = Data.DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=cfg['batch_size'], shuffle=False, num_workers=6, pin_memory=True)
    # print(cnn1)
    # start training
    # utils.train_soft_mse_zl(cnn1, train_loader, test_loader, cfg, criterion, criterion1, criterion2, args.save_folder, args.save_txt, cfg['num_classes'])
    utils.train_soft_0602(cnn1, train_loader, test_loader, cfg, criterion, criterion1, criterion2, args.save_folder, args.save_txt, cfg['num_classes'])
    # model = cnn1
    # summary(model.cuda(), (3, 32, 32))
    # stat(model, (3, 32, 32))


if __name__ == '__main__':
    train()
