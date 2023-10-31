# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# different DATA set configs
CIFAR10 = {
    'num_classes':10,#10
    'max_epoch':150,  # 100
    'LR':1e-1,  # 1e-1
    'lr_steps':(0,50,80,120),  # (0,30,60,90)
    'AvgPool_InputSize':8,
    'PEDCC_Type':'center_pedcc/10_256_s.pkl',
    'PEDCCgT_Type':'center_pedcc/10_256T_s.pkl',
    'feature_size':256,
    'batch_size':128,
    'name':'CIFAR10'
}
CIFAR100 = {
    'num_classes': 100,
    'max_epoch': 150, #150
    'LR':1e-1,
    'lr_steps': (0,50,80,120),   #(0, 50, 100, 150) (0,100,150) (0,40,75,105) (0,50,80,120)
    'AvgPool_InputSize':8,
    'PEDCC_Type':'center_pedcc/100_512_s.pkl',
    'PEDCCgT_Type':'center_pedcc/100_512T_s.pkl',
    'feature_size':512,
    'batch_size':128, #128
    'name':'CIFAR10'
}
MNIST = {
    'num_classes':10,
    'max_epoch':100,
    'LR':1e-3,
    'lr_steps':(0,30,60,90),
    'AvgPool_InputSize':7,
    'PEDCC_Type':'center_pedcc/10_100.pkl',
    'feature_size':100,
    'batch_size':32,
    'name':'CIFAR10'
}
TinyImageNet = {
    'num_classes':200,
    'max_epoch':150,#100
    'LR':1e-1,
    'lr_steps':(0,50,80,120), #(0,30,60,90),
    'AvgPool_InputSize':8,
    'PEDCC_Type':'center_pedcc/200_512_s.pkl',
    'PEDCCgT_Type':'center_pedcc/200_512T_s.pkl',
    'feature_size':512,
    'batch_size':64,# 256
    'name':'CIFAR10'
}
Facescrubs = {
    'num_classes':100,
    'max_epoch':100,
    'LR':1e-1,
    'lr_steps':(0,30,60,90),
    'AvgPool_InputSize':8,
    'PEDCC_Type':'center_pedcc/100_512_s.pkl',
    'PEDCCgT_Type':'center_pedcc/100_512T_s.pkl',
    'feature_size':512,
    'batch_size':128, # 128
    'name':'CIFAR10'
}
miniImageNet= {
    'num_classes':100,
    'max_epoch':100,
    'LR':1e-1,
    'lr_steps':(0,30,60,90),
    'AvgPool_InputSize':8,
    'PEDCC_Type':'center_pedcc/100_512_s.pkl',
    'PEDCCgT_Type':'center_pedcc/100_512T_s.pkl',
    'feature_size':512,
    'batch_size':100,
    'name':'CIFAR10'
}
Imagenet1000 = {
    'num_classes':200,
    'max_epoch':100,
    'LR':1e-1,
    'lr_steps':(0, 30, 60, 90),
    'AvgPool_InputSize':8,
    'PEDCC_Type':'center_pedcc/200_512_s.pkl',
    'PEDCCgT_Type':'center_pedcc/200_512T_s.pkl',
    'feature_size':512,
    'batch_size':100,
    'name':'CIFAR10'
}
