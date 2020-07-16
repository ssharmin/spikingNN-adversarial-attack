# File : evaluate attacks.py
# Descr: Adversarial example generation from a source model and 
# attacking  a target model

#--------------------------------------------------
# Imports
#--------------------------------------------------
import os
from os.path import expanduser
import torch
import torch.nn as nn
from   torchvision import datasets, transforms
from   torch.utils.data.dataloader import DataLoader
import model_loader
import argparse
import numpy as np
from fgsm import FastGradientSign
from pgd import ProjectedGradientDescent
from utils import random_targets
from evaluate_minibatch import eval_minibatch

###########################################################################
#                           MAIN
###########################################################################
def main():
    #--------------------------------------------------
    # Parse input arguments
    #--------------------------------------------------
    parser = argparse.ArgumentParser(description='Generation and evaluation of VGG-ANN crafted adversarial attack on the CIFAR10 or CIFAR100 dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', default=2, type=int, help='Random seed')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='{CIFAR10, CIFAR100} dataset type')
    parser.add_argument('--arch', default='VGG5', type=str, help='{VGG5, VGG11} network architecture')
    parser.add_argument('--log',  default=None, type=str, help='Log file name')
    #-------------------------------------------------------
    # dataset parameters
    #--------------------------------------------------------
    parser.add_argument('--mean', default='0.5,0.5,0.5', type=str, help='mean of each channel of image in the format mean1,mean2,mean3')
    parser.add_argument('--std',  default='0.5,0.5,0.5',  type=str, help='std of each channel of image in the format std1,std2,std3')
    parser.add_argument('--batch_size',  default=64, type=int, help='Batch size')
    #------------------------------------------------------
    # SNN parameters
    #-------------------------------------------------------
    parser.add_argument('--timesteps', default=100, type=int, help='Total number of time steps')
    parser.add_argument('--leak_mem',  default=0.99, type=float, help='Leak in membrane potential for LIF neuron')
    #-------------------------------------------------------
    # Adversarial attack parameters
    #---------------------------------------------------------
    parser.add_argument('--epsilon', default=8, type=int, help='Attack intensity. Possible values: 2,4,6,8,16')
    parser.add_argument('--eps_iter', default=2, type=int, help='Attack intensity per iteration for PGD attack. Possible values:1,2')
    parser.add_argument('--pgd_steps', default=7, type=int,help='#steps in pgd. Possible values: 7, 10, 20, 40')
    parser.add_argument('--rand_init', default=1, type=int, help='randomly perturb the original image in l-inf ball of epsilon for pgd. Possible values: 1, 0')
    parser.add_argument('--source',  default='ann', type=str, help='source network. Possible values: ann|snnconv|snnbp')
    parser.add_argument('--target',  default='ann', type=str, help='target network. Possible values: ann|snnconv|snnbp')
    parser.add_argument('--attack', default='fgsm', type=str,   help='attack category. Possible values: fgsm|pgd')
    parser.add_argument('--type',  default='wb', type=str, help='Whitebox or Blackbox attack. possible values: wb|bb')
    parser.add_argument('--targeted', default='False', type=str,   help='True|False')

    global args
    args = parser.parse_args()   
    #--------------------------------------------------
    # Initialize seed
    #--------------------------------------------------
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #-----------------------------------------------------
    # dataset and architecture
    #-----------------------------------------------------
    dataset         = args.dataset
    architecture    = args.arch
    #-----------------------------------------------------
    # batch size 
    #----------------------------------------------------
    batch_size      = args.batch_size
    #-----------------------------------------------------
    # Adversarial attack parameters
    #-----------------------------------------------------
    model_source_name = args.source
    model_target_name = args.target    
    attack = args.attack
    targeted = args.targeted
    epsilon = args.epsilon
    eps_iter = args.eps_iter
    num_iter = args.pgd_steps
    rand_init = args.rand_init
    attack_type = args.type
    clip_min = 0.0
    clip_max = 1.0    
    #---------------------------------------------------
    # log file
    #----------------------------------------------------
    logfile_name = dataset.lower()+architecture.lower()+'_'+model_source_name+'_crafted_on_'+model_target_name+'_'+attack_type+'_'+attack+'.log'
    f = open(logfile_name, 'a', buffering=1)
    #--------------------------------------------------
    # Load the CIFAR10 dataset
    #--------------------------------------------------
    data_root     = expanduser("~")
    # These are the mean and std values used to train the models
    args.mean1, args.mean2, args.mean3 = [float(a) for a in args.mean.split(',')]  
    args.std1, args.std2, args.std3 = [float(a) for a in args.std.split(',')]  
    mean = [args.mean1, args.mean2, args.mean3]
    std = [args.std1, args.std2, args.std3]        
    transform_test = transforms.Compose([transforms.ToTensor()])    
    if dataset == 'CIFAR10':
        testset     = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    testloader  = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)    
    #--------------------------------------------------
    # Instantiate the Source & Target model 
    #--------------------------------------------------
    model_dir = os.getcwd()
    # map between model name and model_file directory
    model_source_dir = {
        'ann_wb'         : model_dir+'/ann_checkpoint1.pt',
        'ann_bb'         : model_dir+'/ann_checkpoint2.pt',
        'snnconv_wb'     : model_dir+'/ann_checkpoint1.pt',
        'snnconv_bb'     : model_dir+'/ann_checkpoint2.pt',
        'snnbp_wb'       : model_dir+'/snnbp_checkpoint1.pt',
        'snnbp_bb'       : model_dir+'/snnbp_checkpoint2.pt'
    }
    # The threshold values of SNN models. Two sets of thresholds corresponding to two separately initialized models
    snn_thresholds= [[20.76, 2.69, 2.33, 0.28, 1.14],[20.86, 2.74, 2.21, 0.38, 0.99]]# VGG5 
    # wb: whitebox, bb: blackbox
    if attack_type=='wb':
        threshold_id = 0
    elif attack_type=='bb':
        threshold_id = 1
    model_source = model_loader.load(model_source_name, batch_size,model_source_dir[model_source_name+'_'+attack_type], snn_thresholds[threshold_id][0:])
    model_target = model_loader.load(model_target_name, batch_size, model_source_dir[model_target_name+'_wb'], snn_thresholds[0][0:])
    # Used during analyzing the variation of number of timesteps and leak factor
#    if model_target.module.type == 'SNN':
#        model_target.module.timesteps_init(args.timesteps)
#    if model_target_name == 'snnbp':
#        model_target.module.leak_init(args.leak_mem)
    #--------------------------------------------------
    # Initialize the loss function 
    #--------------------------------------------------
    loss_func = nn.CrossEntropyLoss().cuda()
    
    # Print the simulation parameters
    print('Batch size : {}'.format(batch_size))
    print('{}-crafted attack on {}:{}'.format(model_source_name, model_target_name, attack_type))
    print('Attack category: {}'.format(attack))
    print('epsilon:{}|targeted:{}'.format(epsilon, targeted))
    if attack=='pgd':
        print('epsilon_iter:{}|num_iter:{}'.format(eps_iter, num_iter))
    print('*********************************************************')
    
    #---------------------------------------------------------------------
    # Adversarial attack generation and evaluation
    #--------------------------------------------------------------------
    count_clean = 0 # keeps track of the correct predictions for clean input
    count_adv = 0 # keeps track of the correct predictions for the adversarial input
    # Instantiate the attack model
    if attack =='fgsm':
        attack_model = FastGradientSign(epsilon*1.0/255.0, clip_min, clip_max, targeted)
    elif attack == 'pgd':
        attack_model = ProjectedGradientDescent(num_iter, epsilon*1.0/255.0, eps_iter*1.0/255.0, clip_min, clip_max, targeted, rand_init, seed)
    for batch_id, (images, labels) in enumerate(testloader):
        images, labels = images.cuda(), labels.cuda()
        # Calculate the clean test accuracy of the target model
        acc,loss = eval_minibatch(model_target, images, labels, mean, std, loss_func)
        count_clean +=acc
        # create target label in case of 'targeted' attack
        if targeted=='False':
            target_labels = labels
        else:
            target_labels = random_targets(10, labels.cpu(), seed)
        target_labels = target_labels.cuda()
        # Generate adversarial example
        images_adv = attack_model.generate(images,labels, model_source, loss_func, mean, std)             
        # Calculate the adv test accuracy of the target model
        acc,loss = eval_minibatch(model_target, images_adv, target_labels, mean, std, loss_func)
        count_adv +=acc
        # Print the clean and adv count 
        if (batch_id!=0) and (batch_id%2==0):      
            print('Minibatch : {}'.format(batch_id))
            print('clean/adv: {}/{}\n'.format(int(count_clean), int(count_adv)))
    # Adversarial loss = precision_clean - precision_adv
    precision_adv = float(count_adv)*100/len(testloader.dataset)
    precision_clean = float(count_clean)*100/len(testloader.dataset)
    print('Adversarial Loss: {:.3f}%\n'.format(precision_clean-precision_adv))
    if attack=='pgd':
        f.write("\t {}:{}:{}|targeted:{}|eps_iter:{}, num_iter:{}--> adv loss = {:.3f}%\n".format(model_source_name, model_target_name, attack_type, targeted, eps_iter, num_iter,precision_clean-precision_adv))
    elif attack=='fgsm':
        f.write("\t {}:{}:{}|targeted:{}|epsilon:{}--> adv loss = {:.3f}%\n".format(model_source_name, model_target_name, attack_type, targeted, epsilon,precision_clean-precision_adv))
    f.close()
        

if __name__ == '__main__':
    main()
