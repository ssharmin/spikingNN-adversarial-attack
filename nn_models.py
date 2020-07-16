# File : nn_models.py
# Descr: Define ANN and SNN models 

#--------------------------------------------------
# Imports
#--------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import copy

cfg = {
    'VGG5' : [64, 'A', 128, 'D', 128, 'A'],
    'VGG11': [64, 'A', 128, 'A', 256, 'D', 256, 'A', 512, 'D', 512, 'A', 512, 'D', 512, 'A'],
    'VGG16': [64, 'D', 64, 'A', 128, 'D', 128, 'A', 256, 'D', 256, 'D', 256, 'A', 512, 'D', 512, 'D', 512, 'A', 512, 'D', 512, 'D', 512, 'A']
}

#--------------------------------------------------
# Define the VGG-ANN model
#--------------------------------------------------
class ANN_VGG(nn.Module):
    def __init__(self, vgg_name='VGG5', labels=10, dropout=0):
        super().__init__()
        self.type = 'ANN'
        self.vgg_name = vgg_name
        self.labels   = labels
        self.dropout = dropout # During, dropout=0 (no dropout) is used        
        self.features, self.classifier = self._make_layers(cfg[self.vgg_name])
        self.init_weight()

    def _make_layers(self, cfg):
        layers  = []
        in_channels = 3
        for x in (cfg):                       
            if x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif x == 'D':
                layers += [nn.Dropout(self.dropout)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=1, bias=False),
                            nn.ReLU(inplace=True)
                            ]
                in_channels = x        
        features = nn.Sequential(*layers)        
        layers = []
        layers += [nn.Linear(8192, 1024, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(0)]
        layers += [nn.Linear(1024, 1024, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(0)]
        layers += [nn.Linear(1024, self.labels, bias=False)]
        classifier = nn.Sequential(*layers)
        return (features, classifier)   
    
    def init_weight(self):
        for pos in range(len(self.features)):
            if isinstance(self.features[pos], nn.Conv2d):
                n1 = self.features[pos].kernel_size[0] * self.features[pos].kernel_size[1] * self.features[pos].in_channels
                variance1 = math.sqrt(2. / (n1)) 
                self.features[pos].weight.data.normal_(0, variance1)
                # define threshold
                self.features[pos].threshold = 1 
        for pos in range(len(self.classifier)-1):
            if isinstance(self.classifier[pos], nn.Linear):
                size = self.classifier[pos].weight.size()
                fan_in = size[1]  # number of columns
                variance2 = math.sqrt(2.0 / (fan_in)) 
                self.classifier[pos].weight.data.normal_(0.0, variance2)
                # define threshold
                self.classifier[pos].threshold = 1 
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#--------------------------------------------------
# Spiking neuron with fast-sigmoid surrogate gradient
# This class is replicated from:
# https://github.com/fzenke/spytorch/blob/master/notebooks/SpyTorchTutorial2.ipynb
#--------------------------------------------------

#--------------------------------------------------
# Spiking neuron with piecewise-linear surrogate gradient
#--------------------------------------------------
class LinearSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3 # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass, we compute a step function of the input Tensor and
        return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use
        the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we receive a Tensor we need to compute
        the surrogate gradient of the loss with respect to the input.
        Here we use the piecewise-linear surrogate gradient as was
        done in Bellec et al. (2018).
        """
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad       = grad_input*LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad

#--------------------------------------------------
# Spiking neuron with exponential surrogate gradient
#--------------------------------------------------
class ExpSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the exponential surrogate gradient as was done in
    Shrestha et al. (2018).
    """
    alpha = 0.3 # Controls the dampening of the exponential surrogate gradient
    beta  = 1.0 # Controls the steepness of the exponential surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass, we compute a step function of the input Tensor and
        return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use
        the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we receive a Tensor we need to compute
        the surrogate gradient of the loss with respect to the input.
        Here we use the exponential surrogate gradient as was done
        in Shrestha et al. (2018).
        """
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad       = grad_input*ExpSpike.alpha*torch.exp(-ExpSpike.beta*torch.abs(input))
        return grad

class STDPSpike(torch.autograd.Function):

    alpha     = ''
    beta     = ''
    
    @staticmethod
    def forward(ctx, input, last_spike):
        
        ctx.save_for_backward(last_spike)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
                
        last_spike, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = STDPSpike.alpha * torch.exp(-1*last_spike)**STDPSpike.beta
        return grad*grad_input, None
# Overwrite the native spike function by differentiable spiking nonlinearity which implements a surrogate gradient
def init_spike_fn(grad_type):
    if(grad_type == 'Linear'):
       spike_fn = LinearSpike.apply
    elif(grad_type == 'Exp'):
       spike_fn = ExpSpike.apply
    else:
       sys.exit("Unknown gradient type '{}'".format(grad_type))
    return spike_fn

#--------------------------------------------------
# Poisson spike generator
#   Positive spike is generated (i.e.  1 is returned) if rand()<=abs(input) and sign(input)= 1
#   Negative spike is generated (i.e. -1 is returned) if rand()<=abs(input) and sign(input)=-1
#--------------------------------------------------
class PoissonGenerator(nn.Module):    
    def __init__(self):
        super().__init__()
    def forward(self,input, inp_rate=1.0):        
        out = torch.mul(torch.le(torch.rand_like(input), torch.abs(input)*inp_rate).float(),torch.sign(input))
        return out
    
class SNN_VGG(nn.Module):
    def __init__(self, batch_size, vgg_name, activation='Linear', labels=10, timesteps=175, leak_mem=0.99, STDP_alpha=0.3, STDP_beta=0.01, drop=0, inp_rate=1.0):
        """
        :param batch_size: bath size of the input images
        :param vgg_name: vgg architecture name (e.g. VGG5)
        :param activation: the type of approximation used for surrogate gradient calculatio
        :param labels: number of output classes. 10 for CIFAR10
        :param timesteps: number of total timesteps
        :param leak_mem: leak factor, lambda. The value is 1 for IF neuron
        :param STDP_alpha, STDP_beta: hyper parameters used in the STDB surrogate gradient approximation
        :param drop: dropout rate used during training. During inference, drop=0
        :param inp_rate: This parameter controls the spike generation rate in the PoissonGenerator
        """
        super().__init__()
        self.type = 'SNN'
        STDPSpike.alpha = STDP_alpha
        STDPSpike.beta  = STDP_beta 
        self.timesteps = timesteps
        self.vgg_name  = vgg_name
        self.labels = labels
        self.leak_mem = leak_mem
        if activation == 'Linear':
            self.act_func = LinearSpike.apply
        elif activation == 'STDB': # Spike Time Dependent Backprop
            self.act_func = STDPSpike.apply
        self.input_layer = PoissonGenerator()
        self.batch_size = batch_size
        self.inp_rate = inp_rate        
        self.features, self.classifier = self._make_layers(cfg[self.vgg_name])
        # This variable stores the gradient of the membrane potential at 1st conv layer
        # (Used for adversarial attack generation)
        self.grad_mem1 = torch.zeros(self.batch_size, self.features[0].out_channels, 32, 32).cuda()
        
    def threshold_init(self, scaling_threshold=1.0, reset_threshold=0.0, thresholds=[], default_threshold=1.0):
        # Initialize thresholds
        self.scaling_threshold     = scaling_threshold
        self.reset_threshold     = reset_threshold        
        self.threshold             = {}      
        for pos in range(len(self.features)):
            if isinstance(self.features[pos], nn.Conv2d):
                self.threshold[pos] = round(thresholds.pop(0) * self.scaling_threshold  + self.reset_threshold * default_threshold, 2)
        prev = len(self.features)
        for pos in range(len(self.classifier)-1):
            if isinstance(self.classifier[pos], nn.Linear):
                self.threshold[prev+pos] = round(thresholds.pop(0) * self.scaling_threshold  + self.reset_threshold * default_threshold, 2)
        return self.threshold
        
    def counting_spikes(self, cur_time, layer, spikes):
        self.spike_count

    def _make_layers(self, cfg):
        layers         = []
        in_channels = 3
        for x in (cfg):
            stride = 1                        
            if x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif x == 'D':
                layers += [nn.Dropout(0.2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=stride, bias=False),
                            nn.ReLU(inplace=True)
                            ]
                in_channels = x        
        features = nn.Sequential(*layers)        
        layers = []
        layers += [nn.Linear(8192, 1024, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(0)]
        layers += [nn.Linear(1024, 1024, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(0)]
        layers += [nn.Linear(1024, self.labels, bias=False)]

        classifer = nn.Sequential(*layers)
        return (features, classifer)


    def timesteps_init(self, timesteps):
        self.timesteps = timesteps
        
    def leak_init(self, leak_mem):
        self.leak_mem = leak_mem

    def neuron_init(self, x):
        self.batch_size = x.size(0)
        self.width = x.size(2)
        self.height = x.size(3)                
        self.mem = {}
        self.spike = {}
        self.mask = {}
        self.spike_count = {}
        self.grad_mem1 = torch.zeros(self.batch_size, self.features[0].out_channels, self.width, self.height).cuda()
        for l in range(len(self.features)):                                
            if isinstance(self.features[l], nn.Conv2d):
                self.mem[l] = torch.zeros(self.batch_size, self.features[l].out_channels, self.width, self.height).cuda()
                self.spike_count[l] = torch.zeros(self.mem[l].size()).cuda()
            elif isinstance(self.features[l], nn.Dropout):
                self.mask[l] = self.features[l](torch.ones(self.mem[l-2].shape)).cuda()
            elif isinstance(self.features[l], nn.AvgPool2d):
                self.width = self.width//2
                self.height = self.height//2        
        prev = len(self.features)
        for l in range(len(self.classifier)):            
            if isinstance(self.classifier[l], nn.Linear):
                self.mem[prev+l]             = torch.zeros(self.batch_size, self.classifier[l].out_features).cuda()
                self.spike_count[prev+l]     = torch.zeros(self.mem[prev+l].size()).cuda()
            elif isinstance(self.classifier[l], nn.Dropout):
                self.mask[prev+l] = self.classifier[l](torch.ones(self.mem[prev+l-2].shape)).cuda()                
        self.spike = copy.deepcopy(self.mem)
        for key, values in self.spike.items():
            for value in values:
                value.fill_(-1000)
                
    def variable_hook(self,grad):
        """
        Accumulates the gradient of the mem_pot of 1st conv layer during backprop
        at each timestep. grad_mem1 is used to store the data.
        """
        self.grad_mem1 +=grad
        
    def forward(self, x, cur_time, register_hook=False):        
        if cur_time == 0:
            self.neuron_init(x)

        for t in range(cur_time, cur_time+self.timesteps):            
            out_prev = self.input_layer(x, self.inp_rate)              
            for l in range(len(self.features)):                
                if isinstance(self.features[l], (nn.Conv2d)):
                    mem_thr                     = (self.mem[l]/self.threshold[l]) - 1.0
                    out                         = self.act_func(mem_thr)
                    rst                         = self.threshold[l] * (mem_thr>0).float()                    
                    self.mem[l]     = self.leak_mem*self.mem[l] + self.features[l](out_prev) - rst
                    out_prev          = out.clone()
                    if register_hook and l==0:
                        self.mem[l].register_hook(self.variable_hook)
                elif isinstance(self.features[l], nn.AvgPool2d):
                    out_prev         = self.features[l](out_prev)                
                elif isinstance(self.features[l], nn.Dropout):
                    out_prev         = out_prev * self.mask[l]            
            out_prev           = out_prev.reshape(self.batch_size, -1)
            prev = len(self.features)
            
            for l in range(len(self.classifier)-1):                                                    
                if isinstance(self.classifier[l], (nn.Linear)):
                    mem_thr                     = (self.mem[prev+l]/self.threshold[prev+l]) - 1.0
                    out                         = self.act_func(mem_thr)
                    rst                         = self.threshold[prev+l] * (mem_thr>0).float()
                    self.mem[prev+l]     = self.leak_mem*self.mem[prev+l] + self.classifier[l](out_prev) - rst
                    out_prev          = out.clone()

                elif isinstance(self.classifier[l], nn.Dropout):
                    out_prev         = out_prev * self.mask[prev+l]
            # Compute the classification layer outputs
            self.mem[prev+l+1]         = self.mem[prev+l+1] + self.classifier[l+1](out_prev)
        return self.mem[prev+l+1], self.mem[0], self.grad_mem1
    


#--------------------------------------------------
# Define a class for recording the SNN train/test loss.
#--------------------------------------------------
class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count
        
def Accuracy(output, labels, topk=1):
    _, pred = output.topk(topk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    return correct[:1].view(-1).float().sum(0, keepdim=True)

