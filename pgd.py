import torch
from utils import normalization_function
from input_gradient_calculation import inp_grad_calc

class ProjectedGradientDescent:
    """
    The Projected Gradient Descent Method for adversarial attack
    This method was introduced by Madry et. al.
    Paper link: https://arxiv.org/pdf/1706.06083.pdf
    """
    def __init__(self, num_iter, epsilon, eps_iter, clip_min, clip_max, targeted, rand_init, seed):
        """
        Initialize the attack parameters
        :param num_iter: number of iterations for PGD
        :param epsilon: attack strength
        :param eps_iter: attack strength per iteration
        :param clip_min, clip_max: clip the adversarial input between clip_min and clip_max
        :param targeted: True or False
        :param rand_init: 0 or 1. If 1, adds a random perturbation with strength epsilon
        to the input before performing PGD operation
        :param seed: seed used for random number generator
        """
        super(ProjectedGradientDescent, self).__init__()
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.num_iter = num_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.seed = seed
        
            
    def generate(self, x, labels, model, loss_func, mean, std):
        """
        returns the pgd adversarial output 
        
        x: The inputs to the model, x lies in the range [0,1]
        labels: Correct labels corresponding to x (untargeted) or the target label (targeted attacks)
        model: the source model for generating adversary
        loss_func: Used to calculate the error
        """
        x_orig = x.clone()
        torch.manual_seed(self.seed)
        if self.rand_init:
            x = x + (2*self.epsilon)*torch.rand_like(x) - self.epsilon
            x = torch.clamp(x, self.clip_min, self.clip_max)
        for i in range(self.num_iter):
            # We need to normalize the input with mean and std, which was used to train the model
            x_norm = normalization_function(x, mean, std)
            inp_grad = inp_grad_calc(x_norm, labels, model, loss_func) # Calculate the gradient of loss w.r.t. input
            if self.targeted == 'True':
                inp_grad = -1*inp_grad
            x += self.eps_iter * torch.sign(inp_grad)
            eta = torch.clamp(x-x_orig, min=-self.epsilon, max=self.epsilon)
            x = torch.clamp(x_orig+eta, self.clip_min, self.clip_max)                
        return x
        
        
    
        
            
        