import torch
from utils import normalization_function
from input_gradient_calculation import inp_grad_calc

class FastGradientSign:
    """
    The Fast Gradient Sign Method for adversarial attack
    This method was introduced by goodfellow et. al.
    Paper link: https://arxiv.org/abs/1412.6572
    """
    def __init__(self, epsilon, clip_min, clip_max, targeted):
        """
        Initialize the attack parameters
        """
        super(FastGradientSign, self).__init__()
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
    
    def generate(self, x, labels, model, loss_func, mean, std):
        """
        returns the adversarial output = x +/- epsilon*sign(input_grad)
        
        x: Inputs to the model. x lies in the range [0,1]
        labels: Correct labels corresponding to x (untargeted) or the target label (targeted attacks)
        model: the source model for generating adversary
        loss_func: Used to calculate the error
        """
        # We need to normalize the input with mean and std, which was used to train the model
        x_norm = normalization_function(x, mean, std)
        inp_grad = inp_grad_calc(x_norm, labels, model, loss_func) # Calculate the gradient of loss w.r.t. input
        if self.targeted == 'True':
            inp_grad = -1*inp_grad
        x += self.epsilon * torch.sign(inp_grad)
        x = torch.clamp(x, self.clip_min, self.clip_max)
        return x
        
        
    
        
            
        