import torch
from utils import Accuracy, normalization_function

def eval_minibatch(model, images, labels, mean, std, loss_func):
    """
    Calculates the loss and accuracy of the model over a minibatch.
    :param model: the model object
    :param image: minibatch images,which lie in the range [0,1]. Need to normalize with 
     mean,std (which was used for training) before feeding into model
    :param mean: mean of all the channels of the images, a list of 3 floats
    :param std: standard deviation of all the channels of the images, a list of 3 floats
    :param loss_func: The loss_func object used to calculate the error
    """
    model.eval()
    count = 0
    # input images are normalized to [0,1]. After normalization with 
    # mean(per channel)=0.5, std(per channel)=0.5, x_norm lies in the range [-1,1]
    x_norm = normalization_function(images, mean, std) 
    with torch.no_grad():
        if model.module.type=='SNN':
            output, _, _ = model(x_norm, 0, False)
            output = output/model.module.timesteps
        elif model.module.type =='ANN':
            output = model(x_norm)        
    count = Accuracy(output, labels)
    loss = loss_func(output, labels)
    return count, loss