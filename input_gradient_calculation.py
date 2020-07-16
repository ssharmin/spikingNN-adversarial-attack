import torch
import numpy as np
import torch.nn.functional as F

def inp_grad_calc(x, labels, model, loss_func):
        """
        Calculates the gradient of error with respect to input (del_loss/del_input)
        :param x: input
        :param labels: The labels used to calculate the loss or error
        :param model: Source model
        :loss_func: Loss function used to calculate the error or loss
        """
        inputs = x.clone().detach()
        inputs.requires_grad_(True)
        inputs.grad = None       
        if model.module.type=='SNN':
            #----------------------------------------------------------------------------
            # preprocessing: conv1 weight matrix rotated by 180
            ############################################################################
            out_channel = model.module.features[0].weight.size()[0] # 64
            in_channel = model.module.features[0].weight.size()[1] # 3
            weight_new = np.zeros((model.module.features[0].weight.size()))
            for i in range(out_channel):
                for j in range(in_channel):
                    weight_new[i,j,0,0:] = np.flip(model.module.features[0].weight.detach().cpu().numpy()[i,j,2,0:])
                    weight_new[i,j,1,0:] = np.flip(model.module.features[0].weight.detach().cpu().numpy()[i,j,1,0:])
                    weight_new[i,j,2,0:] = np.flip(model.module.features[0].weight.detach().cpu().numpy()[i,j,0,0:])
            weight_new = np.transpose(weight_new,(1,0,2,3))
            weight_rotate = torch.from_numpy(weight_new).float().cuda()
            
            output, input_spike_count, grad_mem_conv1 = model(inputs,0, True)
            output = output/model.module.timesteps
        elif model.module.type == 'ANN':
            output = model(inputs)
            
        error = loss_func(output, labels)
        error.backward()        
        if model.module.type=='SNN':
            grad_mem_conv1 = model.module.grad_mem1
            # del_loss/del_input = conv(del_loss/del_conv1, W_conv1(180rotated))
            inp_grad = F.conv2d(grad_mem_conv1, weight_rotate, padding=1)
        elif model.module.type=='ANN':
            inp_grad = inputs.grad.data
        # reset        
        inputs.requires_grad_(False)
        inputs.grad = None
        model.module.zero_grad()
        return inp_grad
