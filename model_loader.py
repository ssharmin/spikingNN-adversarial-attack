import os
import torch
import nn_models as vgg


# map between model name and function
models = {
    'ann'              : vgg.ANN_VGG,
    'snnconv'          : vgg.SNN_VGG,
    'snnbp'            : vgg.SNN_VGG,
}

def load(model_name, batch_size, model_file_name='', snn_thresholds=[20.76, 2.69, 2.33, 0.28, 1.14]):
    """
    Creates an instance of the model, initializes and loads the trained model parameters
    :param model_name: ann, snnconv or snnbp
    :param batch_size: bath size of the input
    :param model_file_name: location and name of the trained parameter file
    """
    if model_name=='ann':
        net = models[model_name](vgg_name = 'VGG5', labels = 10)
    elif model_name == 'snnconv':
        net = models[model_name](batch_size = batch_size, vgg_name = 'VGG5', activation = 'Linear', labels=10, timesteps=2000, leak_mem=1.0)
        net.threshold_init(scaling_threshold=1.0, reset_threshold=0, thresholds = snn_thresholds[:], default_threshold=1.0)    
    elif model_name == 'snnbp':
        net = models[model_name](batch_size = batch_size, vgg_name = 'VGG5', activation = 'Linear', labels=10, timesteps=175, leak_mem=0.99)
        net.threshold_init(scaling_threshold=0.7, reset_threshold=0, thresholds = snn_thresholds[:], default_threshold=1.0)    

    net = torch.nn.DataParallel(net.cuda())
    model_file = model_file_name
    assert os.path.exists(model_file), model_file + " does not exist."
    stored = torch.load(model_file, map_location=lambda storage, loc: storage)
#    torch.save(stored['state_dict'],'snnbp_checkpoint1.pt')
    net.load_state_dict(stored)

    return net
