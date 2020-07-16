import torch
import numpy as np

def Accuracy(output, labels, topk=1):
    _, pred = output.topk(topk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    return correct[:1].view(-1).float().sum(0, keepdim=True)

def other_classes(num_classes, class_id):
    """
    Returns  a list of classes in [0,...num_classes-1]
    excluding the class_id
    """
    if class_id>=num_classes or class_id<0:
        raise ValueError("class_id must be within [0,...,num_classes-1]")
    results = list(range(num_classes))
    results.remove(class_id)
    return results
    
def random_targets(num_classes, correct_labels, seed):
    """
    Create a list of random labels which are different from correct_labels.
    This is used as target during the loss calculation of
    targeted adversarial attacks.
    :param correct_labels: the correct labels, 1-by-N array.
    :param num_classes: number of classes
    :return: 1-by-N array.
    """
    np.random.seed(seed)
    if len(correct_labels.shape)>1:
        raise ValueError("correct_labels must be a 1-D vecotr")
    result = np.zeros(correct_labels.shape, dtype=np.int)
    class_id = 0
    for class_id in range(num_classes):
        #Find labels with class_id
        labels_class_id = np.multiply((correct_labels==class_id),1) 
        num_labels = torch.sum(labels_class_id)
        possible_targets = other_classes(num_classes, class_id)

        if num_labels>0:
            result[correct_labels==class_id]= np.random.choice(possible_targets, num_labels.numpy())
    return torch.from_numpy(result)

def normalization_function(x, mean, std):
    """
    Normalizes input variable with a mean and std.
    output = (input-mean)/std
    :param x: input data, a 3-channel image
    :param mean: mean of all the input channels, a list of 3 floats
    :param std: standard deviation of all the input channels, a list of 3 floats
    """
    assert len(mean) == 3, 'Custom norm function is for 3 channel images. Expected 3 elements for mean, got {}'.format(len(mean))
    assert len(std) == 3, 'Custom norm function is for 3 channel images. Expected 3 elements for std, got {}'.format(len(std))
    img_dims = x.size()[1:] # 1st dimension is batchsize   
    mean_expanded = torch.cat((torch.ones((1, img_dims[1], img_dims[2]))*mean[0],
                                    torch.ones((1, img_dims[1], img_dims[2]))*mean[1],
                                    torch.ones((1, img_dims[1], img_dims[2]))*mean[2]
                                    ), dim = 0).cuda()    
    std_expanded = torch.cat((torch.ones((1, img_dims[1], img_dims[2]))*std[0],
                                   torch.ones((1, img_dims[1], img_dims[2]))*std[1],
                                   torch.ones((1, img_dims[1], img_dims[2]))*std[2]
                                   ), dim = 0).cuda()
    normalized_tensor = x.sub(mean_expanded.expand_as(x)).div(std_expanded.expand_as(x))
    return normalized_tensor

