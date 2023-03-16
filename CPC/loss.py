import torch

def InfoNCELoss(pred, encode_samples):
    '''
    pred : linear predictions (c_t)
    - shape : (timestep, batch_size, hidden_dim) = (12, 8, 512)

    encode_samples : localized representations (z_{t+k})
    - shape : (timestep, batch_size, hidden_dim) = (12, 8, 512)
    '''

    K, B, H = pred.shape

    nce = 0.
    for k in range(K):
        # dot product between z_{t+k} and (W_k * c_t)
        # --> similarity[i,j] = (if i == j) similarity with positive sample
        #                       (otherwise) similarity with negative sample
        similarity = torch.mm(encode_samples[k], pred[k].T) # shape : (batch_size, batch_size) = (8, 8)

        # InfoNCE loss for k-timestep prediction
        nce += torch.sum(torch.diag(torch.log_softmax(similarity, dim = 0)))

    nce /= (-1. * B * K)

    return nce