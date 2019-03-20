#nvidia-smi #which gpu is free to use
#CUDA_VISIBLE_DEVICE = 1 nohup python something.py > output &
# scipy entropy => https://github.com/scipy/scipy/blob/v1.1.0/scipy/stats/_distn_infrastructure.py#L2478-L2519
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch

def _KL(p, q, device):
    """Calculate the Kullback-Leibler divergens
     between probability distribution p and q.
    Parameters
    ----------
    p : vector like array
        (n x 1) array containing porbability values.
    q : vector like array
        (n x 1) array containing porbability values.

    Returns
    -------
    scalar
        KL divergence value

    """

    #convert to array
    #p = p.detach().cpu().numpy()
    #q = q.detach().cpu().numpy()
    zero = torch.zeros(1)
    zero = zero.to(device)
    return torch.sum(torch.where(q != 0, p * torch.log2(p / q), zero))
    #return np.sum(np.where(q != 0, p * np.log(p / q), 0))




def calculate_KL_batch(batch, batch_size, device = "cpu"):
    """Calculates the KL divergence scores for each image in a batch.

    Parameters
    ----------
    batch : torch.Tensor
        (n x C) tensor of the outputs of the model for n instances

    Returns
    -------
    torch.tensor
        (n x 1) tensor of KL scores for each image within the batch

    """
    #print(torch.max(torch.stack(batch)))
    # Numpy (CPU)
    # Calculate KL divergence values between every instance
    KL_scores = np.zeros((len(batch), len(batch)))
    for i in range(len(batch)):
        for j in range((i + 1), len(batch)):
            running_sum = _KL(batch[i], batch[j], device) + _KL(batch[j], batch[i], device)
            KL_scores[i][j] = running_sum
            KL_scores[j][i] = running_sum
    # print(KL_scores)

    # Make the first greedy choice by taking the max from KL_scores
    ind1, ind2 = np.unravel_index(np.argmax(KL_scores, axis=None), KL_scores.shape)

    # Intitialize index and KL_with_chosen_sum for the first two indices
    index = [ind1, ind2]
    KL_with_chosen_sum = KL_scores[ind1]

    # Only need batch_size amount of instances to be queried
    while(len(index) != batch_size):
        # Update KL_with_chosen_sum
        KL_with_chosen_sum += KL_scores[ind2]
        for ind in index:
            KL_with_chosen_sum[ind] = 0
        
        # Find the next index of instances to be queried and update it
        ind2 = np.argmax(KL_with_chosen_sum)
        index.append(ind2)

    return index

def smoothed_hist_kl_distance(a, b, nbins=10, sigma=1):
    #compute histogram of a set of data
    ahist, bhist = (np.histogram(a, bins=nbins)[0],
                    np.histogram(b, bins=nbins)[0])
    #multi-dimensional guassian filter
    asmooth, bsmooth = (gaussian_filter(ahist, sigma),
                        gaussian_filter(bhist, sigma))

    return kl(asmooth, bsmooth)

if __name__ == "__main__":
    # a = np.linspace(-10.0, 10.0, 1000)
    # b = np.linspace(0, 30.0, 300 )
    # v=[[1,1,1,1,1,1,1],[1,2,1,1,1,2,1],[1,1,1,1,1,1,1]]
    # c=v[0]
    # list_of_batches = []
    # for x in v:
    #     print(kl(x,c))
    #     #if some condition:
    #         #append the input to the list
    # print(smoothed_hist_kl_distance(a,b))

    batch = torch.rand((5, 5))
    print(batch)
    print(calculate_KL_batch(batch, 4))
