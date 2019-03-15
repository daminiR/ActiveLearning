#nvidia-smi #which gpu is free to use
#CUDA_VISIBLE_DEVICE = 1 nohup python something.py > output &
# scipy entropy => https://github.com/scipy/scipy/blob/v1.1.0/scipy/stats/_distn_infrastructure.py#L2478-L2519
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch

def _KL(p, q):
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
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))




def calculate_KL_batch(batch):
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
    KL_scores = [0] * batch.size()[0]
    for i in range(batch.size()[0]):
        running_sum = 0
        for j in range(batch.size()[0]):
            running_sum += _KL(batch[i], batch[j])
        KL_scores[i] = running_sum

    return KL_scores


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

    batch = torch.rand((3, 5))
    print(batch)
    print(calculate_KL_batch(batch))
