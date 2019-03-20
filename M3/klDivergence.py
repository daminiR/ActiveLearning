#nvidia-smi #which gpu is free to use
#CUDA_VISIBLE_DEVICE = 1 nohup python something.py > output & 

# scipy entropy => https://github.com/scipy/scipy/blob/v1.1.0/scipy/stats/_distn_infrastructure.py#L2478-L2519
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def kl(p, q):
    #convert to array
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def smoothed_hist_kl_distance(a, b, nbins=10, sigma=1):
    #compute histogram of a set of data
    ahist, bhist = (np.histogram(a, bins=nbins)[0],
                    np.histogram(b, bins=nbins)[0])
    #multi-dimensional guassian filter
    asmooth, bsmooth = (gaussian_filter(ahist, sigma),
                        gaussian_filter(bhist, sigma))

    return kl(asmooth, bsmooth)

if __name__ == "__main__":
    a = np.linspace(-10.0, 10.0, 1000)
    b = np.linspace(0, 30.0, 300 )
    v=[[1,1,1,1,1,1,1],[1,2,1,1,1,2,1],[1,1,1,1,1,1,1]]
    c=v[0]
    list_of_batches = []
    for x in v:
        print(kl(x,c))
        #if some condition:
            #append the input to the list
    print(smoothed_hist_kl_distance(a,b))
    