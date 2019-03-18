#nvidia-smi #which gpu is free to use
#CUDA_VISIBLE_DEVICE = 1 nohup python something.py > output & 

# scipy entropy => https://github.com/scipy/scipy/blob/v1.1.0/scipy/stats/_distn_infrastructure.py#L2478-L2519
#scipy entropy cannot be used since it's for discrete distributions
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import tensorflow as tf
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

def gaussian():
    ds = tf.contrib.distributions
    kl_divergence=tf.contrib.distributions.kl_divergence

    # Gaussian Mixure1
    mix = 0.3# weight
    bimix_gauss1 = ds.Mixture(
    cat=ds.Categorical(probs=[mix, 1.-mix]),#weight
    components=[
    ds.Normal(loc=-1., scale=0.1),
    ds.Normal(loc=+1., scale=0.5),
    ])

    # Gaussian Mixture2
    mix = 0.4# weight
    bimix_gauss2 = ds.Mixture(
        cat=ds.Categorical(probs=[mix, 1.-mix]),#weight
        components=[
            ds.Normal(loc=-0.4, scale=0.2),
            ds.Normal(loc=+1.2, scale=0.6),
    ])

    # KL between GM1 and GM2
    kl_value=kl_divergence(
        distribution_a=bimix_gauss1,
        distribution_b=bimix_gauss2,
        allow_nan_stats=True,
        name=None
    )
    sess = tf.Session() # 
    with sess.as_default():
        x = tf.linspace(-2., 3., int(1e4)).eval()
        plt.plot(x, bimix_gauss1.prob(x).eval(),'r-')
        plt.plot(x, bimix_gauss2.prob(x).eval(),'b-')
        plt.show()

        print('kl_value=',kl_value.eval())

def gmm_kl(gmm_p, gmm_q, n_samples=10**5):
    X = gmm_p.sample(n_samples)
    log_p_X, _ = gmm_p.score_samples(X)
    log_q_X, _ = gmm_q.score_samples(X)
    return log_p_X.mean() - log_q_X.mean()

#Approximating the Kullback Leibler divergence between Gaussian mixture models, 2007 IEEE
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
    