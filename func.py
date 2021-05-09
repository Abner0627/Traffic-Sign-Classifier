import numpy as np


def _norm(x, Z=True):
    x_n = np.zeros_like(x)
    for n in range(x.shape[0]):
        xs = x[n,...]
        for i in range(xs.shape[0]):
            xss = xs[i,:,:]
            x_mu = np.mean(xss)
            if Z:
                x_std = np.std(xss)
                x_n[n,i,:,:] = (xss-x_mu)/x_std
            else:
                x_min = np.min(xss)
                x_max = np.max(xss)
                x_n[n,i,:,:] = (xss-x_mu)/(x_max-x_min)
    return x_n

def _gray(x):
    # 3, 32, 32
    gray = np.zeros((x.shape[0], 1, x.shape[-1], x.shape[-1]))
    rgb_weights = [0.2989, 0.5870, 0.1140]
    for i in range(len(x)):
        img = x[i,:,:,:].transpose(1,2,0)
        gray[i,:,:,:] = np.dot(img[...,:3], rgb_weights)
    return gray


#%% Test
if __name__ == "__main__":
    x = np.random.rand(2,3,32,32)
    y = _norm(x)
    print(y.shape)

