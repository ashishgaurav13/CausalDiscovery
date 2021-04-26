import numpy as np
import torch
import pyro
import matplotlib.pyplot as plt
import warnings
import tqdm
warnings.filterwarnings("ignore")


SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
pyro.set_rng_seed(SEED)

# X -> M samples x N
def linear_regression(orig):

    def model(X):
        N = X.shape[1]
        B = X.shape[0]
        lowertriangular = torch.tril(torch.ones((N, N)), -1)
        W = pyro.sample("weights", pyro.distributions.Normal(torch.zeros(N, N), torch.ones(N, N)).independent(2))
        Y = pyro.sample("pred", pyro.distributions.Normal((W*lowertriangular).matmul(X.T), torch.ones(N, B)).independent(2), obs=X.T)
    
    def guide(X):
        N = X.shape[1]
        B = X.shape[0]
        # Gp = pyro.param("graph_params", torch.rand(N))
        lowertriangular = torch.tril(torch.ones((N, N)), -1)
        Wp1 = pyro.param("weights_1", torch.rand(N, N))
        Wp2 = pyro.param("weights_2", torch.rand(N, N))
        W = pyro.sample("weights", pyro.distributions.Normal(Wp1, torch.abs(Wp2)).independent(2))
        Y = pyro.sample("pred", pyro.distributions.Normal((W*lowertriangular).matmul(X.T), torch.ones(N, B)).independent(2))

    pyro.clear_param_store()
    svi = pyro.infer.SVI(model, guide, pyro.optim.Adam({'lr': 0.01}), pyro.infer.Trace_ELBO())
    BS = 100
    pbar = tqdm.trange(50)
    for epoch in pbar:
        loss = 0.
        for _ in range(100):
            idx = np.random.choice(orig.shape[0], BS)
            batch = torch.tensor(orig[idx, :]).float()
            loss += svi.step(batch)
        loss /= BS*100
        pbar.set_description("Loss=%g" % (loss))
    
    lowertriangular = torch.tril(torch.ones((N, N)), -1)
    print(lowertriangular*pyro.param("weights_1"))
    print(lowertriangular*pyro.param("weights_2"))
    return lowertriangular*pyro.param("weights_1")



N = 7
P = 0.5
W = np.random.rand(N, N) < P # W ~ Bernoulli(P)
W = np.float32(W)
W = np.tril(W, -1)
U = np.random.rand(N, N) # uniform(low=0.5, high=2., size=[N, N])
U = np.round(U, 1)
U[np.random.randn(N, N) < 0] *= -1
W = (W != 0).astype(float) * U
print("True Causal Graph")
print(W)

M = 5000
S = np.ones([N])
noise = np.random.laplace(size=[M, N]) # * S
X = np.zeros([M, N])
for m in range(M):
    for n in range(N):
        X[m, n] = X[m, :].dot(W[n, :])+noise[m, n]

torch.set_printoptions(sci_mode=False, precision=2)
W2 = linear_regression(X)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(W, vmin=-1., vmax=1.)
plt.subplot(1, 2, 2)
plt.imshow(W2.detach().numpy(), vmin=-1., vmax=1.)
plt.show()

