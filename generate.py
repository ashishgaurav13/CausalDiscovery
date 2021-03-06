import numpy as np
import torch
import pyro
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# DATA GENERATION

N = 5
P = 0.5
W = np.random.rand(N, N) < P # W ~ Bernoulli(P)
W = np.float32(W)
W = np.tril(W, -1)
OrigW = np.copy(W)
U = np.random.uniform(low=0.5, high=2., size=[N, N])
U = np.round(U, 1)
U[np.random.randn(N, N) < 0] *= -1
W = (W != 0).astype(float) * U
print("True Causal Graph")
print(OrigW)

M = 3
S = np.ones([N])
noise = np.random.randn(M, N) * S
X = np.zeros([M, N])
for m in range(M):
    for n in range(N):
        X[m, n] = X[m, :].dot(W[n, :])+noise[m, n]

# PYRO INFERENCE

def model(D):
    lowertriangular = torch.tril(torch.ones((N, N)), -1)
    W = pyro.sample("graph", pyro.distributions.Bernoulli(0.5 * lowertriangular))
    U = pyro.sample("weights", pyro.distributions.Uniform(-torch.ones((N, N)), torch.ones((N, N))))
    W *= U
    B = D.shape[0]
    X = torch.zeros([B, N])
    for n in range(N):
        X[:, n] = pyro.sample("X_%d" % (n), pyro.distributions.Normal(X.matmul(W[n, :]), 1), obs=D[:, n])

def guide(D):
    lowertriangular = torch.tril(torch.ones((N, N)), -1)
    Wp = pyro.param("graph_params", torch.rand(N, N))
    Up = pyro.param("weights_params", torch.rand(N, N)*2-1)
    W = pyro.sample("graph", pyro.distributions.Bernoulli(torch.clamp(Wp, min=0., max=1.) * lowertriangular))
    U = pyro.sample("weights", pyro.distributions.Normal(Up, 1.))
    W *= U
    B = D.shape[0]
    X = torch.zeros([B, N])
    for n in range(N):
        X[:, n] = pyro.sample("X_%d" % (n), pyro.distributions.Normal(X.matmul(W[n, :]), 1))


svi = pyro.infer.SVI(model, guide, pyro.optim.Adam({'lr': 0.01}), pyro.infer.Trace_ELBO())

BS = 50
for epoch in range(10):
    loss = 0.
    for _ in range(100):
        batch = torch.tensor(X[np.random.choice(X.shape[0], BS), :])
        loss += svi.step(batch)
    loss /= BS*100
    print("Epoch%d: Loss=%g" % (epoch, loss))
