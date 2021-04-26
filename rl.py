import math
import gym
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.optim
import pyro.infer
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
import random
import time
import tqdm
import matplotlib.pyplot as plt

SEED = 1
pyro.set_rng_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Structural causal model
N = 7
M = 500
x3 = np.random.uniform(size=M)
x0 = 3.0*x3 + np.random.uniform(size=M)
x2 = 6.0*x0 + np.random.uniform(size=M)
x5 = 4.0*x2 + np.random.uniform(size=M)
x1 = 3.0*x5 + 2.0*x0 + np.random.uniform(size=M)
x4 = 8.0*x5 + 1.0*x1 + np.random.uniform(size=M)
x6 = 2.0*x4 + np.random.uniform(size=M)
X = np.stack([x0, x1, x2, x3, x4, x5, x6]).T
print([3,0,2,5,1,4,6])


def _residual(xi, xj):
    """The residual when xi is regressed on xj."""
    return xi - (np.cov(xi, xj)[0, 1] / np.var(xj)) * xj

def _entropy(u):
    """Calculate entropy using the maximum entropy approximations."""
    k1 = 79.047
    k2 = 7.4129
    gamma = 0.37457
    return (1 + np.log(2 * np.pi)) / 2 - \
        k1 * (np.mean(np.log(np.cosh(u))) - gamma)**2 - \
        k2 * (np.mean(u * np.exp((-u**2) / 2)))**2

def _diff_mutual_info(xi_std, xj_std, ri_j, rj_i):
    """Calculate the difference of the mutual informations."""
    return (_entropy(xj_std) + _entropy(ri_j / np.std(ri_j))) - \
            (_entropy(xi_std) + _entropy(rj_i / np.std(rj_i)))

def check(X2, i, j):
    xi_std = (X2[:, i] - np.mean(X2[:, i])) / np.std(X2[:, i])
    xj_std = (X2[:, j] - np.mean(X2[:, j])) / np.std(X2[:, j])
    ri_j = _residual(xi_std, xj_std)
    rj_i = _residual(xj_std, xi_std)
    return min([0, _diff_mutual_info(xi_std, xj_std, ri_j, rj_i)])**2

def check2(perm):
    s = 0
    X2 = np.copy(X)
    for pi in range(len(perm)):
        for pj in range(len(perm)):
            if pj != pi:
                X2[:, perm[pj]] = _residual(X2[:, perm[pj]], X2[:, perm[pi]])
        for pj in range(pi+1, len(perm)):
            s += check(X2, perm[pi], perm[pj])
    return s

def c2ij(c):
    return [c//N, c%N]

def ij2c(ij):
    return ij[0]*N+ij[1]

import common
PERMUTATIONS = {}
t = common.TorchHelper()

class PermEnv(gym.Env):
    def reset(self):
        self.curr = np.random.permutation(N).tolist()
        if tuple(self.curr) not in PERMUTATIONS.keys():
            PERMUTATIONS[tuple(self.curr)] = check2(self.curr) # linear_regression(X[:, self.curr])
            if PERMUTATIONS[tuple(self.curr)] == min(PERMUTATIONS.values()):
                print(tuple(self.curr), PERMUTATIONS[tuple(self.curr)])
        self.count = 0
        return t.f(self.curr)
    def step(self, action):
        a1, a2 = c2ij(action)
        if tuple(self.curr) not in PERMUTATIONS.keys():
            PERMUTATIONS[tuple(self.curr)] = check2(self.curr) # linear_regression(X[:, self.curr])
            if PERMUTATIONS[tuple(self.curr)] == min(PERMUTATIONS.values()):
                print(tuple(self.curr), PERMUTATIONS[tuple(self.curr)])

        errold = PERMUTATIONS[tuple(self.curr)]
        temp = self.curr[a1]
        self.curr[a1] = self.curr[a2]
        self.curr[a2] = temp
        if tuple(self.curr) not in PERMUTATIONS.keys():
            PERMUTATIONS[tuple(self.curr)] = check2(self.curr) # linear_regression(X[:, self.curr])
            if PERMUTATIONS[tuple(self.curr)] == min(PERMUTATIONS.values()):
                print(tuple(self.curr), PERMUTATIONS[tuple(self.curr)])

        errnew = PERMUTATIONS[tuple(self.curr)]
        self.count += 1
        return t.f(self.curr), errold-errnew, self.count >= 50, {}






env = PermEnv()
env.seed(SEED)
episode = 0
alpha = 200
MAXTIME = 100
init_state = None
num_steps = 500
total_duration = 0
def reset_env():
    global env
    global init_state
    observation = env.reset()
    if (init_state is not None):
        observation = init_state
    return observation

def reset_init_state():
    global env
    init_state = env.reset()
    return init_state

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_net = nn.Sequential(
            nn.Linear(N, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, N*N), nn.Softmax(dim=-1))

    def forward(self, observation):
        return self.neural_net(observation)

final_epsilon = 0.05
initial_epsilon = 1
epsilon_decay = 5000
global steps_done
steps_done = 0
imme_time = 50

class AgentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = Policy()
        self.target_policy = Policy()
        self.imme_timestamp = 0
        self.initial_t = round(time.time())
        self.echo = False
        self.results = []
        self.timestamps = []
        self.avg_results = []
    
    def guide(self, max_time_step):
        pyro.module("agentmodel", self)

        observation = reset_env()
        for t in range(MAXTIME):
            state = observation
            action_logits = self.policy(observation)
            action = pyro.sample("action_{}".format(t), dist.Categorical(logits=action_logits))
            observation, reward, done, _ = env.step(action.item())
            
            if done and self.echo:
                return t
            
            if done:
                self.results.append(t)
                self.avg_results.append(np.mean(self.results[-10:]))
                self.timestamps.append(round(time.time()) - self.initial_t)
                return t
        
        # solved
        self.results.append(max_time_step)
        self.avg_results.append(np.mean(self.results[-10:]))
        self.timestamps.append(round(time.time()) - self.initial_t)
        
        if self.echo:
            print("guide solve the problem at t:", max_time_step)
            return max_time_step
    
    def model(self, max_time_step):
        pyro.module("agentmodel", self)

        observation = reset_env()
        add = True
        total_reward = torch.tensor(0.)
        for t in range(MAXTIME):
            action_logits = torch.nn.Softmax(dim=-1)(torch.ones([N*N]))
            action = pyro.sample("action_{}".format(t), dist.Categorical(logits=action_logits))
            observation, reward, done, _ = env.step(action.item())

            if done and add:
                add = False
            
            if add:
                total_reward += reward * 10
                
            if done:
                break

        global episode
        episode += 1
        pyro.factor("Episode_{}".format(episode), total_reward * alpha)

    def run_guide(self):
        global imme_time
        self.echo = True
        results = []
        for _ in range(20):
            global init_state
            init_state = reset_init_state()
            survive = guide(MAXTIME)
            results.append(survive)
        self.echo = False

agent = AgentModel()
guide = agent.guide
model = agent.model
learning_rate = 8e-4 #1e-5
optimizer = optim.Adam({"lr":learning_rate})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

def optimize():
    global imme_time
    loss = 0
    print("Optimizing...")
    for t in range(num_steps):
        global init_state
        init_state = reset_init_state()
        loss += svi.step(imme_time)
        if (t % 100 == 0) and (t > 0):
            print("at {} step loss is {}".format(t, loss / t))

def train(epoch=2, batch_size=10):
    # memory = []
    global start_time
    global total_duration
    for epoc in range(epoch):
        pyro.get_param_store().clear()
        optimize()

def test_loop(n=10):
    results = []
    for _ in range(n):
        results.append(test())

def test(max_timestamp=MAXTIME):
    global init_state
    init_state = reset_init_state()
    observation = reset_env()
    initial_obs = torch.clone(observation)
    for t in range(max_timestamp):
        action_logits = agent.policy(observation)
        action = dist.Categorical(logits=action_logits).sample()
        observation, reward, done, _ = env.step(action.item())
        if done:
            print(initial_obs, '->', observation)
            return t
    print(initial_obs, '->', observation)
    return max_timestamp

def save():
    print("save to cartpole_model.pt and cartpole_model_params.pt")
    optimizer.save("cartpole_optimzer.pt")
    torch.save({"model" : None, "policy" : agent.policy, "steps_done" : steps_done}, "cartpole_model.pt")
    pyro.get_param_store().save("cartpole_model_params.pt")

def load():
    print("load from cartpole_model.pt and cartpole_model_params.pt")
    saved_model_dict = torch.load("cartpole_model.pt")
    agent.policy.load_state_dict(saved_model_dict['policy'].state_dict())
    pyro.get_param_store().load("cartpole_model_params.pt")

start_time = time.time()
print("time start", start_time)
train(epoch=1)
test(2000)

topological_order = sorted(PERMUTATIONS.items(), key=lambda x:x[1])[0][0]