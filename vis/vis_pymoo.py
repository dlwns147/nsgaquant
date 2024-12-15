import matplotlib.pyplot as plt
import numpy as np

from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.binx import BinomialCrossover

import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm


# fig_save = 'fig/pymoo.png'
fig_save = 'fig/mut.png'

# n_var, n_matings = 50, 30

# problem = Problem(n_var=n_var, xl=0.0, xu=1.0, var_type=int)

# a, b = Individual(X=np.arange(1, n_var + 1)), Individual(X=-np.arange(1, n_var + 1))

# parents = [[a, b] for _ in range(n_matings)]

# off = BinomialCrossover(prob=1.0).do(problem, parents)
# Xp = off.get("X")

# plt.figure(figsize=(4, 6))
# plt.imshow(Xp[:n_matings] != a.X, cmap='Greys', interpolation='nearest')
# plt.xlabel("Variables")
# plt.ylabel("Individuals")
# plt.show()
# plt.savefig(fig_save)
np.random.seed(0)

problem = Problem(n_var=1, xl=0.0, xu=1.0)

# X = np.full((5000, 1), 0.5)
# n_var = 50
# n_var = 224
# n_var = 280
n_var = 560
X = np.full((n_var, 1), 0.5)
# X = np.random.randint(0, 2, size=(n_var, 1))
# pop = Population.new(X=X)

# mutation = PolynomialMutation(prob=1.0, eta=1)
# mutation = PolynomialMutation(prob=(1/n_var), eta=1)
mutation = PolynomialMutation(prob=0.02, eta=1)
# mutation = PolynomialMutation(prob=0.05, eta=1)
# mutation = PolynomialMutation(prob=0.1, eta=1)
# mutation = PolynomialMutation(prob=0.2, eta=1)

diff_list = []
n_off = 100
n_gen = 20
total_var = n_var * n_off * n_gen
for _ in tqdm(range(n_off * n_gen)):
    pop = Population.new(X=X)
    off = mutation(problem, pop)
    Xp = off.get("X").round().astype(int)

    diff = np.where(X == Xp, 0, Xp)
    diff = diff[np.flatnonzero(diff)].flatten()
    # print(diff.shape)
    diff_list.append(diff)

diff_list = np.concatenate(diff_list, axis=0)
# print(diff_list)
mut_prob = len(diff_list) / (total_var)
print(f'mut_prob : {mut_prob}')
plt.hist(diff_list, range=(0.0, 1.0), bins=200, color='b')
plt.title(f'total_var : {total_var}, n_diff : {len(diff_list)}')
plt.xlabel('value')
plt.ylabel('n_value')
plt.show()
plt.savefig(fig_save, dpi=300)
# off = mutation(problem, pop)
# Xp = off.get("X")

# print(f'X : {X.shape}')
# print(f'Xp : {Xp.shape}')
# print(np.where(X == Xp, 0, 1).sum())
# exit()

# Xp_round = np.round(Xp)

# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# fig.subplots_adjust(hspace=0.5, wspace=0.3)
# axes[0].hist(X, range=(0.0, 1.0), bins=200, density=True, color="red")
# axes[0].set_xlabel("value")
# axes[0].set_ylabel("# pop")
# axes[0].set_title('Parents')
# axes[1].hist(Xp, range=(0.0, 1.0), bins=200, density=True, color="red")
# axes[1].set_xlabel("value")
# axes[1].set_ylabel("# pop")
# axes[1].set_title('Polynomial Mutation')
# # axes[2].hist(Xp_round, range=(0.0, 1.0), bins=200, density=True, color="red")
# # axes[2].set_xlabel("value")
# # axes[2].set_ylabel("# pop")
# # axes[2].set_title('IntPolynomial mutation')
# plt.show()
# plt.savefig(fig_save, dpi=300)
