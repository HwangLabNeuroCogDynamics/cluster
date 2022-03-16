import numpy as np
from cluster import run
from nbs import nbs_bct
import seaborn as sns
import matplotlib.pyplot as plt


# # load data
# x = np.load('f2_thalamus_network_group_matrix.npy')
# y = np.load('fh_thalamus_network_group_matrix.npy')
#
# # set paired t-test, threshold of t=1.8 (arbitrary)
# # return pvalues, adj, and null
# pvals, adj, null = nbs_bct(x, y, thresh=2.583, k=1000,
#                            tail='both', paired=True, verbose=False, seed=None)
# # 2.583 corresponds roughly to p <.01 for dof=16
#
# # print the list of pvalues of the significant lesymap_clusters
# print(pvals)
# # if there are multiple "clusters", the p values will be entered into pvals in order
#
# # plot the cluster it finds
# # the edges will be labeled in integer, corresponding to the order in pvals
# sns.heatmap(adj)
# plt.show()
#
#
# # you can also mask the contrast, and plot it.
# plot = sns.heatmap((adj > 0)*np.mean(x-y, axis=2))
# plt.show()
#
# # to find out the indices of significant edges
# # xi are indces with significant edges along the x axis, same for yi.
# xi, yi = np.where(adj)


# Simulation to test if the "zeros" matter for nbs
# generate a random 10 by 10 by 17 matrix, make the scale 0 to 0.1, similar to FC.
x = np.random.rand(10, 10, 17) * 0.05
y = np.random.rand(10, 10, 17) * 0.05

# make parts of this matrix super significant
x[2:5, 2:5, :] = x[2:5, 2:5, :]+0.7

for z in np.arange(0, np.shape(x)[2]):
    np.fill_diagonal(x[:, :, z], 0)  # fill diagnal with zero
    np.fill_diagonal(y[:, :, z], 0)

sns.heatmap(np.mean(x, axis=2))
plt.show()

# nbs
pvals, adj_orig, null = nbs_bct(
    x, y, thresh=2.583, k=100, tail='both', paired=True, verbose=False, seed=None)
print(pvals)
print(np.where(adj_orig))
sns.heatmap(adj_orig)
plt.show()

# Simulation to test if the "zeros" matter for nbs
# generate a random 10 by 10 by 17 matrix, make the scale 0 to 0.1, similar to FC.
x = np.random.rand(20, 30, 17, 1) * 0.05
y = np.random.rand(20, 30, 17, 1) * 0.05

# make parts of this matrix super significant
x[2:5, 2:5, :] = x[2:5, 2:5, :]+0.7

# sns.heatmap(np.mean(x_zeros, axis=2))
# plt.show()
# sns.heatmap(np.mean(y_zeros, axis=2))
# plt.show()

# will nbs find similar clusters?
comps, pvals, adj, null, sz_comps = run(x, y, thresh=2.583,
                                        k=100, tail='both', paired=True, verbose=False, seed=None)
print(pvals)
print(np.where(adj))
sns.heatmap(adj)
plt.show()

# compare
print(np.where(adj_orig))
print(np.where(adj))
# looks like it doesnt! still need to hack
