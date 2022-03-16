import numpy as np
import functools
from scipy import ndimage
from bct.utils import binarize, get_rng
import multiprocessing
import itertools
from neuro_cluster.test_statistics import ttest, ttest_perm
from nibabel import Nifti1Image


def get_edge_components(A):
    """Short summary.

    Returns the components of an undirected graph specified by the binary and
    undirected matrix adj. Components and their constitutent nodes
    are assigned the same index and stored in the vector, comps. The vector,
    comp_sizes, contains the number of nodes beloning to each component.

    Parameters
    ----------
    A : NxM np.ndarray
        binary undirected matrix

    Returns
    -------
    comp_list : ([N], [M]) tuple of lists
        tuple of vector of component assignments for each node
    comp_sizes : Mx1 np.ndarray
        vector of component sizes
    """

    A = binarize(A, copy=True)
    n = A.shape[0]
    m = A.shape[1]

    edge_map = [(u, v) for u in range(n) for v in range(m) if A[u, v] == 1]
    comp_list = []

    # iterate through edge map and sort into components
    for item in edge_map:
        new_comp = [[item]]

        # find connected edges
        for s in comp_list:
            for edge in s:
                if item[0] == edge[0] or item[1] == edge[1]:
                    new_comp.append(s)
                    break

        # remove merged sets from component list
        for s in new_comp:
            if s in comp_list:
                comp_list.remove(s)

        # merge component and add new component to component list
        new_comp = list(itertools.chain.from_iterable(new_comp))
        comp_list.append(new_comp)

    # create list of component sizes
    edge_sizes = np.array([len(s) for s in comp_list])

    return comp_list, edge_sizes


def threshold_adj(tstat, thresh, num_edges):
    ind_t, = np.where(tstat > thresh)

    if len(ind_t) == 0:
        print("Unsuitable Threshold")

    # suprathreshold adjacency matrix
    adj = np.zeros([num_edges])
    adj[ind_t] = 1

    return adj


def find_clusters(adj, x_shape, masker):
    if len(x_shape) == 2:
        # will have to transform back into 3d space and then get clusters
        comps, sz_comps = get_edge_components3d(adj, masker)
    else:
        n = x_shape[0]
        m = x_shape[1]
        adj.resize([n, m])
        comps, sz_comps = get_edge_components(adj)

    return comps, sz_comps


def get_edge_components3d(adj, masker):
    img_3d = masker.inverse_transform(adj).get_fdata()

    binary_struct = ndimage.generate_binary_structure(3, 3)
    comps, num_features = ndimage.measurements.label(
        img_3d, binary_struct)
    comp_sizes = np.array([np.count_nonzero(comps == x)
                           for x in range(1, num_features + 1)])

    return comps, comp_sizes


def vectorize(x, y, num_sub_x, num_sub_y, num_edges):
    xmat, ymat = np.zeros((num_edges, num_sub_x)), np.zeros(
        (num_edges, num_sub_y))

    if len(x.shape) == 2:
        for i in range(num_sub_x):
            xmat[:, i] = x[:, i].flatten()
        for i in range(num_sub_y):
            ymat[:, i] = y[:, i].flatten()
    elif len(x.shape) == 3:
        for i in range(num_sub_x):
            xmat[:, i] = x[:, :, i].flatten()
        for i in range(num_sub_y):
            ymat[:, i] = y[:, :, i].flatten()
    else:
        raise Exception("X and Y must be 2D or 3D at this stage.")

    return xmat, ymat


def permute(xmat, ymat, x_shape, num_sub_x, num_sub_y, num_edges, paired, tail, thresh, masker, rng, null):
    """Generate null permutation and return the max component size.

    Parameters
    ----------
    n : int
        Description of parameter `n`.
    m : int
        Description of parameter `m`.
    num_sub_x : int
        number of subjects in x population
    num_sub_y : int
        number of subjects in y population
    xmat : type
        vectorized x population
    ymat : type
        vectorized y population
    tail : {'left', 'right', 'both'}
        enables specification of particular alternative hypothesis
        'left' : mean population of X < mean population of Y
        'right' : mean population of Y < mean population of X
        'both' : means are unequal (default)
    paired : bool
        use paired sample t-test instead of population t-test. requires both
        subject populations to have equal N. default value = False
    thresh : float
        minimum t-value used as threshold
    rng : type
        seed for randomizing permutation
    null : int
        will be returned as max component size

    Returns
    -------
    null : int
        max component size

    """

    # randomize
    if paired:
        indperm = np.sign(0.5 - rng.rand(1, num_sub_x))
        d = np.hstack((xmat, ymat)) * np.hstack((indperm, indperm))
    else:
        d = np.hstack((xmat, ymat))[:, rng.permutation(num_sub_x + num_sub_y)]

    # perform ttest
    tstat = ttest_perm(d, paired, num_edges, num_sub_x, num_sub_y, tail)

    # threshold tstat and transform into adjacency matrix
    adj = threshold_adj(tstat, thresh, num_edges)

    # get number of components and sizes
    comps, sz_comps = find_clusters(adj, x_shape, masker)

    # max size of null components
    if np.size(sz_comps):
        null = np.max(sz_comps)
    else:
        null = 0

    return null


def run(
    x,
    y,
    thresh,
    k=1000,
    tail="both",
    condition="within",
    paired=True,
    masker=None,
    verbose=False,
    seed=None,
    cores=4


):
    """Perform the NBS for populations X and Y for a t-statistic threshold of alpha.

    Parameters
    ----------
    x : NxMxP np.ndarray, NXP np.ndarray, or 4D Nifti1Image - VOXELSxP
        matrix representing the first population with P subjects. must include masker if data is NXP
    y : NxMxQ np.ndarray, NXQ np.ndarray, or 4D Nifti1Image - VOXELSxQ
        matrix representing the second population with Q subjects. Q need not
        equal P unless paired is set to true.
    thresh : float
        minimum t-value used as threshold
    k : int
        number of permutations used to estimate the empirical null
        distribution
    tail : {'left', 'right', 'both'}
        enables specification of particular alternative hypothesis
        'left' : mean population of X < mean population of Y
        'right' : mean population of Y < mean population of X
        'both' : means are unequal (default)
    paired : bool
        use paired sample t-test instead of population t-test. requires both
        subject populations to have equal N. default value = False
    verbose : bool
        print some extra information each iteration. defaults value = False
    seed : hashable, optional
        If None (default), use the np.random's global random state to generate random numbers.
        Otherwise, use a new np.random.RandomState instance seeded with the given value.
    Returns
    -------
    comps: np.ndarray
        A matrix of components
    pval : Cx1 np.ndarray
        A vector of corrected p-values for each component of the networks
        identified. If at least one p-value is less than alpha, the omnibus
        null hypothesis can be rejected at alpha significance. The null
        hypothesis is that the value of the connectivity from each edge has
        equal mean across the two populations.
    adj : IxIxC np.ndarray
        an adjacency matrix identifying the edges comprising each component.
        edges are assigned indexed values.
    null : Kx1 np.ndarray
        A vector of K sampled from the null distribution of maximal component
        size.
    sz_comps: Kx1 np.ndarray
    """
    rng = get_rng(seed)

    if tail not in ("both", "left", "right"):
        raise Exception("Tail must be both, left, right")

    if not x.shape[:-1] == y.shape[:-1]:
        raise Exception("Population matrices are of inconsistent size")

    if not masker and (len(x.shape) == 2 or len(x.shape) == 4):
        print("raising error")
        raise Exception(
            "Masker is required for transforming data back to 3D for clustering analysis.")

    # 4d niffti images need to be transformed to 2d (voxels by subjects)
    if len(x.shape) == 4:
        if type(x) != Nifti1Image:
            raise Exception("Must be Nifti1Image")
        x = masker.fit_transform(x)
        y = masker.fit_transform(y)

    num_sub_x = x.shape[-1]
    num_sub_y = y.shape[-1]
    if paired and num_sub_x != num_sub_y:
        raise Exception("Population matrices must be an equal size")

    num_edges = functools.reduce(lambda a, b: a * b, x.shape[:-1])

    # vectorize connectivity matrices for speed
    xmat, ymat = vectorize(x, y, num_sub_x, num_sub_y, num_edges)
    shape = x.shape
    del x, y

    # perform ttest
    tstat = ttest(xmat, ymat, paired, num_edges, tail)

    # threshold tstat and transform into adjacency matrix
    adj = threshold_adj(tstat, thresh, num_edges)

    # get number of components and sizes
    comps, sz_comps = find_clusters(adj, shape, masker)
    nr_components = np.size(sz_comps)

    # get max component size
    if np.size(sz_comps):
        max_sz = np.max(sz_comps)
    else:
        raise Exception('True matrix is degenerate')
    print('max component size is %i' % max_sz)

    print('estimating null distribution with %i permutations' % k)
    null_arr = np.zeros((k))

    # parallel processing of null distribution, drastically speeds up comp time
    pool = multiprocessing.Pool(cores)
    null_arr = pool.map(functools.partial(permute, xmat, ymat, shape,
                                          num_sub_x, num_sub_y, num_edges,
                                          paired, tail, thresh, masker, rng),
                        null_arr)

    # calculate p-vals
    pvals = np.zeros((nr_components))
    for i in range(nr_components):
        print(f'component index: {i}')
        print(f'comp size: {sz_comps[i]}')
        pvals[i] = np.size(np.where(null_arr >= sz_comps[i])) / k
        print(f'p-value: {pvals[i]:.4f}')

    return comps, pvals, adj, null_arr, sz_comps
