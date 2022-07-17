# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import jax.numpy as jnp
# import numpy
import numpy as np


# import scipy.sparse as sparse


def _symbolic_factor_csc(n, A_indices, A_indptr, A_x):
    r""" This produces the symbolic Cholesky factorisation of a sparse symmetric matrix in CSC format.
    It will output all of the structurally non-zero elements of the Cholesky factor.

    Parameters
    ----------

    n : int
        The number of rows/columns of the sparse matrix

    A_indices : static array[int]
        CSC format index array. The row indices for column i are stored in `indices[indptr[i]:indptr[i+1]]`.

    A_indptr : static array[int]
        CSC format index pointer array. The row indices for column i are stored in `indices[indptr[i]:indptr[i+1]]`.

    A_x : static array[double]-like

    """

    L_sym = [np.array([], dtype=int) for j in range(n)]
    children = [np.array([], dtype=int) for j in range(n)]  # this really should be a length n linked list...

    for j in range(n):
        L_sym[j] = A_indices[A_indptr[j]:A_indptr[j + 1]]
        for child in children[j]:
            tmp = L_sym[child][L_sym[child] > j]
            L_sym[j] = np.unique(np.append(L_sym[j], tmp))

        if len(L_sym[j]) > 1:
            p = L_sym[j][1]
            children[p] = np.append(children[p], j)

    L_indptr = np.zeros(n + 1, dtype=int)
    L_indptr[1:] = np.cumsum([len(x) for x in L_sym])
    L_indices = np.concatenate(L_sym)

    # Ok. Now let's set up our L_x array for the factorisation.
    # This is basically a deep copy of A to L that adds in zeros so we have
    # a copy of A with the same sparsity structure as L.
    L_x = np.zeros(len(L_indices))

    # for j in range(0, n):
    #  if (L_indptr[j + 1] - L_indptr[j]) != (A_indptr[j + 1] - A_indptr[j]):
    #   # There is fill in! Slot it in!
    #   n_fill = 0
    #   for i in range(0, A_indptr[j + 1] - A_indptr[j]):
    #    while L_indices[L_indptr[j] + i + n_fill] < A_indices[A_indptr[j] + i]:
    #     n_fill = n_fill + 1
    #    L_x[L_indptr[j] + i + n_fill] = A_x[A_indptr[j] + i]
    #  else:  # No fill
    #   L_x[L_indptr[j]:L_indptr[j + 1]] = A_x[A_indptr[j]:A_indptr[j + 1]]
    for j in range(0, n):
        copy_idx = np.nonzero(np.in1d(L_indices[L_indptr[j]:L_indptr[j + 1]],
                                      A_indices[A_indptr[j]:A_indptr[j + 1]]))[0]
        L_x[L_indptr[j] + copy_idx] = A_x[A_indptr[j]:A_indptr[j + 1]]
    return L_indices, L_indptr, L_x


def _sparse_cholesky_csc_impl_(n, L_indices, L_indptr, L_x):
    r"""
    Implements a sparse cholesky factorisation, assuming that a symbolic
    factorisation has been performed and the matrix to be factored is stored
    (with 0s for fill in) in the input data structure


    NB: Assumes indices are sorted!!!!

    :param n: (int) Number of columns
    :param L_indices: (static int array-like)
    :param L_indptr: (static int array-like)
    :param L_x: (double array-like)
    :return: L_x (double array-like)
    """

    # Allocate the list of descendants (these are the values of L[i,j] we need
    # to compute L[:,j]. To make this easy for ourselves, we are storing the
    # index into L_x that we need to retrieve these values
    descendant = [[] for j in range(0, n)]

    # Now we factor!
    for j in range(0, n):
        # Let's save time on accessing this damn thing
        tmp = L_x[L_indptr[j]:L_indptr[j + 1]]

        for bebe in descendant[j]:
            # tmp[j:]  -= L[j, k] * L[j:, k]
            k = bebe[0]
            Ljk = L_x[bebe[1]]

            # Where does the jth row of column k start? (non-empty otherwise wouldn't be a descendant)
            pad = np.nonzero(L_indices[L_indptr[k]:L_indptr[k + 1]] == L_indices[L_indptr[j]])[0][0]

            # Iterate through the non-zero elements of L[j:, k]
            # (NB: _Always_ a proper subset of the nonzero elements of L[:,j])
            # See George & Liu Lemma 5.5.1
            # Find values of tmp to update
            update_idx = np.nonzero(np.in1d(L_indices[L_indptr[j]:L_indptr[j + 1]],
                                            L_indices[(L_indptr[k] + pad):L_indptr[k + 1]]))[0]

            tmp[update_idx] = tmp[update_idx] - Ljk * L_x[(L_indptr[k] + pad):L_indptr[k + 1]]

        diag = np.sqrt(tmp[0])
        L_x[L_indptr[j]] = diag
        L_x[(L_indptr[j] + 1):L_indptr[j + 1]] = tmp[1:] / diag

        # I believe that children are the future (tell our future selves
        # that we are going to need to look at column j when we
        # factor it later. For those of us who can't be arsed playing with trees
        for idx in range(L_indptr[j] + 1, L_indptr[j + 1]):
            descendant[L_indices[idx]].append((j, idx))
    return L_x


def sparse_cholesky_csc(n, A_indices, A_indptr, A_x):
    r"""
    Returns the sparse cholesky factorisation of A in CSC format.
    Note: Only input the lower triangular part of A.
    Note: Assumes csc indices are sorted (easy to inforce. read the scipy docs)

    :param n: (int) number of columns
    :param A_indices: (static int array-like) row indices of tril(A) (csc storage)
    :param A_indptr: (static int array-like) column pointers of tril(A) (csc storage)
    :param A_x: (static double array-like) values of tril(A) (csc storage)
    :return:
    """
    L_indices, L_indptr, A_x_ = _symbolic_factor_csc(n, A_indices, A_indptr, A_x)
    L_x = _sparse_cholesky_csc_impl_(n, L_indices, L_indptr, A_x_)
    return L_indices, L_indptr, L_x
