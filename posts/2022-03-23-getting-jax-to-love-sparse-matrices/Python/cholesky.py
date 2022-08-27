import numpy as np
def _symbolic_factor_csc(A_indices, A_indptr):
    # Assumes A_indices and A_indptr index the lower triangle of $A$ ONLY.
    n = len(A_indptr) - 1
    L_sym = [np.array([], dtype=int) for j in range(n)]
    children = [np.array([], dtype=int) for j in range(n)]

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

    return L_indices, L_indptr


def _deep_copy_csc(A_indices, A_indptr, A_x, L_indices, L_indptr):
    n = len(A_indptr) - 1
    L_x = np.zeros(len(L_indices))

    for j in range(0, n):
        copy_idx = np.nonzero(np.in1d(L_indices[L_indptr[j]:L_indptr[j + 1]],
                                      A_indices[A_indptr[j]:A_indptr[j + 1]]))[0]
        L_x[L_indptr[j] + copy_idx] = A_x[A_indptr[j]:A_indptr[j + 1]]
    return L_x


def _sparse_cholesky_csc_impl(L_indices, L_indptr, L_x):
    n = len(L_indptr) - 1
    descendant = [[] for j in range(0, n)]
    for j in range(0, n):
        tmp = L_x[L_indptr[j]:L_indptr[j + 1]]
        for bebe in descendant[j]:
            k = bebe[0]
            Ljk = L_x[bebe[1]]
            pad = np.nonzero(L_indices[L_indptr[k]:L_indptr[k + 1]] == L_indices[L_indptr[j]])[0][0]
            update_idx = np.nonzero(np.in1d(L_indices[L_indptr[j]:L_indptr[j + 1]],
                                            L_indices[(L_indptr[k] + pad):L_indptr[k + 1]]))[0]
            tmp[update_idx] = tmp[update_idx] - Ljk * L_x[(L_indptr[k] + pad):L_indptr[k + 1]]

        diag = np.sqrt(tmp[0])
        L_x[L_indptr[j]] = diag
        L_x[(L_indptr[j] + 1):L_indptr[j + 1]] = tmp[1:] / diag
        for idx in range(L_indptr[j] + 1, L_indptr[j + 1]):
            descendant[L_indices[idx]].append((j, idx))
    return L_x


def sparse_cholesky_csc(A_indices, A_indptr, A_x):
    L_indices, L_indptr = _symbolic_factor_csc(A_indices, A_indptr)
    L_x = _deep_copy_csc(A_indices, A_indptr, A_x, L_indices, L_indptr)
    L_x = _sparse_cholesky_csc_impl(L_indices, L_indptr, L_x)
    return L_indices, L_indptr, L_x
