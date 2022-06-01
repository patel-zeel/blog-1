import numpy as np
import jax.numpy as jnp


def _sparse_cholesky_csc_jax_impl(L_indices, L_indptr, L_x):
    n = len(L_indptr) - 1
    descendant = [[] for j in range(0, n)]
    for j in range(0, n):
        tmp = L_x[L_indptr[j]:L_indptr[j + 1]]
        for bebe in descendant[j]:
            k = bebe[0]
            Ljk = L_x[bebe[1]]
            pad = jnp.nonzero(L_indices[L_indptr[k]:L_indptr[k + 1]] == L_indices[L_indptr[j]])[0][0]
            update_idx = np.nonzero(np.in1d(L_indices[L_indptr[j]:L_indptr[j + 1]],
                                            L_indices[(L_indptr[k] + pad):L_indptr[k + 1]]))[0]
            tmp[update_idx] = tmp[update_idx] - Ljk * L_x[(L_indptr[k] + pad):L_indptr[k + 1]]

        diag = jnp.sqrt(tmp[0])
        L_x[L_indptr[j]] = diag
        L_x[(L_indptr[j] + 1):L_indptr[j + 1]] = tmp[1:] / diag
        for idx in range(L_indptr[j] + 1, L_indptr[j + 1]):
            descendant[L_indices[idx]].append((j, idx))
    return L_x


def _symbolic_factor_csc2(A_indices, A_indptr):
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
    out = [[] for j in range(n)]
    for j in range(n):
        out[j] = jnp.array(L_sym[j])

    return out
