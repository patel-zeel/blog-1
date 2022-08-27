import cholesky
import jax_cholesky
import numpy as np
import jax.numpy as jnp
from jax import jit, grad, tree_map
from scipy import sparse
import timeit
from functools import partial

def _structured_copy_csc(A_index, A_x, L_index):
    def body_fun(A_rows, A_vals, L_rows):
      out = jnp.zeros(len(L_rows))
      copy_idx =  jnp.nonzero(jnp.in1d(L_rows, A_rows), size = len(A_rows))[0]
      out = out.at[copy_idx].set(A_vals)
      return out
    L_x = tree_map(body_fun, A_index, A_x, L_index)
    return L_x


def test_func(A_index, A_x, params):
  I_index = [jnp.array([j]) for j in range(len(A_index))]
  I_x = [jnp.array([params[0]]) for j in range(len(A_index))]
  I_x2 = _structured_copy_csc(I_index, I_x, A_index)
  def inner_func(i, a):
      return sum((i + params[1] * a)**2)

  return jnp.sum((jnp.concatenate(I_x2) + params[1] * jnp.concatenate(A_x))**2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    A_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    A_indptr = np.array([0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    A_x = np.array([0.6, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    L_indices_true = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 4, 5, 6,
         7,
         8, 9, 5, 6, 7, 8, 9, 6, 7, 8, 9, 7, 8, 9, 8, 9, 9])
    L_indptr_true = np.array([0, 10, 19, 27, 34, 40, 45, 49, 52, 54, 55])
    L_x_true = np.array(
        [0.774596669241483, -0.258198889747161, -0.258198889747161, -0.258198889747161, -0.258198889747161,
         -0.258198889747161, -0.258198889747161, -0.258198889747161, -0.258198889747161, -0.258198889747161,
         0.966091783079296, -0.0690065559342354, -0.0690065559342354, -0.0690065559342354, -0.0690065559342354,
         -0.0690065559342354, -0.0690065559342354, -0.0690065559342354, -0.0690065559342354, 0.963624111659432,
         -0.0741249316661101, -0.0741249316661101, -0.0741249316661101, -0.0741249316661101, -0.0741249316661101,
         -0.0741249316661101, -0.0741249316661101, 0.960768922830523, -0.0800640769025436, -0.0800640769025436,
         -0.0800640769025436, -0.0800640769025436, -0.0800640769025436, -0.0800640769025436, 0.957427107756338,
         -0.0870388279778489, -0.0870388279778489, -0.0870388279778489, -0.0870388279778489, -0.0870388279778489,
         0.953462589245592, -0.0953462589245593, -0.0953462589245593, -0.0953462589245593, -0.0953462589245593,
         0.948683298050514, -0.105409255338946, -0.105409255338946, -0.105409255338946, 0.942809041582063,
         -0.117851130197758, -0.117851130197758, 0.935414346693485, -0.133630620956212, 0.925820099772551])

    #L_indices2, L_indptr2, L_x2 = cholesky.sparse_cholesky_csc(A_indices2, A_indptr2, A_x2)

    L_indices, L_indptr = cholesky._symbolic_factor_csc(A_indices, A_indptr)
    L_x = cholesky._deep_copy_csc(A_indices, A_indptr, A_x, L_indices, L_indptr)

    L_indices_jax = jnp.array(L_indices)
    L_indptr_jax = jnp.array(L_indptr)
    A_x_jax = jnp.array(L_x)

    #L_x_jax2 = jax_cholesky._sparse_cholesky_csc_jax_impl(L_indices_jax, L_indptr_jax, A_x_jax)

    # print("========================\n Big boy indices\n========================")
    # print(sum(L_indices - L_indices_true))
    # print("========================\n Sexy pointers \n========================")
    # print(sum(L_indptr - L_indptr_true))
    # print("========================\n Cholesky time! \n========================")
    # print(np.sum(np.abs(L_x - L_x_true)))
    #
    # print("==================\n Will this work? \n===================")

    A_sym = jnp.split(A_indices, A_indptr[1:-1])
    L_sym = jax_cholesky._symbolic_factor_csc2(A_indices, A_indptr)
    A_x_sym = jnp.split(jnp.array(A_x), A_indptr[1:-1])

    L_x = _structured_copy_csc(A_sym, A_x_sym, L_sym)
    print(L_x)
    print(type(L_x))

    L_x2 = jit(_structured_copy_csc)(A_sym, A_x_sym, L_sym)
    print(L_x2)
    print(type(L_x2))

    print(test_func(A_sym, A_x_sym, (2.0, 3.0)))
    print(jit(test_func)(A_sym, A_x_sym, (2.0, 3.0)))
    print(grad(test_func, argnums=2)(A_sym, A_x_sym, (2.0, 3.0)))



    n = 50
    one_d = sparse.diags([[-1.] * (n - 1), [2.] * n, [-1.] * (n - 1)], [-1, 0, 1])
    A = sparse.kronsum(one_d, one_d) + sparse.eye(n * n)
    A_lower = sparse.tril(A, format="csc")
    A_indices = A_lower.indices
    A_indptr = A_lower.indptr
    A_x = A_lower.data

    A_index2 = jnp.split(jnp.array(A_indices), A_indptr[1:-1])
    A_x2 = jnp.split(jnp.array(A_x), A_indptr[1:-1])

    print("here\n")

    grad_func = grad(test_func, argnums=2)
    print("here\n")
    jit_func = jit(test_func)
    print("here\n")

 #   print(f"The value at (2.0, 2.0) is {jit_func(A_index2, A_x2, (2.0, 2.0))}")#. The gradient is {grad_func(A_index2, A_x2, (2.0, 2.0))}")
  #  print(f"The value at (2.0, 2.0) is {jit_func(A_index2, A_x2, (3.0, -1.0))}")

   # _ = jit_func(A_index2, A_x2, (2.0, 3.0))

    print(timeit.repeat(partial(jit_func, A_index2, A_x2, (2.0, 2.0)), number =1))

    # sparse_cholesky(10, 3, A_indices, A_indptr)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
