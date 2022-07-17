import numpy as np
from jax import numpy as jnp
from scipy import sparse
from jax._src import abstract_arrays
from jax._src import ad_util
from jax import core
from jax.interpreters import ad
from jax import lax
from jax.experimental import sparse as jsparse
from functools import partial



## ==================================== sparse_triangular_solve ====================================

sparse_triangular_solve_p = core.Primitive("sparse_triangular_solve")

def sparse_triangular_solve(L_indices, L_indptr, L_x, b, *, transpose: bool = False):
  """A JAX traceable sparse  triangular solve"""
  return sparse_triangular_solve_p.bind(L_indices, L_indptr, L_x, b, transpose = transpose)

@sparse_triangular_solve_p.def_impl
def sparse_triangular_solve_impl(L_indices, L_indptr, L_x, b, *, transpose = False):
  """The implementation of the sparse triangular solve. This is not JAX traceable."""
  L = sparse.csc_array((L_x, L_indices, L_indptr)) 
  
  assert L.shape[0] == L.shape[1]
  assert L.shape[0] == b.shape[0]
  
  if transpose:
    return sparse.linalg.spsolve_triangular(L.T, b, lower = False)
  else:
    return sparse.linalg.spsolve_triangular(L.tocsr(), b, lower = True)

@sparse_triangular_solve_p.def_abstract_eval
def sparse_triangular_solve_abstract_eval(L_indices, L_indptr, L_x, b, *, transpose = False):
  assert L_indices.shape[0] == L_x.shape[0]
  assert b.shape[0] == L_indptr.shape[0] - 1
  return abstract_arrays.ShapedArray(b.shape, b.dtype)

def sparse_triangular_solve_value_and_jvp(arg_values, arg_tangent, *, transpose):
  """
  A jax-traceable jacobian-vector product. In order to make it traceable, 
  we use the experimental sparse CSC matrix in JAX.
  
  Input:
    arg_values:   A tuple of (L_indices, L_indptr, L_x, b) that describe
                  the triangular matrix L and the rhs vector b
    arg_tangent:  A tuple of tangent values (same lenght as arg_values).
                  The first two values are nonsense - we don't differentiate
                  wrt integers!
    transpose:    (boolean) If true, solve L^Tx = b. Otherwise solve Lx = b.
  Output:         A tuple containing the maybe_transpose(L)^{-1}b and the corresponding
                  Jacobian-vector product.
  """
  L_indices, L_indptr, L_x, b = arg_values
  _, _, L_xt, bt = arg_tangent
  value = sparse_triangular_solve(L_indices, L_indptr, L_x, b, transpose=transpose)
  if type(bt) is ad.Zero and type(L_xt) is ad.Zero:
    # I legit do not think this ever happens. But I'm honestly not sure.
    print("I have arrived!")
    return value, lax.zeros_like_array(value) 
  
  if type(L_xt) is not ad.Zero:
    # L is variable
    if transpose:
      Delta = jsparse.CSC((L_xt, L_indices, L_indptr), shape = (b.shape[0], b.shape[0])).transpose()
    else:
      Delta = jsparse.CSC((L_xt, L_indices, L_indptr), shape = (b.shape[0], b.shape[0]))

    jvp_Lx = sparse_triangular_solve(L_indices, L_indptr, L_x, Delta @ value, transpose = transpose) 
  else:
    jvp_Lx = lax.zeros_like_array(value) 

  if type(bt) is not ad.Zero:
    # b is variable
    jvp_b = sparse_triangular_solve(L_indices, L_indptr, L_x, bt, transpose = transpose)
  else:
    jvp_b = lax.zeros_like_array(value)

  return value, jvp_b - jvp_Lx

def sparse_triangular_solve_transpose_rule(cotangent, L_indices, L_indptr, L_x, b, *, transpose):
  """
  Transposition rule. Translated from here https://github.com/google/jax/blob/41417d70c03b6089c93a42325111a0d8348c2fa3/jax/_src/lax/linalg.py#L747
  """
  assert not ad.is_undefined_primal(L_x) and ad.is_undefined_primal(b)
  if type(cotangent) is ad_util.Zero:
    cot_b = ad_util.Zero(b.aval)
  else:
    cot_b = sparse_triangular_solve(L_indices, L_indptr, L_x, cotangent, transpose = not transpose)
  return None, None, None, cot_b

# Register jvp and transpose
ad.primitive_jvps[sparse_triangular_solve_p] = sparse_triangular_solve_value_and_jvp
ad.primitive_transposes[sparse_triangular_solve_p] = sparse_triangular_solve_transpose_rule

## ==================================== sparse_cholesky() ====================================
sparse_cholesky_p = core.Primitive("sparse_cholesky")

def sparse_cholesky(A_indices, A_indptr, A_x, *, L_nse: int = None):
  """A JAX traceable sparse cholesky decomposition"""
  if L_nse is None:
    err_string = "You need to pass a value to L_nse when doing fancy sparse_cholesky."
    ind = core.concrete_or_error(None, A_indices, err_string)
    ptr = core.concrete_or_error(None, A_indptr, err_string)
    L_ind, _ = _symbolic_factor(ind, ptr)
    L_nse = len(L_ind)
  
  return sparse_cholesky_p.bind(A_indices, A_indptr, A_x, L_nse = L_nse)

@sparse_cholesky_p.def_impl
def sparse_cholesky_impl(A_indices, A_indptr, A_x, *, L_nse):
  """The implementation of the sparse cholesky This is not JAX traceable."""
  
  L_indices, L_indptr= _symbolic_factor(A_indices, A_indptr)
  if L_nse is not None:
    assert len(L_indices) == L_nse
    
  L_x = _structured_copy(A_indices, A_indptr, A_x, L_indices, L_indptr)
  L_x = _sparse_cholesky_impl(L_indices, L_indptr, L_x)
  return L_indices, L_indptr, L_x

def _symbolic_factor(A_indices, A_indptr):
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
        
  L_indptr = np.zeros(n+1, dtype=int)
  L_indptr[1:] = np.cumsum([len(x) for x in L_sym])
  L_indices = np.concatenate(L_sym)
  
  return L_indices, L_indptr



def _structured_copy(A_indices, A_indptr, A_x, L_indices, L_indptr):
  n = len(A_indptr) - 1
  L_x = np.zeros(len(L_indices))
  
  for j in range(0, n):
    copy_idx = np.nonzero(np.in1d(L_indices[L_indptr[j]:L_indptr[j + 1]],
                                  A_indices[A_indptr[j]:A_indptr[j+1]]))[0]
    L_x[L_indptr[j] + copy_idx] = A_x[A_indptr[j]:A_indptr[j+1]]
  return L_x

def _sparse_cholesky_impl(L_indices, L_indptr, L_x):
  n = len(L_indptr) - 1
  descendant = [[] for j in range(0, n)]
  for j in range(0, n):
    tmp = L_x[L_indptr[j]:L_indptr[j + 1]]
    for bebe in descendant[j]:
      k = bebe[0]
      Ljk= L_x[bebe[1]]
      pad = np.nonzero(                                                       \
          L_indices[L_indptr[k]:L_indptr[k+1]] == L_indices[L_indptr[j]])[0][0]
      update_idx = np.nonzero(np.in1d(                                        \
                    L_indices[L_indptr[j]:L_indptr[j+1]],                     \
                    L_indices[(L_indptr[k] + pad):L_indptr[k+1]]))[0]
      tmp[update_idx] = tmp[update_idx] -                                     \
                        Ljk * L_x[(L_indptr[k] + pad):L_indptr[k + 1]]
            
    diag = np.sqrt(tmp[0])
    L_x[L_indptr[j]] = diag
    L_x[(L_indptr[j] + 1):L_indptr[j + 1]] = tmp[1:] / diag
    for idx in range(L_indptr[j] + 1, L_indptr[j + 1]):
      descendant[L_indices[idx]].append((j, idx))
  return L_x

@sparse_cholesky_p.def_abstract_eval
def sparse_cholesky_abstract_eval(A_indices, A_indptr, A_x, *, L_nse):
  return core.ShapedArray((L_nse,), A_indices.dtype),                   \
         core.ShapedArray(A_indptr.shape, A_indptr.dtype),             \
         core.ShapedArray((L_nse,), A_x.dtype)


def sparse_cholesky_value_and_jvp(arg_values, arg_tangent, *, L_nse):
  """
  A jax-traceable jacobian-vector product for the sparse cholesky factorisation.
  """

  A_indices, A_indptr, A_x = arg_values
  _, _, A_xt = arg_tangent

  L_indices, L_indptr, L_x = sparse_cholesky(A_indices, A_indptr, A_x, L_nse = L_nse)
  L_xt = _structured_copy(A_indices, A_indptr, A_xt, L_indices, L_indptr)

  n = len(L_indptr) - 1
  descendant = [[] for j in range(n)]
  for j in range(n):
    tmp_t = L_xt[L_indptr[j]: L_indptr[j+1]]
    for bebe in descendant[j]:
      k = bebe[0]
      Ljk = L_x[bebe[1]]
      Ljk_t = L_xt[bebe[1]]
      pad = 0 ## not done


## ==================================== sparse_solve() ====================================

def sparse_solve(A_indices, A_indptr, A_x, b, *, L_nse = None):
  """
  A JAX-traceable sparse solve. For this moment, only for vector b
  """
  assert b.shape[0] == A_indptr.shape[0] - 1
  assert b.ndim == 1
  
  L_indices, L_indptr, L_x = sparse_cholesky(
    lax.stop_gradient(A_indices), 
    lax.stop_gradient(A_indptr), 
    lax.stop_gradient(A_x), L_nse = L_nse)
  
  def chol_solve(L_indices, L_indptr, L_x, b):
    out = sparse_triangular_solve(L_indices, L_indptr, L_x, b, transpose = False)
    return sparse_triangular_solve(L_indices, L_indptr, L_x, out, transpose = True)
  
  def matmult(A_indices, A_indptr, A_x, b):
    # The things we have to do when addition isn't implemented!
    A_lower = jsparse.CSC((A_x, A_indices, A_indptr), shape = (b.shape[0], b.shape[0]))
    return A_lower @ b + A_lower.transpose() @ b - A_x[A_indptr[:-1]] * b

  solver = partial(
    lax.custom_linear_solve,
    lambda x: matmult(A_indices, A_indptr, A_x, x),
    solve = lambda _, x: chol_solve(L_indices, L_indptr, L_x, x),
    symmetric = True)

  return solver(b)



# sparse_solve_p = core.Primitive("sparse_solve")

# def sparse_solve(A_indices, A_indptr, A_x, b):
#   """A JAX traceable sparse solve"""
#   return sparse_solve_p.bind(A_indices, A_indptr, A_x, b)

# @sparse_solve_p.def_impl
# def sparse_solve_impl(A_indices, A_indptr, A_x, b):
#   """The implementation of the sparse solve. This is not JAX traceable."""
#   A_lower = sparse.csc_array((A_x, A_indices, A_indptr)) 
  
#   assert A_lower.shape[0] == A_lower.shape[1]
#   assert A_lower.shape[0] == b.shape[0]
  
#   A = A_lower + A_lower.T - sparse.diags(A_lower.diagonal())
#   return sparse.linalg.spsolve(A, b)

# @sparse_solve_p.def_abstract_eval
# def sparse_solve_abstract_eval(A_indices, A_indptr, A_x, b):
#   assert A_indices.shape[0] == A_x.shape[0]
#   assert b.shape[0] == A_indptr.shape[0] - 1
#   return abstract_arrays.ShapedArray(b.shape, b.dtype)

# def sparse_solve_value_and_jvp(arg_values, arg_tangents):
#   """ 
#   Jax-traceable jacobian-vector product implmentation for sparse_solve. 
#   Note that there's no transpose here, so there are no keyword arguments (yet)
#   """
  
#   A_indices, A_indptr, A_x, b = arg_values
#   _, _, A_xt, bt = arg_tangents

#   # Needed for shared computation
#   L_indices, L_indptr, L_x = sparse_cholesky(A_indices, A_indptr, A_x)

#   # Make the primal
#   primal_out = sparse_triangular_solve(L_indices, L_indptr, L_x, b, transpose = False)
#   primal_out = sparse_triangular_solve(L_indices, L_indptr, L_x, primal_out, transpose = True)

#   if type(A_xt) is not ad.Zero:
#     Delta_lower = jsparse.CSC((A_xt, A_indices, A_indptr), shape = (b.shape[0], b.shape[0]))
#     # We need to do Delta @ primal_out, but we only have the lower triangle
#     rhs = Delta_lower @ primal_out + Delta_lower.transpose() @ primal_out - A_xt[A_indptr[:-1]] * primal_out
#     jvp_Ax = sparse_triangular_solve(L_indices, L_indptr, L_x, rhs)
#     jvp_Ax = sparse_triangular_solve(L_indices, L_indptr, L_x, jvp_Ax, transpose = True)
#   else:
#     jvp_Ax = lax.zeros_like_array(primal_out)

#   if type(bt) is not ad.Zero:
#     jvp_b = sparse_triangular_solve(L_indices, L_indptr, L_x, bt)
#     jvp_b = sparse_triangular_solve(L_indices, L_indptr, L_x, jvp_b, transpose = True)
#   else:
#     jvp_b = lax.zeros_like_array(primal_out)

#   return primal_out, jvp_b - jvp_Ax

# def sparse_solve_transpose_rule(cotangent, A_indices, A_indptr, A_x, b):
#   assert not ad.is_undefined_primal(A_x) and ad.is_undefined_primal(b)
#   if type(cotangent) is ad.Zero:
#     cot_b = ad_util.Zero(b.aval)
#   else:

  
# Register jvp and transpose
#ad.primitive_jvps[sparse_solve_p] = sparse_solve_value_and_jvp

## ========================================= partial inverse =====================================

sparse_partial_inverse_p = core.Primitive("sparse_partial_inverse")

def sparse_partial_inverse(L_indices, L_indptr, L_x, out_indices, out_indptr):
  """
  Computes the elements (out_indices, out_indptr) of the inverse of a sparse matrix (A_indices, A_indptr, A_x)
   with Choleksy factor (L_indices, L_indptr, L_x). (out_indices, out_indptr) is assumed to be either
   the sparsity pattern of A or a subset of it in lower triangular form. 
  """
  return sparse_partial_inverse_p.bind(L_indices, L_indptr, L_x, out_indices, out_indptr)

@sparse_partial_inverse_p.def_abstract_eval
def sparse_partial_inverse_abstract_eval(L_indices, L_indptr, L_x, out_indices, out_indptr):
  return abstract_arrays.ShapedArray(out_indices.shape, L_x.dtype)

@sparse_partial_inverse_p.def_impl
def sparse_partial_inverse_impl(L_indices, L_indptr, L_x, out_indices, out_indptr):
  n = len(L_indptr) - 1
  Linv = sparse.dok_array((n,n), dtype = L_x.dtype)
  counter = len(L_x) - 1
  for col in range(n-1, -1, -1):
    for row in L_indices[L_indptr[col]:L_indptr[col+1]][::-1]:
      if row != col:
        Linv[row, col] = Linv[col, row] = 0.0
      else:
        Linv[row, col] = 1 / L_x[L_indptr[col]]**2

      L_col  = L_x[L_indptr[col]+1:L_indptr[col+1]] / L_x[L_indptr[col]]
      for k, L_kcol in zip(L_indices[L_indptr[col]+1:L_indptr[col+1]], L_col):
       Linv[col,row] = Linv[row,col] =  Linv[row, col] -  L_kcol * Linv[k, row]

  Linv_x = sparse.tril(Linv, format = "csc").data
  if len(out_indices) == len(L_indices):
    return Linv_x

  out_x = np.zeros(len(out_indices))
  for col in range(n):
    ind = np.nonzero(np.in1d(L_indices[L_indptr[col]:L_indptr[col+1]], out_indices[out_indptr[col]:out_indptr[col+1]]))[0]
    out_x[out_indptr[col]:out_indptr[col+1]] = Linv_x[L_indptr[col] + ind]
  return out_x



## ========================================= log-determinant =====================================

sparse_log_det_p = core.Primitive("sparse_log_det")

def sparse_log_det(A_indices, A_indptr, A_x):
  return sparse_log_det_p.bind(A_indices, A_indptr, A_x)

@sparse_log_det_p.def_impl
def sparse_log_det_impl(A_indices, A_indptr, A_x):
  L_indices, L_indptr, L_x = sparse_cholesky(A_indices, A_indptr, A_x)
  return 2.0 * jnp.sum(jnp.log(L_x[L_indptr[:-1]]))

@sparse_log_det_p.def_abstract_eval
def sparse_log_det_abstract_eval(A_indices, A_indptr, A_x):
  return abstract_arrays.ShapedArray((1,), A_x.dtype)

def sparse_log_det_value_and_jvp(arg_values, arg_tangent):
  A_indices, A_indptr, A_x = arg_values
  _, _, A_xt = arg_tangent
  L_indices, L_indptr, L_x = sparse_cholesky(A_indices, A_indptr, A_x)
  value = 2.0 * jnp.sum(jnp.log(L_x[L_indptr[:-1]]))
  Ainv_x = sparse_partial_inverse(L_indices, L_indptr, L_x, A_indices, A_indptr)
  jvp = 2.0 * sum(Ainv_x * A_xt) - sum(Ainv_x[A_indptr[:-1]] * A_xt[A_indptr[:-1]])
  return value, jvp

ad.primitive_jvps[sparse_log_det_p] = sparse_log_det_value_and_jvp
