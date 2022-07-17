from linalg import *
from jax import numpy as jnp
from scipy import sparse
import numpy as np
from jax import jvp, grad
from jax import scipy as jsp
from jax.interpreters import ad
from jax.config import config

    

def make_matrix(n):
  one_d = sparse.diags([[-1.]*(n-1), [2.]*n, [-1.]*(n-1)], [-1,0,1])
  A = (sparse.kronsum(one_d, one_d) + sparse.eye(n*n)).tocsc()
  A_lower = sparse.tril(A, format = "csc")
  A_index = A_lower.indices
  A_indptr = A_lower.indptr
  A_x = A_lower.data
  return (A_index, A_indptr, A_x, A)

def f(theta):
  Ax_theta = jnp.array(A_x)
  Ax_theta = Ax_theta.at[A_indptr[20]].add(theta[0])
  Ax_theta = Ax_theta.at[A_indptr[50]].add(theta[1])
  b = jnp.ones(100)
  return sparse_triangular_solve(A_indices, A_indptr, Ax_theta, b, transpose = True)

def f_jax(theta):
  Ax_theta = jnp.array(sparse.tril(A).todense())
  Ax_theta = Ax_theta.at[20,20].add(theta[0])
  Ax_theta = Ax_theta.at[50,50].add(theta[1])
  b = jnp.ones(100)
  return jsp.linalg.solve_triangular(Ax_theta, b, lower = True, trans = "T")

def g(theta):
  Ax_theta = jnp.array(A_x)
  b = jnp.ones(100)
  b = b.at[0].set(theta[0])
  b = b.at[51].set(theta[1])
  return sparse_triangular_solve(A_indices, A_indptr, Ax_theta, b, transpose = True)

def g_jax(theta):
  Ax_theta = jnp.array(sparse.tril(A).todense())
  b = jnp.ones(100)
  b = b.at[0].set(theta[0])
  b = b.at[51].set(theta[1])
  return jsp.linalg.solve_triangular(Ax_theta, b, lower = True, trans = "T")

def h(theta):
  Ax_theta = jnp.array(A_x)
  Ax_theta = Ax_theta.at[A_indptr[20]].add(theta[0]) 
  b = jnp.ones(100)
  b = b.at[51].set(theta[1])
  return sparse_triangular_solve(A_indices, A_indptr, Ax_theta, b, transpose = False)

def h_jax(theta):
  Ax_theta = jnp.array(sparse.tril(A).todense())
  Ax_theta = Ax_theta.at[20,20].add(theta[0])
  b = jnp.ones(100)
  b = b.at[51].set(theta[1])
  return jsp.linalg.solve_triangular(Ax_theta, b, lower = True, trans = "N")

def no_diff(theta):
  return sparse_triangular_solve(A_indices, A_indptr, A_x, jnp.ones(100), transpose = False)

def no_diff_jax(theta):
  return jsp.linalg.solve_triangular(jnp.array(sparse.tril(A).todense()), jnp.ones(100), lower = True, trans = "N")

if __name__ == "__main__":
  config.update("jax_enable_x64", True)

  A_indices, A_indptr, A_x, A = make_matrix(10)
  primal1, jvp1 = jvp(f, (jnp.array([-142., 342.]),), (jnp.array([1., 2.]),))
  primal2, jvp2 = jvp(f_jax, (jnp.array([-142., 342.]),), (jnp.array([1., 2.]),))
  grad1 = grad(lambda x: jnp.mean(f(x)))(jnp.array([-142., 342.]))
  grad2 = grad(lambda x: jnp.mean(f_jax(x)))(jnp.array([-142., 342.]))

  primal3, jvp3 = jvp(g, (jnp.array([-142., 342.]),), (jnp.array([1., 2.]),))
  primal4, jvp4 = jvp(g_jax, (jnp.array([-142., 342.]),), (jnp.array([1., 2.]),))
  grad3 = grad(lambda x: jnp.mean(g(x)))(jnp.array([-142., 342.]))
  grad4 = grad(lambda x: jnp.mean(g_jax(x)))(jnp.array([-142., 342.]))  

  primal5, jvp5 = jvp(h, (jnp.array([-142., 342.]),), (jnp.array([1., 2.]),))
  primal6, jvp6 = jvp(h_jax, (jnp.array([-142., 342.]),), (jnp.array([1., 2.]),))
  grad5 = grad(lambda x: jnp.mean(h(x)))(jnp.array([-142., 342.]))
  grad6 = grad(lambda x: jnp.mean(h_jax(x)))(jnp.array([-142., 342.]))

  primal7, jvp7 = jvp(no_diff, (jnp.array([-142., 342.]),), (jnp.array([1., 2.]),))
  primal8, jvp8 = jvp(no_diff_jax, (jnp.array([-142., 342.]),), (jnp.array([1., 2.]),))
  grad7 = grad(lambda x: jnp.mean(no_diff(x)))(jnp.array([-142., 342.]))
  grad8 = grad(lambda x: jnp.mean(no_diff_jax(x)))(jnp.array([-142., 342.]))

  print(f"""
  Variable L:
    Primal difference: {np.linalg.norm(primal1 - primal2)}
    JVP difference: {np.linalg.norm(jvp1 - jvp2)}
    Gradient difference: {np.linalg.norm(grad1 - grad2)}

  Variable b:
    Primal difference: {np.linalg.norm(primal3 - primal4)}
    JVP difference: {np.linalg.norm(jvp3 - jvp4)}
    Gradient difference: {np.linalg.norm(grad3 - grad4)} 

  Variable L and b:
    Primal difference: {np.linalg.norm(primal5 - primal6)}
    JVP difference: {np.linalg.norm(jvp5 - jvp6)}
    Gradient difference: {np.linalg.norm(grad5 - grad6)}

  No diff:
    Primal difference: {np.linalg.norm(primal7 - primal8)}
    JVP difference: {np.linalg.norm(jvp7 - jvp8)}
    Gradient difference: {np.linalg.norm(grad7 - grad8)}
  """)


  print("""
  ========== Time for some solve action ========
  """)
  # b = np.random.standard_normal(100)

  # bt = np.random.standard_normal(100)
  # bt /= np.linalg.norm(bt)

  # A_xt = np.random.standard_normal(len(A_x))
  # A_xt /= np.linalg.norm(A_xt)

  # arg_values = (A_indices, A_indptr, A_x, b )

  # arg_tangent_A = (None, None, A_xt, ad.Zero(type(b)))
  # arg_tangent_b = (None, None, ad.Zero(type(A_xt)), bt)
  # arg_tangent_Ab = (None, None, A_xt, bt)

  # p, t_A = sparse_solve_value_and_jvp(arg_values, arg_tangent_A)
  # _, t_b = sparse_solve_value_and_jvp(arg_values, arg_tangent_b)
  # _, t_Ab = sparse_solve_value_and_jvp(arg_values, arg_tangent_Ab)
  # pT, t_AT = sparse_solve_value_and_jvp(arg_values, arg_tangent_A)
  # _, t_bT = sparse_solve_value_and_jvp(arg_values, arg_tangent_b)

  # pt = sparse_solve(A_indices, A_indptr, A_x, b)

  # eps = 1e-4
  # tt_A = (sparse_solve(A_indices, A_indptr, A_x + eps * A_xt, b) - pt) /eps
  # tt_b = (sparse_solve(A_indices, A_indptr, A_x, b + eps * bt) - pt) / eps
  # tt_Ab = (sparse_solve(A_indices, A_indptr, A_x + eps * A_xt, b + eps * bt) - pt) / eps

  # print(f"""
  # Sparse linear solve: 
  #   Error primal: {np.linalg.norm(p - pt): .2e}
  #   Error A varying: {np.linalg.norm(t_A - tt_A): .2e}
  #   Error b varying: {np.linalg.norm(t_b - tt_b): .2e}
  #   Error A and b varying: {np.linalg.norm(t_Ab - tt_Ab): .2e}
  # """)



  def f(theta):
    Ax_theta = jnp.array(theta[0] * A_x)
    Ax_theta = Ax_theta.at[A_indptr[:-1]].add(theta[1])
    b = jnp.ones(100)
    return sparse_solve(A_indices, A_indptr, Ax_theta, b)

  def f_jax(theta):
    Ax_theta = jnp.array(theta[0] * A.todense())
    Ax_theta = Ax_theta.at[np.arange(100),np.arange(100)].add(theta[1])
    b = jnp.ones(100)
    return jsp.linalg.solve(Ax_theta, b)

  def g(theta):
    Ax_theta = jnp.array(A_x)
    b = jnp.ones(100)
    b = b.at[0].set(theta[0])
    b = b.at[51].set(theta[1])
    return sparse_solve(A_indices, A_indptr, Ax_theta, b)

  def g_jax(theta):
    Ax_theta = jnp.array(A.todense())
    b = jnp.ones(100)
    b = b.at[0].set(theta[0])
    b = b.at[51].set(theta[1])
    return jsp.linalg.solve(Ax_theta, b)

  def h(theta):
    Ax_theta = jnp.array(A_x)
    Ax_theta = Ax_theta.at[A_indptr[:-1]].add(theta[0])
    b = jnp.ones(100)
    b = b.at[51].set(theta[1])
    return sparse_solve(A_indices, A_indptr, Ax_theta, b)

  def h_jax(theta):
    Ax_theta = jnp.array(A.todense())
    Ax_theta = Ax_theta.at[np.arange(100),np.arange(100)].add(theta[0])
    b = jnp.ones(100)
    b = b.at[51].set(theta[1])
    return jsp.linalg.solve(Ax_theta, b)

  primal1, jvp1 = jvp(f, (jnp.array([2., 3.]),), (jnp.array([1., 2.]),))
  primal2, jvp2 = jvp(f_jax, (jnp.array([2., 3.]),), (jnp.array([1., 2.]),))
  grad1 = grad(lambda x: jnp.mean(f(x)))(jnp.array([2., 3.]))
  grad2 = grad(lambda x: jnp.mean(f_jax(x)))(jnp.array([2., 3.]))

  
  primal3, jvp3 = jvp(g, (jnp.array([-142., 342.]),), (jnp.array([1., 2.]),))
  primal4, jvp4 = jvp(g_jax, (jnp.array([-142., 342.]),), (jnp.array([1., 2.]),))
  grad3 = grad(lambda x: jnp.mean(g(x)))(jnp.array([-142., 342.]))
  grad4 = grad(lambda x: jnp.mean(g_jax(x)))(jnp.array([-142., 342.]))
  
  primal5, jvp5 = jvp(h, (jnp.array([2., 342.]),), (jnp.array([1., 2.]),))
  primal6, jvp6 = jvp(h_jax, (jnp.array([2., 342.]),), (jnp.array([1., 2.]),))
  grad5 = grad(lambda x: jnp.mean(f(x)))(jnp.array([2., 342.]))
  grad6 = grad(lambda x: jnp.mean(f_jax(x)))(jnp.array([2., 342.]))
  
  print(f"""
  Check the plumbing!
  Variable A:
    Primal difference: {np.linalg.norm(primal1 - primal2)}
    JVP difference: {np.linalg.norm(jvp1 - jvp2)}
    Gradient difference: {np.linalg.norm(grad1 - grad2)}

  Variable b:
    Primal difference: {np.linalg.norm(primal3 - primal4)}
    JVP difference: {np.linalg.norm(jvp3 - jvp4)}
    Gradient difference: {np.linalg.norm(grad3 - grad4)} 

  Variable A and b:
    Primal difference: {np.linalg.norm(primal5 - primal6)}
    JVP difference: {np.linalg.norm(jvp5 - jvp6)}
    Gradient difference: {np.linalg.norm(grad5 - grad6)}
  """)

  print("""
  ========== Ok. let's try the partial inverse ========
  """)
  
  A_indices, A_indptr, A_x, A = make_matrix(15)
  n = len(A_indptr) - 1
  

  L_indices, L_indptr, L_x = sparse_cholesky(A_indices, A_indptr, A_x)

  a_inv_L = sparse_partial_inverse(L_indices, L_indptr, L_x, L_indices, L_indptr)

  col_counts_L = [L_indptr[i+1] - L_indptr[i] for i in range(n)]
  cols_L = np.repeat(range(n), col_counts_L)

  true_inv = np.linalg.inv(A.todense())
  truth_L = true_inv[L_indices, cols_L]

  a_inv_A = sparse_partial_inverse(L_indices, L_indptr, L_x, A_indices, A_indptr)
  col_counts_A = [A_indptr[i+1] - A_indptr[i] for i in range(n)]
  cols_A = np.repeat(range(n), col_counts_A)
  truth_A = true_inv[A_indices, cols_A]

  
  
  print(f"""
  Error in partial inverse (all of L): {np.linalg.norm(a_inv_L - truth_L): .2e}
  Error in partial inverse (all of A): {np.linalg.norm(a_inv_A - truth_A): .2e}
  """)

  print("""
  ========== It's log-det time ========
  """)


  #lu = sparse.linalg.splu(A)
  ld_true = np.log(np.linalg.det(A.todense())) #np.sum(np.log(lu.U.diagonal()))
  print(f"Error in log-determinant = {ld_true - sparse_log_det(A_indices, A_indptr, A_x): .2e}")

  def f(theta):
    Ax_theta = jnp.array(theta[0] * A_x, dtype = jnp.float64) / n
    Ax_theta = Ax_theta.at[A_indptr[:-1]].add(theta[1])
    return sparse_log_det(A_indices, A_indptr, Ax_theta)

  def f_jax(theta):
    Ax_theta = jnp.array(theta[0] * A.todense(), dtype = jnp.float64) / n 
    Ax_theta = Ax_theta.at[np.arange(n),np.arange(n)].add(theta[1])
    L = jnp.linalg.cholesky(Ax_theta)
    return 2.0*jnp.sum(jnp.log(jnp.diag(L)))

  primal1, jvp1 = jvp(f, (jnp.array([2., 3.], dtype = jnp.float64),), (jnp.array([1., 2.], dtype = jnp.float64),))
  primal2, jvp2 = jvp(f_jax, (jnp.array([2., 3.], dtype = jnp.float64),), (jnp.array([1., 2.], dtype = jnp.float64),))

  eps = 1e-7
  jvp_fd = (f(jnp.array([2.,3.], dtype = jnp.float64) + eps * jnp.array([1., 2.], dtype = jnp.float64) ) - f(jnp.array([2.,3.], dtype = jnp.float64))) / eps

  grad1 = grad(f)(jnp.array([2., 3.], dtype = jnp.float64))
  grad2 = grad(f_jax)(jnp.array([2., 3.], dtype = jnp.float64))

  print(f"""
  primal1: {primal1} {f(jnp.array([2.,3.], dtype = jnp.float64))}
  jvp1: {jvp1}
  primal2: {primal2}  {f_jax(jnp.array([2.,3.], dtype = jnp.float64))}
  jvp2: {jvp2}
  jvp_fd: {jvp_fd}
  frad1: {grad1}
  grad2: {grad2}
  
  """)

  print(f"""
  Check the Derivatives!
  Variable A:
    Primal difference: {np.linalg.norm(primal1 - primal2)}
    JVP difference: {np.linalg.norm(jvp1 - jvp2)}
    JVP difference (FD): {np.linalg.norm(jvp1 - jvp_fd)}
    Gradient difference: {np.linalg.norm(grad1 - grad2)}
  """)