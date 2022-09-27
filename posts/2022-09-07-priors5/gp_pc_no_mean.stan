functions {
  matrix cov(int N, matrix s, real sigma, real ell) {
    matrix[N,N] R;
    row_vector[2] s1, s2;
    real sigma2 = sigma * sigma;
    for (i in 1:N) {
      for (j in 1:N){
        s1 = s[i, 1:2];
        s2 = s[j, 1:2];
        R[i,j] = sigma2 * exp(-sqrt(dot_self(s1-s2))/ell);
      }
    }
    return 0.5 * (R + R');
  }
}

data {
  int<lower=0> N;
  vector[N] y;
  matrix[N,2] s;
  real<lower = 0> lambda_ell;
  real<lower = 0> lambda_sigma;
}

parameters {
  real<lower=0> sigma;
  real<lower=0> ell;
}

model {
  matrix[N,N] R = cov(N, s, sigma, ell);
  y ~ multi_normal(rep_vector(0.0, N), R);
  sigma ~ exponential(lambda_sigma);
  ell ~ frechet(1, lambda_ell); // Only in 2D
}

// generated quantities {
//   real check = 0.0; // should be the same as lp__
//   { // I don't want to print R!
//     matrix[N,N] R = cov(N, s, sigma, ell);
//     check -= 0.5* dot_product(y,(R\ y)) + 0.5 * log_determinant(R);
//     check += log(sigma) - lambda_sigma * sigma;
//     check += log(ell) - 2.0 * log(ell) - lambda_ell / ell;
//   }
// }
