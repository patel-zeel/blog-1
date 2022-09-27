functions {
  matrix cov(int N, matrix s,  real ell) {
    matrix[N,N] R;
    row_vector[2] s1, s2;
    for (i in 1:N) {
      for (j in 1:N){
        s1 = s[i, 1:2];
        s2 = s[j, 1:2];
        R[i,j] = exp(-sqrt(dot_self(s1-s2))/ell);
      }
    }
    return 0.5 * (R + R');
  }
  matrix cov_diff(int N, matrix s,  real ell) {
    // dR /d ell = cov(N, p ,s, sigma2*|x-y|/ell^2, ell)
    matrix[N,N] R;
    row_vector[2] s1, s2;
    for (i in 1:N) {
      for (j in 1:N){
        s1 = s[i, 1:2];
        s2 = s[j, 1:2];
        R[i,j] =  sqrt(dot_self(s1-s2)) * exp(-sqrt(dot_self(s1-s2))/ell) / ell^2 ;
      }
    }
    return 0.5 * (R + R');
  }

  real log_prior(int N, matrix s, real sigma2, real ell) {
    matrix[N,N] R = cov(N, s,  ell);
    matrix[N,N] W = (cov_diff(N, s, ell)) / R;
    return 0.5 * log(trace(W * W) - (1.0 / (N)) * (trace(W))^2) - log(sigma2);
  }
}

data {
  int<lower=0> N;
  vector[N] y;
  matrix[N,2] s;
}

parameters {
  real<lower=0> sigma2;
  real<lower=0> ell;
}

model {
  {
    matrix[N,N] R = cov(N, s, ell);
    target += multi_normal_lpdf(y | rep_vector(0.0, N), sigma2 * R);
  }
  target += log_prior(N,  s, sigma2, ell);
}

generated quantities {
  real sigma = sqrt(sigma2);
}
