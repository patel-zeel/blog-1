//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
   int N;
   int y_spec;
   int y[N];
   int n_spec;
   int n[N];
   real alpha_spec;
   real beta_spec;
   real<lower= 0> sd;
  
}


parameters {
  real<lower = 0, upper = 1> gamma;
  real mu;
}
transformed parameters{
  real pi = 1/(1+exp(-mu));
  real p = pi*0.8 + (1-pi)*(1-gamma);
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  y ~ binomial_logit(n, mu);
  y_spec ~ binomial(n_spec, gamma);
  mu ~ normal(0, sd);
  gamma ~ beta(alpha_spec, beta_spec);
}

