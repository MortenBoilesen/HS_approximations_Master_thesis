functions {
  vector gp_pred_rng(array[] real x2, vector y1, array[] real x1, real kappa,
                     real scale, real sigma, real delta) {
    int N1 = rows(y1);
    int N2 = size(x2);
    vector[N2] f2;
    
    {
      matrix[N1, N1] L_K;
      vector[N1] L_div_y1;
      matrix[N1, N2] k_x1_x2;
      matrix[N1, N2] L_div_k_x1_x2;
      vector[N2] f2_mu;
      matrix[N2, N2] cov_f2;
      matrix[N2, N2] diag_delta;
      matrix[N1, N1] K;
      K = gp_exp_quad_cov(x1, kappa, scale);
      for (n in 1 : N1) {
        K[n, n] = K[n, n] + square(sigma);
      }
      L_K = cholesky_decompose(K);
      L_div_y1 = mdivide_left_tri_low(L_K, y1);
      k_x1_x2 = gp_exp_quad_cov(x1, x2, kappa, scale);
      L_div_k_x1_x2 = mdivide_left_tri_low(L_K, k_x1_x2);
      f2_mu = L_div_k_x1_x2' * L_div_y1;
      cov_f2 = gp_exp_quad_cov(x2, kappa, scale)
               - L_div_k_x1_x2' * L_div_k_x1_x2;
      diag_delta = diag_matrix(rep_vector(delta, N2));
      
      f2 = multi_normal_rng(f2_mu, cov_f2 + diag_delta);
    }
    return f2;
  }
}
data {
  int<lower=1> num_train;
  array[num_train] real xtrain;
  vector[num_train] ytrain;

  int<lower=1> num_test;
  array[num_test] real xtest;
  vector[num_test] ytest;

  int<lower=1> num_pred;
  array[num_pred] real xpred;
}
transformed data {
  vector[num_train] mu = rep_vector(0, num_train);
  real delta = 1e-9;
}
parameters {
  real<lower=0> scale;
  real<lower=0> kappa;
  real<lower=0> sigma;
}
model {
  matrix[num_train, num_train] L_K;
  {
    matrix[num_train, num_train] K = gp_exp_quad_cov(xtrain, kappa, scale);
    real sq_sigma = square(sigma);
    
    // diagonal elements
    for (n in 1 : num_train) {
      K[n, n] = K[n, n] + sq_sigma + delta;
    }
    
    L_K = cholesky_decompose(K);
  }
  
  scale ~ inv_gamma(5, 5);
  kappa ~ normal(0, 1);
  sigma ~ normal(0, 1);
  
  ytrain ~ multi_normal_cholesky(mu, L_K);
}
generated quantities {
  vector[num_test] f_test;
  vector[num_pred] f_pred;
  
  f_test = gp_pred_rng(xtest, ytrain, xtrain, kappa, scale, sigma, delta);
  f_pred = gp_pred_rng(xpred, ytrain, xtrain, kappa, scale, sigma, delta);

}