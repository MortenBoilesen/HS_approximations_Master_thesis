functions {
    matrix pairwise_difference(array[] real x1, array[] real x2, int N1, int N2){
        matrix[N1, N2] diff;
        diff = rep_matrix(to_vector(x1),N2) - rep_matrix(to_vector(x2)', N1);
        return diff;
    }
    matrix dx_x_cov_exp_quad(array[] real xvirtual, array[] real x, real kappa, real scale, matrix pair_diff) {
        return -pair_diff ./ scale^2 .* gp_exp_quad_cov(xvirtual, x, kappa, scale);
    }
    matrix cov_exp_quad_full(array[] real x, array[] real xvirtual, int N, int num_virtual, real kappa, real scale, matrix dist_xvirtual, matrix pair_diff) {
        matrix[N + num_virtual, N + num_virtual] Sigma;
        matrix[num_virtual, N] Sigma_dx_d = dx_x_cov_exp_quad(xvirtual, x, kappa, scale, pair_diff);
        
        Sigma[1:N,1:N] = gp_exp_quad_cov(x, kappa, scale);
        Sigma[N+1:N+num_virtual, 1:N] = Sigma_dx_d;
        Sigma[1:N, N+1:N+num_virtual] = Sigma_dx_d';

        Sigma[N+1:N+num_virtual, N+1:N+num_virtual] = gp_exp_quad_cov(xvirtual, kappa, scale) .* (1 ./ scale^2 -dist_xvirtual .* dist_xvirtual ./ scale^4);
        
        return Sigma;
    }
    
    vector gp_pred_rng(array[] real x2,
                       vector y1,
                       array[] real x1,
                       array[] real xvirtual,
                       real kappa,
                       real rho,
                       real delta,
                       int num_train,
                       int num_virtual,
                       int num_test,
                       matrix dist_xvirtual, matrix diff_virtual_train) {
      int N1 = num_train + num_virtual;
      int N2 = num_test;
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
        matrix[num_virtual, N2] pair_diff = pairwise_difference(xvirtual, x2, num_virtual, N2);
        
        // joint covariance of train+virtual
        K = cov_exp_quad_full(x1, xvirtual, num_train, num_virtual, kappa, rho, dist_xvirtual, diff_virtual_train);

        for (n in 1:N1)
          K[n, n] = K[n,n] + delta;
        L_K = cholesky_decompose(K);
        L_div_y1 = mdivide_left_tri_low(L_K, y1);
        
        // cross covariance between train+virtual and test
            
        k_x1_x2[1:num_train, :] = gp_exp_quad_cov(x1, x2, kappa, rho);
        k_x1_x2[num_train+1:num_train+num_virtual, :] = dx_x_cov_exp_quad(xvirtual, x2, kappa, rho, pair_diff);

        
        
        L_div_k_x1_x2 = mdivide_left_tri_low(L_K, k_x1_x2);
        f2_mu = L_div_k_x1_x2' * L_div_y1;
        cov_f2 = gp_exp_quad_cov(x2, kappa, rho) - L_div_k_x1_x2' * L_div_k_x1_x2;
        diag_delta = diag_matrix(rep_vector(delta,N2));
  
        f2 = multi_normal_rng(f2_mu, cov_f2 + diag_delta);
      }
    return f2;
  }
}
data {
    int<lower=1> num_train;
    array[num_train] real xtrain;
    array[num_train] real ytrain;

    int<lower=1> num_test;
    array[num_test] real xtest;
    array[num_test] real ytest;

    int<lower=1> num_virtual;
    array[num_virtual] real xvirtual;
    array[num_virtual] int yvirtual;

    real nu;
    real jitter;
}
transformed data {
    // Pre-compute pair-wise differences
    matrix[num_virtual, num_virtual] dist_xvirtual = pairwise_difference(xvirtual, xvirtual, num_virtual, num_virtual);
    matrix[num_virtual, num_train] diff_virtual_train = pairwise_difference(xvirtual, xtrain, num_virtual, num_train);
}
parameters {
    real <lower=jitter> sigma;
    real<lower=jitter> scale;
    real<lower=jitter> kappa;
    real f0;
    
    vector[num_train+num_virtual] eta;
}
transformed parameters {
    vector[num_train+num_virtual] f_train;
    matrix[num_train+num_virtual, num_train+num_virtual] L;
    matrix[num_train+num_virtual, num_train+num_virtual] K;
    
    K = cov_exp_quad_full(xtrain ,xvirtual, num_train, num_virtual, kappa, scale, dist_xvirtual, diff_virtual_train);
    for (n in 1:(num_train+num_virtual)){
        K[n, n] = K[n, n] + jitter;
    }

    L = cholesky_decompose(K);        
    f_train = L * eta;
}
model {
    scale ~ inv_gamma(5, 5);
    kappa ~ std_normal();
    sigma ~ std_normal();
    eta ~ std_normal();
    f0 ~ std_normal();
    
    ytrain ~ normal(f0 + f_train[1:num_train], sigma);
    for (n in 1:num_virtual)
        yvirtual[n] ~ bernoulli(Phi(nu*f_train[num_train+n]));
}

generated quantities {
vector[num_test] f_test;
vector[num_test] y_test;

f_test = gp_pred_rng(xtest, f_train, xtrain, xvirtual, kappa, scale,  jitter, num_train,  num_virtual, num_test, dist_xvirtual, diff_virtual_train);
}
