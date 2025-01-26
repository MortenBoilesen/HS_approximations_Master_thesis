functions {
    
    vector spectral_density(vector omega, real kappa, real scale) {
        return kappa^2*sqrt(2*pi()*scale^2)*exp(omega .* omega  * (-0.5)*scale^2);
    }
    
    real lambda_sqrt(real j, real L) {
        return j*pi()/(2*L);
    }
    
    vector phi(array[] real x, real j, real L) {
        return 1./sqrt(L)*sin(lambda_sqrt(j, L)*(to_vector(x) + L));
    }
    
    vector Psi(array[] real x, int n, real i, real j, real L) {
        real gamma_minus = lambda_sqrt(i, L) - lambda_sqrt(j, L);
        real gamma_plus = lambda_sqrt(i, L) + lambda_sqrt(j, L);
        
        vector[n] xpL = to_vector(x) + L;

        if (i == j) {
            return (xpL.*xpL)/4 + (cos(gamma_plus*xpL) - 1)/(2*gamma_plus*gamma_plus);

        }
        else  {
            return (1 - cos(gamma_minus*xpL))/(2*gamma_minus*gamma_minus) + (cos(gamma_plus*xpL) - 1)/(2*gamma_plus*gamma_plus);
        }
    }
}
data {
    int<lower=1> num_test;
    array[num_test] real xtest;
    array[num_test] real ytest;
    
    int<lower=1> num_train;
    array[num_train] real xtrain;
    array[num_train] real ytrain;

    real<lower=0> jitter;
    
    real F0_prior_mean;

    int num_basis_functions;
    real L;
}
transformed data 
{
    array[num_basis_functions, num_basis_functions] vector[num_train] PSI_train;
    array[num_basis_functions, num_basis_functions] vector[num_test] PSI_test;
    vector[num_basis_functions] lambda_sqrt_vec;
        
    for (j in 1:num_basis_functions) {
        lambda_sqrt_vec[j] = lambda_sqrt(j, L);
    }
    
    for (i in 1:num_basis_functions) {
        for (j in 1:num_basis_functions) {
            PSI_train[i, j] = Psi(xtrain, num_train, i, j, L);
            PSI_test[i, j] = Psi(xtest, num_test, i, j, L);
        }
    }

}
parameters {
    vector[num_basis_functions] alpha;  
    real<lower=jitter> kappa;
    real<lower=jitter> scale;
    real<lower=jitter> sigma;
    real<lower=jitter> F0_param;
    real<lower=jitter> f0_param;
    real F0;
    real<lower=jitter> f0;

}
transformed parameters {
    vector[num_train] f_train;
    vector[num_basis_functions] alpha_scaled;
    vector[num_basis_functions] lambda;
    vector[num_train] f0_train;

    lambda =  sqrt(spectral_density(lambda_sqrt_vec, kappa, scale));
    
    alpha_scaled = lambda .* alpha;

    f0_train = -f0*(to_vector(xtrain) + L);

    f_train = rep_vector(0, num_train);    
    for (j in 1:num_basis_functions) {
        for (i in 1:num_basis_functions ) {
            f_train += alpha_scaled[j]*alpha_scaled[i]*PSI_train[i, j]/L;
        }
    }
    f_train += F0_prior_mean;
    f_train += F0;
    f_train += f0_train;
}
model {
    alpha ~ std_normal();
    kappa ~ std_normal();
    scale ~ std_normal();
    sigma ~ std_normal();
    F0_param ~ std_normal();
    f0_param ~ std_normal();
    F0 ~ student_t(4, 0, F0_param);
    f0 ~ lognormal(0, f0_param);
    
    ytrain ~ normal(f_train,sigma);

}
generated quantities {
    vector[num_test] f0_test;
    vector[num_test] f_test;

    f0_test = -f0*(to_vector(xtest) + L);

    f_test = rep_vector(0, num_test);   

    for (j in 1:num_basis_functions) {
        for (i in 1:num_basis_functions ) {
            f_test += alpha_scaled[j]*alpha_scaled[i]*PSI_test[i, j]/L;
        }
    }
    
    f_test += F0_prior_mean;
    f_test += F0; 
    f_test += f0_test;
}

