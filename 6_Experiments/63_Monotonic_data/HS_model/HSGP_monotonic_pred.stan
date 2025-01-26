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
    
    vector psi(array[] real x, int n, real i, real j, real L) {
        real lam_diff = lambda_sqrt(i, L) - lambda_sqrt(j, L);
        real lam_sum = lambda_sqrt(i, L) + lambda_sqrt(j, L);

        vector[n] xpL = to_vector(x) + L;

        if (i == j) {
            return 0.5*(xpL) - 1./(4*lambda_sqrt(j, L))*sin(2*lambda_sqrt(j, L)*(xpL));
        }
        else  {
            return 1./(2*lam_diff)*sin(lam_diff*(xpL)) - 1./(2*lam_sum)*sin(lam_sum*(xpL));
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

    int<lower=1> num_pred;
    array[num_pred] real xpred;
    

    real<lower=0> jitter;
    
    int num_basis_functions;
    real L;
}

transformed data 
{
    array[num_basis_functions, num_basis_functions] vector[num_train] PSI_train;
    array[num_basis_functions, num_basis_functions] vector[num_test] PSI_test;
    array[num_basis_functions, num_basis_functions] vector[num_pred] PSI_pred;
    vector[num_basis_functions] lambda_sqrt_vec;
        
    for (j in 1:num_basis_functions) {
        lambda_sqrt_vec[j] = lambda_sqrt(j, L);
    }
    
    for (i in 1:num_basis_functions) {
        for (j in 1:num_basis_functions) {
            PSI_train[i, j] = psi(xtrain, num_train, i, j, L);
            PSI_test[i,j] = psi(xtest, num_test, i, j, L);
            PSI_pred[i,j] = psi(xpred, num_pred, i, j, L);

        }
    }
}
parameters {
    vector[num_basis_functions] alpha;  
    real<lower=jitter> kappa;
    real<lower=jitter> scale;
    real<lower=jitter> sigma;
    real<lower=jitter> f0_param;
    real f0;
}
transformed parameters {
    vector[num_train] f_train;
    vector[num_basis_functions] alpha_scaled;
    vector[num_basis_functions] lambda;
    
    lambda =  sqrt(spectral_density(lambda_sqrt_vec, kappa, scale));
    
    alpha_scaled = lambda .* alpha;
    
    f_train = rep_vector(0, num_train);    
    for (j in 1:num_basis_functions) {
        for (i in 1:num_basis_functions ) {
            f_train += alpha_scaled[j]*alpha_scaled[i]*PSI_train[i, j]/L;
        }
    }
    f_train += f0;
    f_train = -f_train;
}
model {
    alpha ~ std_normal();
    kappa ~ std_normal();
    scale ~ std_normal();
    sigma ~ std_normal();
    f0_param ~ std_normal();
    f0 ~ student_t(4, 0, f0_param);

    ytrain ~ normal(f_train,sigma);

}
generated quantities {
    vector[num_test] f_test;
    vector[num_pred] f_pred;
    
    f_test = rep_vector(0, num_test);
    f_pred = rep_vector(0, num_pred);

    for (j in 1:num_basis_functions) {
        for (i in 1:num_basis_functions ) {
            f_test += alpha_scaled[j]*alpha_scaled[i]*PSI_test[i, j]/L;
            f_pred += alpha_scaled[j]*alpha_scaled[i]*PSI_pred[i, j]/L;
        }
    }
    
    f_test += f0;
    f_pred += f0;

    f_test = -f_test;
    f_pred = -f_pred;
    
}