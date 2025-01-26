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

    vector Phi(array[] real x, real j, real L) {
        real sqrt_lambda = lambda_sqrt(j, L);
        return (1 - cos(sqrt_lambda*(to_vector(x) + L)))./sqrt(L)./sqrt_lambda;
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

    int<lower=1> num_pred;
    array[num_pred] real xpred;

    real<lower=0> jitter;
    
    int num_basis_functions;
    real L;
}
transformed data 
{
    array[num_basis_functions] vector[num_train] Phi_train;
    array[num_basis_functions, num_basis_functions] vector[num_train] PSI_train;

    array[num_basis_functions] vector[num_test] Phi_test;
    array[num_basis_functions, num_basis_functions] vector[num_test] PSI_test;

    array[num_basis_functions] vector[num_pred] phi_pred;
    array[num_basis_functions] vector[num_pred] Phi_pred;
    array[num_basis_functions, num_basis_functions] vector[num_pred] Psi_pred;
    array[num_basis_functions, num_basis_functions] vector[num_pred] PSI_pred;

    vector[num_basis_functions] lambda_sqrt_vec;
        
    for (i in 1:num_basis_functions) {
        lambda_sqrt_vec[i] = lambda_sqrt(i, L);

        Phi_train[i] = Phi(xtrain, i, L);

        Phi_test[i] = Phi(xtest, i, L);

        phi_pred[i] = phi(xpred, i, L);
        Phi_pred[i] = Phi(xpred, i, L);

        for (j in 1:num_basis_functions) {
            PSI_train[i, j] = Psi(xtrain, num_train, i, j, L);

            PSI_test[i, j] = Psi(xtest, num_test, i, j, L);

            Psi_pred[i, j] = psi(xpred, num_pred, i, j, L);
            PSI_pred[i, j] = Psi(xpred, num_pred, i, j, L);
        }
    }

}
parameters {
    vector[num_basis_functions] alpha;  
    vector[num_basis_functions] beta;  
    real<lower=jitter> kappa_f;
    real<lower=jitter> scale_f;
    real<lower=jitter> kappa_g;
    real<lower=jitter> scale_g;
    real<lower=jitter> sigma;
    real<lower=jitter> F0_param;
    real<lower=jitter> f0_param;
    real F0;
    real<lower=jitter> f0;

}
transformed parameters {
    vector[num_train] f_train;
    vector[num_train] Gp_train;
    vector[num_basis_functions] alpha_scaled;
    vector[num_basis_functions] beta_scaled;
    vector[num_basis_functions] lambda;
    vector[num_basis_functions] gamma;
    vector[num_train] f0_train;

    lambda =  sqrt(spectral_density(lambda_sqrt_vec, kappa_f, scale_f));
    gamma =  sqrt(spectral_density(lambda_sqrt_vec, kappa_g, scale_g));
    
    alpha_scaled = lambda .* alpha;
    beta_scaled = gamma .* beta;


    f0_train = -f0*(to_vector(xtrain) + L);

    f_train = rep_vector(0, num_train);
    Gp_train = rep_vector(0, num_train);    

    for (j in 1:num_basis_functions) {
        Gp_train += beta_scaled[j]*Phi_train[j];

        for (i in 1:num_basis_functions ) {
            f_train += alpha_scaled[j]*alpha_scaled[i]*PSI_train[i, j]/L;
        }
    }
    f_train += F0;
    f_train += f0_train;
    f_train += Gp_train;
    f_train = -f_train;
}
model {
    alpha ~ std_normal();
    beta ~ std_normal();
    kappa_f ~ std_normal();
    scale_f ~ std_normal();
    kappa_g ~ std_normal();
    scale_g ~ std_normal();
    sigma ~ std_normal();
    F0_param ~ std_normal();
    f0_param ~ std_normal();
    F0 ~ student_t(4, 0, F0_param);
    f0 ~ lognormal(0, f0_param);
    
    ytrain ~ normal(f_train, sigma);

}
generated quantities {
    vector[num_test] f0_test;
    vector[num_test] f_test;
    vector[num_test] Gp_test;

    vector[num_pred] f0_pred;
    vector[num_pred] f_pred;
    vector[num_pred] Gp_pred;
    vector[num_pred] gp_pred;
    vector[num_pred] f_pred_d;

    f0_test = -f0*(to_vector(xtest) + L);
    f0_pred = -f0*(to_vector(xpred) + L);

    f_test = rep_vector(0, num_test);   
    Gp_test = rep_vector(0, num_test);   
    
    f_pred = rep_vector(0, num_pred);
    Gp_pred = rep_vector(0, num_pred);
    gp_pred = rep_vector(0, num_pred);
    f_pred_d = rep_vector(0, num_pred);

    for (j in 1:num_basis_functions) {
        Gp_test += beta_scaled[j]*Phi_test[j];
        Gp_pred += beta_scaled[j]*Phi_pred[j];
        gp_pred += beta_scaled[j]*phi_pred[j];
        
        for (i in 1:num_basis_functions ) {
            f_test += alpha_scaled[j]*alpha_scaled[i]*PSI_test[i, j]/L;
            f_pred += alpha_scaled[j]*alpha_scaled[i]*PSI_pred[i, j]/L;
            f_pred_d += alpha_scaled[j]*alpha_scaled[i]*Psi_pred[i, j]/L;
        }
    }
    
    f_test += F0; 
    f_test += f0_test;
    f_test += Gp_test;
    f_test = -f_test;
    
    f_pred += F0; 
    f_pred += f0_pred;
    f_pred += Gp_pred;
    f_pred = -f_pred;

    f_pred_d -= f0;
    f_pred_d += gp_pred;
    f_pred_d = -f_pred_d;

}


