import autograd.numpy as np
import itertools
from scipy.special import gamma
from scipy.optimize import minimize
from autograd import value_and_grad

class HSGP_box_domain:

    def __init__(self, m, D, spectral_density, theta, spectral_density_grad = None): # Constructor
        """
        Arguments:
        m [int]: number of basis functions
        D [list]: domain of the input variable
        spectral_density [function]: spectral density of the GP
        theta [list]: the GP hyperparameters. ON the form [sigma, theta_1, ..., theta_p]
        """
        self.m = m
        self.D = D
        self.L = 0.5*(D[1] - D[0])
        self.sqrtL = np.sqrt(self.L)
        self.twoL = 2*self.L
        self.shift = 0.5*(self.D[0] + self.D[1])

        self.spectral_density = spectral_density
        self.spectral_density_grad = spectral_density_grad
        self.sigma = theta[0]
        self.sigma2 = self.sigma*self.sigma
        self.theta_k = theta[1:]

        self.js = np.arange(1,m+1)

    def phi(self,j,x):
        """
        Arguments:
        j [int]: index of the basis function
        X: input variable of shape (n,1)
        """

        return 1/self.sqrtL*np.sin(np.pi*j*(x-self.D[0])/self.twoL) * (np.abs(x-self.shift) < self.L)

    def sqrt_lambda(self,j):
        return j*np.pi/self.twoL

    def construct_phi(self, X):   
        """
        Arguments:
        X: input variable of shape (n,1)
        """
        Phi = self.phi(self.js, X)
        return Phi

    def construct_Lambda(self, theta_k):
        sqrt_lambdas = self.sqrt_lambda(self.js)
        Lambda = self.spectral_density(sqrt_lambdas, theta_k)
        return Lambda

    def fit(self, X, y): # Fit the model
        """
        Arguments:
        X: input variable of shape (n,1)
        y: output variable of shape (n,1)

        returns: mu_post, sigma_post_inv
        """
        self.Phi = self.construct_phi(X)
        self.Lambda = self.construct_Lambda(self.theta_k)

        K = self.Phi.T @ self.Phi
        sqrt_W = 1/self.sigma*np.diag(np.sqrt(self.Lambda))

        B = sqrt_W @ K @ sqrt_W + np.identity(self.m)
        L = np.linalg.cholesky(B)

        v = np.linalg.solve(L, sqrt_W )

        self.sigma_post =  v.T @ v
        self.mu_post = self.sigma_post @ self.Phi.T @ y
        
        return
        # return self.mu_post, self.sigma_post_inv

    def predict(self, Xpred):
        """
        Arguments:
        Xpred: input variable of shape (k,1)

        returns: mu_pred, sigma_pred
        """
        phi_pred = self.phi(self.js, Xpred) # obs tranponeret ifht artikel

        sigma_pred = self.sigma2 * phi_pred @ self.sigma_post @ phi_pred.T
        mu_pred = phi_pred @ self.mu_post

        return mu_pred, sigma_pred
    
    def generate_samples(self, Xpred, num_samples, return_distribution = False):
        """
        Arguments:
        Xpred: input variable of shape (k,1)
        num_samples: number of samples (scalar)
        """
        mu_pred, Sigma_pred = self.predict(Xpred)

        eps =  1e-8
        L = np.linalg.cholesky(Sigma_pred + eps*np.identity(len(Sigma_pred)))
        z = np.random.normal(0, 1, size=(len(Sigma_pred), num_samples))
        f = (mu_pred + L@z).squeeze()

        if return_distribution:
            return f, mu_pred.squeeze(), Sigma_pred
        else: return f
    
    def neg_lml_value_and_grad(self, Phi, PhiTPhi, y, log_theta, lml_only = False):
        n = len(y)

        # extract hyperparameters
        sigma = np.exp(log_theta[0])
        sigma2 = sigma*sigma
        theta_K = np.exp(log_theta[1:])

        sqrt_lambdas = self.sqrt_lambda(self.js)
        Lambda = self.spectral_density(sqrt_lambdas, theta_K)

        Z = sigma2*np.diag(1/Lambda) + PhiTPhi
        Z_cho = np.linalg.cholesky(Z)
        v = np.linalg.solve(Z_cho, Phi.T@y)

        yty = np.sum(y**2)
        vtv = np.sum(v**2)

        logdet_lml = 2*(n-self.m)*np.log(sigma) + 2*np.sum(np.log(np.diag(Z_cho))) + np.sum(np.log(Lambda))
        quad_lml = (yty - vtv)/sigma2
        const_lml = n*np.log(2*np.pi)

        neg_lml = 0.5*(const_lml + logdet_lml + quad_lml)

        if lml_only:
            return neg_lml

        # Gradient
        theta_k_grad = self.spectral_density_grad(sqrt_lambdas, theta_K)
        w = np.linalg.solve(Z_cho.T, v)
        ZLinv = np.linalg.solve(Z_cho.T, np.linalg.solve(Z_cho, np.diag(1/Lambda)))
       
        lml_dtheta = np.zeros(len(log_theta))

        # Compute derirative with respect to sigma2
        dlogdet_dsigma = 0.5*( (n-self.m)/(sigma2) + np.sum( np.diag(ZLinv) ) )
        dquad_dsigma = np.sum(0.5*( w.T@np.diag(1/Lambda)@w/sigma2 - (yty - vtv)/(sigma2*sigma2)))
        lml_dtheta[0] = 2*(dlogdet_dsigma + dquad_dsigma)*np.exp(log_theta[0])

        for i, theta in enumerate(theta_K):
            logdet_dtheta = 0.5*( np.sum( theta_k_grad[i]/Lambda ) - sigma2*np.sum( np.diag(ZLinv)*theta_k_grad[i]/Lambda ))
            quad_dtheta = np.sum(-0.5*w.T@np.diag(theta_k_grad[i]/(Lambda**2))@w)
            lml_dtheta[i+1] = (logdet_dtheta + quad_dtheta)*theta       

        return [neg_lml, lml_dtheta]

    def optimize_hyperparameters(self, X, y, theta_init):
        Phi = self.phi(self.js, X)
        PhiTPhi = Phi.T @ Phi    

        if self.spectral_density_grad is None:
            print('Warning: No gradient function provided. Using autograd.')
            
            objective = lambda log_th: self.neg_lml_value_and_grad(Phi, PhiTPhi, y, log_th, lml_only = True)
            res = minimize(value_and_grad(objective), np.log(theta_init), jac=True)

        else:
            objective = lambda log_th: self.neg_lml_value_and_grad(Phi, PhiTPhi, y, log_th)
            res = minimize(objective, np.log(theta_init), jac=True)

        # check for success
        if not res.success:
            print('Warning: optimization failed!')
            return 0

        # return results
        log_theta = res.x
        theta = np.exp(log_theta)

        self.sigma = theta[0]
        self.sigma2 = self.sigma*self.sigma
        self.theta_k = theta[1:]
        
        return theta

class HSGP_box_domain_multidim:
    def __init__(self, m, D, spectral_density, theta): # Constructor
        """
        Arguments:
        m [int]: number of basis functions
        D [list]: domain of the input variable
        spectral_density [function]: spectral density of the GP
        theta [list]: the GP hyperparameters. ON the form [sigma, theta_1, ..., theta_p]
        """
        self.m = m
        self.D = D
        self.d = len(D)
        self.L = 0.5*(D[:,1] - D[:,0])
        self.sqrtL = np.sqrt(self.L)
        self.twoL = 2*self.L
        self.shift = 0.5*(self.D[:,0] + self.D[:,1])

        self.spectral_density = spectral_density
        self.sigma = theta[0]
        self.sigma2 = self.sigma*self.sigma
        self.theta_k = theta[1:]

        self.js = np.tile(np.arange(1, m+1), (self.d, 1)).T
        self.combs = np.array(list(itertools.product(*(self.js-1).T)))
        self.xidx = np.arange(0, self.d)

    def phi(self,j,x,L):
        """
        Arguments:
        j [int]: index of the basis function
        X: input variable of shape (n,1)
        L: length of the box domain
        """

        return 1/np.sqrt(L)*np.sin(np.pi*j*(x+L)/2/L) * (np.abs(x) < L).all()

    def lambdas(self,j,L):
        return (j*np.pi/(2*L))**2

    def construct_Phi(self, X):     # what happens when X is multidimensional?
        """
        Arguments:
        X: input variable of shape (n,1)
        """
        phis = self.phi(self.js, X, self.L)
        all_phis = phis[:,self.combs,self.xidx]
        return all_phis.prod(axis=-1)

    def construct_Lambda(self, theta_k):
        lambdas = self.lambdas(self.js, self.L)
        lambdas_comb = lambdas[self.combs,self.xidx].sum(axis=-1)
        self.Lambda = self.spectral_density(np.sqrt(lambdas_comb), self.d, theta_k)

    def fit(self, X, y): # Fit the model
        """
        Arguments:
        X: input variable of shape (n,1)
        y: output variable of shape (n,1)

        returns: mu_post, sigma_post_inv
        """
        X = X.reshape(-1, 1, self.d)
        self.Phi = self.construct_Phi(X)
        self.construct_Lambda(self.theta_k)

        K = self.Phi.T @ self.Phi
        sqrt_W = 1/self.sigma*np.diag(np.sqrt(self.Lambda))
        B = sqrt_W @ K @ sqrt_W + np.identity(self.m**self.d)
        L = np.linalg.cholesky(B)
        v = np.linalg.solve(L, sqrt_W )

        self.sigma_post =  v.T @ v
        self.mu_post = self.sigma_post @ self.Phi.T @ y

    def predict(self, Xpred):
        """
        Arguments:
        Xpred: input variable of shape (k,1)

        returns: mu_pred, sigma_pred
        """
        Xpred = Xpred.reshape(-1, 1, self.d)
        phi_pred = self.construct_Phi(Xpred)

        sigma_pred = self.sigma2*phi_pred @ self.sigma_post @ phi_pred.T
        mu_pred = phi_pred @ self.mu_post

        return mu_pred, sigma_pred


class GP_true:
    def __init__(self, kernel_function, theta): # Constructor
        """
        Arguments:
        kernel [function]: Stationary covariance function of the GP as a function of tau = ||x - x' ||^2
        theta [list]: the GP hyperparameters. ON the form [sigma, theta_1, ..., theta_p]
        """

        self.kernel_function = kernel_function
        self.sigma = theta[0]
        self.sigma2 = self.sigma*self.sigma
        self.theta_k = theta[1:]

    def kernel(self, X1, X2, theta_k):
        """
        Arguments:
        X1: input variable of shape (n,1)
        X2: input variable of shape (m,1)
        theta: hyperparameters

        returns: K
        """
        tau_squared = np.sum((np.expand_dims(X1, 1) - np.expand_dims(X2, 0))**2, axis=-1)

        # ad jitter ??

        return self.kernel_function(tau_squared, *theta_k)

    def fit(self, X, y): # Fit the model
        """
        Arguments:
        X: input variable of shape (n,1)
        y: output variable of shape (n,1)

        returns: mu_post, sigma_post_inv
        """
        self.X = X
        self.K = self.kernel(X,X,self.theta_k)

        sqrt_W = 1/self.sigma*np.identity(len(X))
        B = sqrt_W @ self.K @ sqrt_W + np.identity(len(X))
        L = np.linalg.cholesky(B)
        v = np.linalg.solve(L, sqrt_W)

        self.sigma_post =  v.T @ v
        self.mu_post = self.sigma_post @ y

        return
        # return self.mu_post, self.sigma_post_inv

    def predict(self, Xpred):
        """
        Arguments:
        Xpred: input variable of shape (k,1)

        returns: mu_pred, sigma_pred
        """
        self.k = self.kernel(Xpred,self.X,self.theta_k)
        self.Kp = self.kernel(Xpred,Xpred,self.theta_k)

        mu_pred = np.dot(self.k, self.mu_post)
        sigma_pred = self.Kp - self.k @ self.sigma_post @ self.k.T

        return mu_pred, sigma_pred

    def generate_samples(self, Xpred, num_samples, return_distribution=False):
        """
        Arguments:
        Xpred: input variable of shape (k,1)
        num_samples: number of samples (scalar)

        Output:
        f: saples of shape (num_samples, k) 
        """
        mu_pred, Sigma_pred = self.predict(Xpred)

        eps =  1e-8
        L = np.linalg.cholesky(Sigma_pred + eps*np.identity(len(Sigma_pred)))
        z = np.random.normal(0, 1, size=(num_samples,*mu_pred.shape))
        f = (mu_pred + L@z).squeeze()
        
        if return_distribution:
            return f, mu_pred.squeeze(), Sigma_pred
        else: return f
    
    def log_marginal_likelihood(self, X, y, log_theta):

        n = len(y)

        # extract hyperparameters
        sigma = np.exp(log_theta[0])
        theta_K = np.exp(log_theta[1:])
        
        # prepare kernels
        K = self.kernel(X, X, theta_K)
        C = K + sigma**2*np.identity(n)

        # compute Cholesky decomposition - this will be covered in week 5.
        L = np.linalg.cholesky(C)
        v = np.linalg.solve(L, y)

        # compute log marginal likelihood
        logdet_lml = np.sum(np.log(np.diag(L)))
        quad_lml =  0.5*np.sum(v**2)
        const_lml = -0.5*n*np.log(2*np.pi)
        return const_lml - logdet_lml - quad_lml

    def optimize_hyperparameters(self, X, y, theta_init):
        # define optimization objective as the negative log marginal likelihood
        objective = lambda log_th: -self.log_marginal_likelihood(X, y, log_th)

        # optimize using gradients
        res = minimize(value_and_grad(objective), np.log(theta_init), jac=True)

        # check for success
        if not res.success:
            print('Warning: optimization failed!')

        # return results
        log_theta = res.x
        theta = np.exp(log_theta)
        self.sigma = theta[0]
        self.sigma2 = self.sigma*self.sigma
        self.theta_k = theta[1:]

        return theta

def KL_divergence(mu0, Sigma0, mu1, Sigma1):
    L0 = np.linalg.cholesky(Sigma0 + np.eye(len(Sigma0))*1e-8)
    L1 = np.linalg.cholesky(Sigma1 + np.eye(len(Sigma1))*1e-8)
    
    M = np.linalg.solve(L1, L0)

    v = np.linalg.solve(L1, mu1 - mu0)

    logdet = 2*np.sum(np.log(np.diag(L1)) - np.log(np.diag(L0)))
    return 0.5*(np.sum(M**2) + v.T@v - len(mu0) + logdet)

def squared_exponential(tau_squared, kappa, scale): 
    return kappa**2*np.exp(-0.5*tau_squared/scale**2)

def se_spectral_density_1D(w, theta_k):
    kappa, scale = theta_k[0], theta_k[1]
    return kappa**2*np.sqrt(2*np.pi)*scale*np.exp(-0.5*scale**2*w**2)

def se_spectral_density_multidim(tau, d, theta_k):
    kappa, scale = theta_k[0], theta_k[1]
    return (kappa**2*np.sqrt(2*np.pi)*scale)**d*np.exp(-0.5*scale**2*tau**2)

def matern_spectral_density_1D(w, theta_k):
    kappa, scale, nu = theta_k

    return 2*kappa**2*np.sqrt(np.pi)*(2*nu)**nu*gamma(nu + 0.5)/( scale**(2*nu)*gamma(nu) )*(2*nu/(scale**2) + w**2)**(-nu - 0.5)

def se_spectral_density_grad(w, theta_K):
    se_spectral_density = lambda w, theta_k: theta_k[0]**2*np.sqrt(2*np.pi)*theta_k[1]*np.exp(-0.5*theta_k[1]**2*w**2)
    kappa, scale = theta_K
    dSdkappa = 2/kappa * se_spectral_density(w, theta_K)
    dSdscale = (1/scale - scale*w**2)*se_spectral_density(w, theta_K)

    return np.array([dSdkappa, dSdscale])
