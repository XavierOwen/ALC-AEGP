
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[6,], activation=nn.LogSigmoid, activation_kwargs=None):
        super().__init__()
        if activation_kwargs is None:
            activation_kwargs = {}
        act = lambda: activation(**activation_kwargs)

        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.BatchNorm1d(h, affine=False))
            encoder_layers.append(act())
            prev_dim = h
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(nn.BatchNorm1d(latent_dim, affine=False))
        encoder_layers.append(act())
        self.encoder_net = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(act())
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder_net = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder_net(x)

    def decode(self, z):
        return self.decoder_net(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def ae_loss(self, x, lambda_l2_z=0.001, lambda_l2_param=0.001):
        x_hat, z = self.forward(x)
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        latent_l2 = torch.mean(z.pow(2)) if lambda_l2_z > 0 else torch.tensor(0., device=x.device)
        param_l2 = torch.tensor(0., device=x.device)
        if lambda_l2_param > 0:
            for param in self.parameters():
                if param.requires_grad:
                    param_l2 += torch.sum(param.pow(2))
        total_loss = recon_loss + lambda_l2_z * latent_l2 + lambda_l2_param * param_l2
        return total_loss, recon_loss, latent_l2, param_l2

class ParametricGP(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.log_lengthscale = nn.Parameter(torch.zeros(latent_dim))  # θ_l
        self.log_eta = nn.Parameter(torch.tensor(-1.0))                # log η
        self._last_tau2_hat = 1.0

    def kernel(self, x1, x2):
        """
        Gaussian RBF kernel with ARD (SE-ARD), scaled by tau².
        """
        lengthscale = torch.exp(self.log_lengthscale)
        x1_ = x1 / lengthscale
        x2_ = x2 / lengthscale
        dist_sq = torch.cdist(x1_, x2_, p=2).pow(2)
        return torch.exp(-0.5 * dist_sq)  # base kernel k(x,x')

    def compute_K_eta(self, x):
        """
        Return K + ηI (without τ²), to be scaled externally.
        """
        K = self.kernel(x, x)
        eta = torch.exp(self.log_eta)
        N = x.size(0)
        return K + eta * torch.eye(N, device=x.device)

    def predict(self, x_train, y_train, x_test):
        """
        GP posterior mean and variance following Eqn (2.2) from your paper.
        """
        K_eta = self.compute_K_eta(x_train)
        k_star = self.kernel(x_train, x_test)
        k_ss_diag = torch.ones(x_test.size(0), device=x_test.device)  # since k(x,x)=1

        L = torch.linalg.cholesky(K_eta + 1e-8 * torch.eye(K_eta.shape[0], device=K_eta.device))
        alpha = torch.cholesky_solve(y_train, L)

        # Estimate τ² from data
        N = y_train.shape[0]
        tau2_hat = ((y_train.T @ alpha) / N).squeeze().detach()
        self._last_tau2_hat = tau2_hat.item()

        mean = k_star.T @ alpha                                 # (M, 1)
        v = torch.cholesky_solve(k_star, L)
        k_star_Kinv_k_star = (k_star.T @ v).diagonal().unsqueeze(-1)
        eta = torch.exp(self.log_eta)

        var = tau2_hat * (1 + eta - k_star_Kinv_k_star)         # (M, 1)
        return mean, var

    def log_marginal_likelihood(self, x_train, y_train):
        """
        Profiled log-likelihood over η, using τ² = yᵀ K⁻¹ y / N
        """
        K_eta = self.compute_K_eta(x_train)
        L = torch.linalg.cholesky(K_eta + 1e-8 * torch.eye(K_eta.shape[0], device=K_eta.device))
        alpha = torch.cholesky_solve(y_train, L)

        N = y_train.shape[0]
        tau2_hat = ((y_train.T @ alpha) / N).squeeze().detach()
        self._last_tau2_hat = tau2_hat.item()

        term1 = y_train.T @ alpha
        term2 = torch.logdet(K_eta)
        return -(term1 + term2).squeeze()

    def log_marginal_likelihood(self, x_train, y_train):
        """
        Log marginal likelihood, with τ² = hat τ²
        """
        N = y_train.shape[0]
        K_eta = self.compute_K_eta(x_train)            # (N, N)
        L = torch.linalg.cholesky(K_eta + 1e-8 * torch.eye(N, device=K_eta.device))
        alpha = torch.cholesky_solve(y_train, L)           # (N, 1)

        tau2_hat = (y_train.T @ alpha) / N                 # scalar
        term1 = N * tau2_hat.log()
        term2 = 2 * L.diag().log().sum()
        # const: term3 = N * math.log(2 * math.pi)

        nll = term1 + term2 # half positive log likelihood
        return -nll.squeeze()

    def get_hyperparams(self):
        return {
            "lengthscale": torch.exp(self.log_lengthscale).detach(),
            "eta": torch.exp(self.log_eta).item(),
            "tau²_hat": self._last_tau2_hat
        }

class JointModel(nn.Module):
    def __init__(self, ae, gp):
        """
        ae: Autoencoder model (with .encode(x): (N, d))
        gp: ParametricGP model (with .predict and .log_marginal_likelihood)
        """
        super().__init__()
        self.ae = ae
        self.gp = gp

    def forward(self, x_train, y_train, x_test):
        """
        Run AE + GP prediction pipeline

        Args:
            x_train: (N, D) input data
            y_train: (N, 1) target values
            x_test:  (M, D) test input data

        Returns:
            y_pred: mean predictions (M, 1)
            y_var:  predictive variances (M, 1)
        """
        z_train = self.ae.encode(x_train)  # (N, d)
        z_test = self.ae.encode(x_test)    # (M, d)
        return self.gp.predict(z_train, y_train, z_test)

    def loss(self, x, y):
        """
        Loss function: AE reconstruction + GP negative log marginal likelihood

        Args:
            x: (N, D)
            y: (N, 1)

        Returns:
            total_loss:  scalar
            ae_loss:     scalar
            gp_nll:      scalar
        """
        # Autoencoder loss
        ae_total, ae_recon, _, _ = self.ae.ae_loss(x)

        # GP negative log marginal likelihood
        z = self.ae.encode(x)
        gp_nll = -self.gp.log_marginal_likelihood(z, y)  # (maximize log-likelihood = minimize -ll)

        # Combined loss
        total_loss = ae_total + gp_nll
        return total_loss, ae_total, gp_nll