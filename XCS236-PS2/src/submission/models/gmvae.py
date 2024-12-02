import numpy as np
import torch
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

from torch import nn

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim) # from ./nns/v1.py
        self.dec = nn.Decoder(self.z_dim) # from ./nns/v1.py

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   nelbo, kl_z, rec
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        m_mixture, v_mixture = ut.gaussian_parameters(self.z_pre, dim=1)

        x = x.view(x.size(0), -1)
        z_mean, z_var = self.enc(x)  # Assuming encoder outputs mean and variance directly
        z = ut.sample_gaussian(z_mean, z_var)
        logits = self.dec(z)

        recon_loss = -ut.log_bernoulli_with_logits(x, logits).mean()
        
        # Compute KL divergence
        log_qz = ut.log_normal(z, z_mean, z_var)
        log_pz = ut.log_normal_mixture(z, m_mixture, v_mixture)
        kl_div = (log_qz - log_pz).mean()

        nelbo = recon_loss + kl_div
        return nelbo, kl_div, recon_loss

        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be tensor scalars
        #
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        m_mixture, v_mixture = prior

        ### START CODE HERE ###
        # Duplicate the input to sample multiple z values
        x_repeated = ut.duplicate(x, iw)
        
        # Get encoder outputs for the duplicated x
        z_mean, z_var = self.enc(x_repeated)  # Assuming encoder outputs mean and variance directly

        # Sample z from the Gaussian distribution using the reparameterization trick
        z = ut.sample_gaussian(z_mean, z_var)

        # Get decoder outputs
        logits = self.dec(z)

        # Compute the reconstruction loss
        recon_loss = -ut.log_bernoulli_with_logits(x_repeated, logits).view(iw, -1).mean(0)

        # Compute log probabilities for q(z|x) and p(z)
        log_qz = ut.log_normal(z, z_mean, z_var).view(iw, -1)
        log_pz = ut.log_normal_mixture(z, m_mixture, v_mixture).view(iw, -1)

        # Compute the log importance weights
        log_iw = (log_pz + ut.log_bernoulli_with_logits(x_repeated, logits).view(iw, -1) - log_qz).view(iw, -1)

        # Compute the negative IWAE bound
        niwae = -ut.log_mean_exp(log_iw, dim=0).mean()

        # For reporting purposes, compute KL and reconstruction terms
        kl = (log_qz - log_pz).view(iw, -1).mean(0).mean()
        rec = recon_loss.mean()

        return niwae, kl, rec

        # Return:
        #   niwae, kl, rec
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
