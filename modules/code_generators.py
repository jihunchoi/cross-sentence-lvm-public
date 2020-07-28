import torch
from torch import distributions, nn

from allennlp.modules.feedforward import FeedForward

from distributions.hyperspherical_uniform import HypersphericalUniform
from distributions.von_mises_fisher import VonMisesFisher


class GaussianCodeGenerator(nn.Module):

    def __init__(self, input_dim: int, code_dim: int):
        super().__init__()
        self._code_dim = code_dim

        self._mu_linear = FeedForward(
            input_dim=input_dim, num_layers=1, hidden_dims=code_dim,
            activations=lambda x: x)
        self._logvar_linear = FeedForward(
            input_dim=input_dim, num_layers=1, hidden_dims=code_dim,
            activations=lambda x: x)

    def get_output_dim(self):
        return self._code_dim

    def get_prior(self):
        device = next(self.parameters()).device
        prior = distributions.Normal(
            loc=torch.zeros(self._code_dim, device=device),
            scale=torch.ones(self._code_dim, device=device))
        return prior

    def sample_from_prior(self, batch_size):
        prior = self.get_prior()
        return prior.sample((batch_size,))

    def get_distribution(self, h):
        mu = self._mu_linear(h)
        logvar = self._logvar_linear(h)
        std = (0.5 * logvar).exp()
        dist = distributions.Normal(loc=mu, scale=std)
        return dist

    def forward(self, h):
        dist = self.get_distribution(h)
        if self.training:
            code = dist.rsample()
        else:
            code = dist.mean
        prior = self.get_prior()
        kld = distributions.kl_divergence(p=dist, q=prior)
        kld = kld.sum(1)
        return code, kld


class VmfCodeGenerator(nn.Module):

    def __init__(self, input_dim: int, code_dim: int, kappa: int):
        super().__init__()
        self._code_dim = code_dim
        self._kappa = kappa

        self._mu_linear = FeedForward(
            input_dim=input_dim, num_layers=1, hidden_dims=code_dim,
            activations=lambda x: x / x.norm(dim=-1, keepdim=True))

    def get_output_dim(self):
        return self._code_dim

    def get_prior(self):
        device = next(self.parameters()).device
        prior = HypersphericalUniform(self._code_dim - 1, device=device)
        return prior

    def sample_from_prior(self, batch_size):
        prior = self.get_prior()
        return prior.sample((batch_size,))

    def get_distribution(self, h):
        mu = self._mu_linear(h)
        kappa = torch.empty(mu.shape[:-1]).fill_(self._kappa).unsqueeze(-1)
        kappa = kappa.to(mu)
        dist = VonMisesFisher(loc=mu, scale=kappa)
        return dist

    def forward(self, h):
        dist = self.get_distribution(h)
        if self.training:
            code = dist.rsample()
        else:
            code = dist.mean
        prior = self.get_prior()
        kld = distributions.kl_divergence(p=dist, q=prior)
        return code, kld
