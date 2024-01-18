from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.sampling import MCSampler
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform
from typing import Union, Optional, Dict, Any
from torch.distributions import Normal
from torch import Tensor
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EIPS(AnalyticAcquisitionFunction):
    r"""Modification of Standard Expected Improvement Class defined in BoTorch
    See: https://botorch.org/api/_modules/botorch/acquisition/analytic.html#ExpectedImprovement
    """

    def __init__(
        self,
        model: Model,
        cost_gp: Model,
        best_f: Union[float, Tensor],
        cost_sampler: Optional[MCSampler] = None,
        acq_objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        acq_type: str = "",
        unstandardizer=None,
        normalizer=None,
        unnormalizer=None,
        bounds: Tensor = None,
        iter: int = None,
        params: Dict = None,
        eta=None,
        consumed_budget=None,
        warmup_iters=None,
        **kwargs: Any,
    ) -> None:
        r"""q-Expected Improvement.

        Args:
            model: A fitted objective model.
            cost_model: A fitted cost model.
            best_f: The best objective value observed so far (assumed noiseless).
            cost_sampler: The sampler used to draw base samples.
            acq_objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f, device=DEVICE)

        self.register_buffer("best_f", best_f)
        self.cost_gp = cost_gp
        self.cost_sampler = cost_sampler
        self.acq_obj = acq_objective
        self.acq_type = acq_type
        self.unstandardizer = unstandardizer
        self.normalizer = normalizer
        self.unnormalizer = unnormalizer
        self.bounds = bounds
        self.params = params
        self.iter = iter
        self.eta = eta
        self.consumed_budget = consumed_budget
        self.warmup = True

    def get_mc_samples(self, X: Tensor, gp_model, bounds):
        cost_posterior = gp_model.posterior(X)
        cost_samples = self.cost_sampler(cost_posterior)
        cost_samples = cost_samples.to(DEVICE)
        cost_samples = cost_samples.max(dim=2)[0]

        cost_samples = self.unstandardizer(cost_samples, bounds=bounds)
        cost_samples = torch.exp(cost_samples)

        cost_samples = self.acq_obj(cost_samples)

        cost_samples = cost_samples[:, :, None]
        cost_samples = cost_samples.to(DEVICE)

        return cost_samples

    def get_stagewise_expected_costs(self, X):
        return self.get_mc_samples(X, self.cost_gp[0], self.bounds["c"][:, 0])

    def compute_expected_inverse_cost(self, X: Tensor) -> Tensor:
        stage_costs = self.get_stagewise_expected_costs(X)

        stage_costs = stage_costs.sum(dim=-1)

        inv_cost = 1 / stage_costs
        inv_cost = inv_cost.mean(dim=0)

        return inv_cost

    def compute_expected_cost(self, X: Tensor) -> Tensor:
        all_cost_obj = []
        cost_model = self.cost_gp[0]

        cost_posterior = cost_model.posterior(X)

        cost_samples = self.cost_sampler(cost_posterior)
        cost_samples = cost_samples.to(DEVICE)
        cost_samples = cost_samples.max(dim=2)[0]

        # cost_samples = self.unstandardizer(cost_samples, bounds=self.bounds['c'][:,0])
        # cost_samples = torch.exp(cost_samples)
        cost_obj = self.acq_obj(cost_samples)
        all_cost_obj.append(cost_obj.mean(dim=0).item())
        return all_cost_obj

    def custom_EI(self, X):
        self.best_f = self.best_f.to(X)
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )

        mean = posterior.mean
        view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
        mean = mean.view(view_shape)
        sigma = posterior.variance.clamp_min(1e-9).sqrt().view(view_shape)

        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u

        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))

        ei = sigma * (updf + u * ucdf)

        return ei

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor, delta: int = 0, curr_iter: int = -1) -> Tensor:
        ei_x = self.custom_EI(X)

        inv_cost = self.compute_expected_inverse_cost(X)

        return ei_x * inv_cost
