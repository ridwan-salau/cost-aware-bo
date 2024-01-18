from botorch.acquisition.objective import  MCAcquisitionObjective, PosteriorTransform
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition import ExpectedImprovement
from botorch.sampling import MCSampler
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform
from typing import Union, Optional, Dict, Any
from torch.distributions import Normal
from torch import Tensor
import torch
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EEIPU(AnalyticAcquisitionFunction):
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
        unstandardizer = None,
        normalizer = None,
        unnormalizer = None,
        bounds: Tensor = None,
        iter: int =None,
        params: Dict = None,
        eta = None,
        consumed_budget = None,
        warmup_iters = None,
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
        super().__init__(
            model=model,
            posterior_transform=posterior_transform,
            **kwargs
        )
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

        cost_samples = cost_samples[:,:,None]
        cost_samples = cost_samples.to(DEVICE)
        
        return cost_samples

    def get_memoized_costs(self, X, delta):
        
        stage_costs = None
        for i in range(delta):
            cost_samples = torch.full((self.params['cost_samples'], X.shape[0]), self.params['epsilon'], device=DEVICE)
            cost_samples = cost_samples[:,:,None]
            cost_samples = cost_samples.to(DEVICE)
            
            stage_costs = cost_samples if (not torch.is_tensor(stage_costs)) else torch.cat([stage_costs, cost_samples], axis=2)
            
        return stage_costs

    def get_stagewise_expected_costs(self, X, delta):

        # Discount Memoized stages by setting their costs to epsilon
        stage_costs = self.get_memoized_costs(X, delta)
        
        # Use MC Sampling to get the expected costs of unmemoized stages
        for i in range(delta, len(self.cost_gp)):
            cost_model = self.cost_gp[i]
            hyp_indexes = self.params['h_ind'][i]
            
            cost_samples = self.get_mc_samples(X[:,:,hyp_indexes], cost_model, self.bounds['c'][:,i])
            stage_costs = cost_samples if (not torch.is_tensor(stage_costs)) else torch.cat([stage_costs, cost_samples], axis=2)

        return stage_costs
        
    def compute_expected_inverse_cost(self, X: Tensor, delta: int = 0, alpha_epsilon=False) -> Tensor:
        
        stage_costs = self.get_stagewise_expected_costs(X, delta)
        
        stage_costs = stage_costs.sum(dim=-1)
        
        inv_cost = 1/stage_costs
        inv_cost = inv_cost.mean(dim=0)
        
        return inv_cost
        
    def compute_taylor_expansion(self, X: Tensor, delta: int = 0, alpha_epsilon=False) -> Tensor:
        
        stage_costs = self.get_stagewise_expected_costs(X, delta)
        
        stage_costs = stage_costs.sum(dim=-1)
        
        sample_mean = stage_costs.mean(dim=0)
        sample_var = stage_costs.var(dim=0)

        inv_cost = 1/sample_mean + sample_var/sample_mean**3
        return inv_cost

    # def compute_ground_truth(self, X: Tensor, alpha_epsilon=False) -> Tensor:
    #     stage_costs = self.get_mc_samples(X, self.inv_cost_gp, self.bounds['1/c'])
        
    #     ground_truth = stage_costs.mean(dim=0)
    #     return ground_truth

    def compute_expected_cost(self, X: Tensor) -> Tensor:

        # Check on GPT if this is the proper sampling method
        # Also, test the accuracy of cost expectation
        all_cost_obj = []
        for i, cost_model in enumerate(self.cost_gp):
            hyp_indexes = self.params['h_ind'][i]
            cost_posterior = cost_model.posterior(X[:,hyp_indexes])
            cost_samples = self.cost_sampler(cost_posterior)
            cost_samples = cost_samples.to(DEVICE)
            cost_samples = cost_samples.max(dim=2)[0]
            
            # cost_samples = self.unstandardizer(cost_samples, bounds=self.bounds['c'][:,i])
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

        total_budget = self.params['total_budget'] + 0

        remaining = total_budget - self.consumed_budget
        init_budget = total_budget - self.params['budget_0']

        cost_cool = remaining / init_budget
     
        inv_cost =  self.compute_expected_inverse_cost(X, delta=delta)

        return ei_x * (inv_cost**cost_cool)