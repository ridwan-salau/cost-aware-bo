from __future__ import annotations

import dataclasses

import time
import warnings
import numpy as np

import torch
from torch.optim import Optimizer
from torch import Tensor

from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.optim.optimize import _optimize_acqf, _optimize_acqf_all_features_fixed, _optimize_acqf_sequential_q
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.exceptions import InputDataError, UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.generation.gen import TGenCandidates, _process_scipy_result
from botorch.logging import logger
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
    TGenInitialConditions,
)
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.optim.utils import _filter_kwargs
from functools import partial
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, Type, Union

from botorch.generation.utils import _remove_fixed_features_from_optimization
from botorch.logging import _get_logger
from botorch.optim.parameter_constraints import (
    _arrayify,
    make_scipy_bounds,
    make_scipy_linear_constraints,
    make_scipy_nonlinear_inequality_constraints,
    NLC_TOL,
)
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.optim.utils import _filter_kwargs, columnwise_clamp, fix_features
from botorch.optim.utils.timeout import minimize_with_timeout
from scipy.optimize import OptimizeResult

INIT_OPTION_KEYS = {
    # set of options for initialization that we should
    # not pass to scipy.optimize.minimize to avoid
    # warnings
    "alpha",
    "batch_limit",
    "eta",
    "init_batch_limit",
    "nonnegative",
    "n_burnin",
    "sample_around_best",
    "sample_around_best_sigma",
    "sample_around_best_prob_perturb",
    "seed",
    "thinning",
}


@dataclasses.dataclass(frozen=True)
class OptimizeAcqfInputs:
    """
    Container for inputs to `optimize_acqf`.
    See docstring for `optimize_acqf` for explanation of parameters.
    """

    acq_function: AcquisitionFunction
    acq_type: str
    bounds: Tensor
    q: int
    delta: int
    curr_iter: int
    num_restarts: int
    raw_samples: Optional[int]
    options: Optional[Dict[str, Union[bool, float, int, str]]]
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]]
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]]
    nonlinear_inequality_constraints: Optional[List[Callable]]
    fixed_features: Optional[Dict[int, float]]
    post_processing_func: Optional[Callable[[Tensor], Tensor]]
    batch_initial_conditions: Optional[Tensor]
    return_best_only: bool
    gen_candidates: TGenCandidates
    sequential: bool
    ic_generator: Optional[TGenInitialConditions] = None
    timeout_sec: Optional[float] = None
    return_full_tree: bool = False
    retry_on_optimization_warning: bool = True
    ic_gen_kwargs: Dict = dataclasses.field(default_factory=dict)

    @property
    def full_tree(self) -> bool:
        return self.return_full_tree or (
            not isinstance(self.acq_function, OneShotAcquisitionFunction)
        )

    def __post_init__(self) -> None:
        if self.inequality_constraints is None and not (
            self.bounds.ndim == 2 and self.bounds.shape[0] == 2
        ):
            raise ValueError(
                "bounds should be a `2 x d` tensor, current shape: "
                f"{list(self.bounds.shape)}."
            )

        # TODO: Validate constraints if provided:
        # https://github.com/pytorch/botorch/pull/1231
        if self.batch_initial_conditions is not None and self.sequential:
            raise UnsupportedError(
                "`batch_initial_conditions` is not supported for sequential "
                "optimization. Either avoid specifying "
                "`batch_initial_conditions` to use the custom initializer or "
                "use the `ic_generator` kwarg to generate initial conditions "
                "for the case of nonlinear inequality constraints."
            )

        d = self.bounds.shape[1]
        if self.batch_initial_conditions is not None:
            batch_initial_conditions_shape = self.batch_initial_conditions.shape
            if len(batch_initial_conditions_shape) not in (2, 3):
                raise ValueError(
                    "batch_initial_conditions must be 2-dimensional or "
                    "3-dimensional. Its shape is "
                    f"{batch_initial_conditions_shape}."
                )
            if batch_initial_conditions_shape[-1] != d:
                raise ValueError(
                    f"batch_initial_conditions.shape[-1] must be {d}. The "
                    f"shape is {batch_initial_conditions_shape}."
                )

        elif self.ic_generator is None:
            if self.nonlinear_inequality_constraints is not None:
                raise RuntimeError(
                    "`ic_generator` must be given if "
                    "there are non-linear inequality constraints."
                )
            if self.raw_samples is None:
                raise ValueError(
                    "Must specify `raw_samples` when "
                    "`batch_initial_conditions` is None`."
                )

        if self.sequential and self.q > 1:
            if not self.return_best_only:
                raise NotImplementedError(
                    "`return_best_only=False` only supported for joint optimization."
                )
            if isinstance(self.acq_function, OneShotAcquisitionFunction):
                raise NotImplementedError(
                    "sequential optimization currently not supported for one-shot "
                    "acquisition functions. Must have `sequential=False`."
                )

    def get_ic_generator(self) -> TGenInitialConditions:
        if self.ic_generator is not None:
            return self.ic_generator
        elif isinstance(self.acq_function, qKnowledgeGradient):
            return gen_one_shot_kg_initial_conditions
        return gen_batch_initial_conditions

def optimize_acqf(
    acq_function: AcquisitionFunction,
    acq_type: str,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    delta: int = 0,
    curr_iter: int = 0,
    raw_samples: Optional[int] = None,
    options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    nonlinear_inequality_constraints: Optional[List[Callable]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    return_best_only: bool = True,
    gen_candidates: Optional[TGenCandidates] = None,
    sequential: bool = False,
    *,
    ic_generator: Optional[TGenInitialConditions] = None,
    timeout_sec: Optional[float] = None,
    return_full_tree: bool = False,
    retry_on_optimization_warning: bool = True,
    **ic_gen_kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    
    # using a default of None simplifies unit testing
    if gen_candidates is None:
        gen_candidates = gen_candidates_scipy
    opt_acqf_inputs = OptimizeAcqfInputs(
        acq_function=acq_function,
        acq_type=acq_type,
        bounds=bounds,
        q=q,
        delta=delta,
        curr_iter=curr_iter,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
        fixed_features=fixed_features,
        post_processing_func=post_processing_func,
        batch_initial_conditions=batch_initial_conditions,
        return_best_only=return_best_only,
        gen_candidates=gen_candidates,
        sequential=sequential,
        ic_generator=ic_generator,
        timeout_sec=timeout_sec,
        return_full_tree=return_full_tree,
        retry_on_optimization_warning=retry_on_optimization_warning,
        ic_gen_kwargs=ic_gen_kwargs,
    )
    return _optimize_acqf(opt_acqf_inputs)

def _optimize_acqf_batch(
    opt_inputs: OptimizeAcqfInputs, start_time: float, timeout_sec: Optional[float]
) -> Tuple[Tensor, Tensor]:
    options = opt_inputs.options or {}

    initial_conditions_provided = opt_inputs.batch_initial_conditions is not None

    if initial_conditions_provided:
        batch_initial_conditions = opt_inputs.batch_initial_conditions
    else:
        # pyre-ignore[28]: Unexpected keyword argument `acq_function` to anonymous call.
        batch_initial_conditions = opt_inputs.get_ic_generator()(
            acq_function=opt_inputs.acq_function,
            bounds=opt_inputs.bounds,
            q=opt_inputs.q,
            num_restarts=opt_inputs.num_restarts,
            raw_samples=opt_inputs.raw_samples,
            fixed_features=opt_inputs.fixed_features,
            options=options,
            inequality_constraints=opt_inputs.inequality_constraints,
            equality_constraints=opt_inputs.equality_constraints,
            **opt_inputs.ic_gen_kwargs,
        )

    batch_limit: int = options.get(
        "batch_limit",
        opt_inputs.num_restarts
        if not opt_inputs.nonlinear_inequality_constraints
        else 1,
    )
    has_parameter_constraints = (
        opt_inputs.inequality_constraints is not None
        or opt_inputs.equality_constraints is not None
        or opt_inputs.nonlinear_inequality_constraints is not None
    )

    def _optimize_batch_candidates(
        timeout_sec: Optional[float],
    ) -> Tuple[Tensor, Tensor, List[Warning]]:
        batch_candidates_list: List[Tensor] = []
        batch_acq_values_list: List[Tensor] = []
        batched_ics = batch_initial_conditions.split(batch_limit)
        opt_warnings = []
        if timeout_sec is not None:
            timeout_sec = (timeout_sec - start_time) / len(batched_ics)

        bounds = opt_inputs.bounds
        gen_kwargs: Dict[str, Any] = {
            "lower_bounds": None if bounds[0].isinf().all() else bounds[0],
            "upper_bounds": None if bounds[1].isinf().all() else bounds[1],
            "options": {k: v for k, v in options.items() if k not in INIT_OPTION_KEYS},
            "fixed_features": opt_inputs.fixed_features,
            "timeout_sec": timeout_sec,
        }

        if has_parameter_constraints:
            # only add parameter constraints to gen_kwargs if they are specified
            # to avoid unnecessary warnings in _filter_kwargs
            gen_kwargs.update(
                {
                    "inequality_constraints": opt_inputs.inequality_constraints,
                    "equality_constraints": opt_inputs.equality_constraints,
                    # the line is too long
                    "nonlinear_inequality_constraints": (
                        opt_inputs.nonlinear_inequality_constraints
                    ),
                }
            )
        filtered_gen_kwargs = _filter_kwargs(opt_inputs.gen_candidates, **gen_kwargs)

        for i, batched_ics_ in enumerate(batched_ics):
            # optimize using random restart optimization
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always", category=OptimizationWarning)
                (
                    batch_candidates_curr,
                    batch_acq_values_curr,
                ) = opt_inputs.gen_candidates(
                    batched_ics_, opt_inputs.acq_function, opt_inputs.acq_type,
                    opt_inputs.delta, opt_inputs.curr_iter, **filtered_gen_kwargs
                )
            opt_warnings += ws
            batch_candidates_list.append(batch_candidates_curr)
            batch_acq_values_list.append(batch_acq_values_curr)
            logger.info(f"Generated candidate batch {i+1} of {len(batched_ics)}.")

        batch_candidates = torch.cat(batch_candidates_list)
        has_scalars = batch_acq_values_list[0].ndim == 0
        if has_scalars:
            batch_acq_values = torch.stack(batch_acq_values_list)
        else:
            batch_acq_values = torch.cat(batch_acq_values_list).flatten()
        return batch_candidates, batch_acq_values, opt_warnings

    batch_candidates, batch_acq_values, ws = _optimize_batch_candidates(timeout_sec)

    optimization_warning_raised = any(
        (issubclass(w.category, OptimizationWarning) for w in ws)
    )
    if optimization_warning_raised and opt_inputs.retry_on_optimization_warning:
        first_warn_msg = (
            "Optimization failed in `gen_candidates_scipy` with the following "
            f"warning(s):\n{[w.message for w in ws]}\nBecause you specified "
            "`batch_initial_conditions`, optimization will not be retried with "
            "new initial conditions and will proceed with the current solution."
            " Suggested remediation: Try again with different "
            "`batch_initial_conditions`, or don't provide `batch_initial_conditions.`"
            if initial_conditions_provided
            else "Optimization failed in `gen_candidates_scipy` with the following "
            f"warning(s):\n{[w.message for w in ws]}\nTrying again with a new "
            "set of initial conditions."
        )
        warnings.warn(first_warn_msg, RuntimeWarning)

        if not initial_conditions_provided:
            batch_initial_conditions = opt_inputs.get_ic_generator()(
                acq_function=opt_inputs.acq_function,
                bounds=opt_inputs.bounds,
                q=opt_inputs.q,
                num_restarts=opt_inputs.num_restarts,
                raw_samples=opt_inputs.raw_samples,
                fixed_features=opt_inputs.fixed_features,
                options=options,
                inequality_constraints=opt_inputs.inequality_constraints,
                equality_constraints=opt_inputs.equality_constraints,
                **opt_inputs.ic_gen_kwargs,
            )

            batch_candidates, batch_acq_values, ws = _optimize_batch_candidates(
                timeout_sec
            )

            optimization_warning_raised = any(
                (issubclass(w.category, OptimizationWarning) for w in ws)
            )
            if optimization_warning_raised:
                warnings.warn(
                    "Optimization failed on the second try, after generating a "
                    "new set of initial conditions.",
                    RuntimeWarning,
                )

    if opt_inputs.post_processing_func is not None:
        batch_candidates = opt_inputs.post_processing_func(batch_candidates)

    if opt_inputs.return_best_only:
        best = torch.argmax(batch_acq_values.view(-1), dim=0)
        batch_candidates = batch_candidates[best]
        batch_acq_values = batch_acq_values[best]

    if not opt_inputs.full_tree:
        batch_candidates = opt_inputs.acq_function.extract_candidates(
            X_full=batch_candidates
        )

    return batch_candidates, batch_acq_values

def gen_candidates_scipy(
    initial_conditions: Tensor,
    acquisition_function: AcquisitionFunction,
    acq_type: str,
    delta: int = 0,
    curr_iter: int = 0,
    lower_bounds: Optional[Union[float, Tensor]] = None,
    upper_bounds: Optional[Union[float, Tensor]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    nonlinear_inequality_constraints: Optional[List[Callable]] = None,
    options: Optional[Dict[str, Any]] = None,
    fixed_features: Optional[Dict[int, Optional[float]]] = None,
    timeout_sec: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    
    options = options or {}
    options = {**options, "maxiter": options.get("maxiter", 2000)}

    # if there are fixed features we may optimize over a domain of lower dimension
    reduced_domain = False
    if fixed_features:
        # TODO: We can support fixed features, see Max's comment on D33551393. We can
        # consider adding this at a later point.
        if nonlinear_inequality_constraints:
            raise NotImplementedError(
                "Fixed features are not supported when non-linear inequality "
                "constraints are given."
            )
        # if there are no constraints things are straightforward
        if not (inequality_constraints or equality_constraints):
            reduced_domain = True
        # if there are we need to make sure features are fixed to specific values
        else:
            reduced_domain = None not in fixed_features.values()

    if reduced_domain:
        _no_fixed_features = _remove_fixed_features_from_optimization(
            fixed_features=fixed_features,
            acquisition_function=acquisition_function,
            acq_type=acq_type,
            delta=delta,
            curr_iter=curr_iter,
            initial_conditions=initial_conditions,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )
        # call the routine with no fixed_features
        clamped_candidates, batch_acquisition = gen_candidates_scipy(
            initial_conditions=_no_fixed_features.initial_conditions,
            acquisition_function=_no_fixed_features.acquisition_function,
            acq_type=_no_fixed_features.acq_type,
            delta=_no_fixed_features.delta,
            curr_iter=_no_fixed_features.curr_iter,
            lower_bounds=_no_fixed_features.lower_bounds,
            upper_bounds=_no_fixed_features.upper_bounds,
            inequality_constraints=_no_fixed_features.inequality_constraints,
            equality_constraints=_no_fixed_features.equality_constraints,
            options=options,
            fixed_features=None,
            timeout_sec=timeout_sec,
        )
        clamped_candidates = _no_fixed_features.acquisition_function._construct_X_full(
            clamped_candidates
        )
        return clamped_candidates, batch_acquisition
    clamped_candidates = columnwise_clamp(
        X=initial_conditions, lower=lower_bounds, upper=upper_bounds
    )

    shapeX = clamped_candidates.shape
    x0 = clamped_candidates.view(-1)
    bounds = make_scipy_bounds(
        X=initial_conditions, lower_bounds=lower_bounds, upper_bounds=upper_bounds
    )
    constraints = make_scipy_linear_constraints(
        shapeX=shapeX,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
    )

    with_grad = options.get("with_grad", True)
    if with_grad:

        def f_np_wrapper(x: np.ndarray, f: Callable):
            """Given a torch callable, compute value + grad given a numpy array."""
            if np.isnan(x).any():
                raise RuntimeError(
                    f"{np.isnan(x).sum()} elements of the {x.size} element array "
                    f"`x` are NaN."
                )
            X = (
                torch.from_numpy(x)
                .to(initial_conditions)
                .view(shapeX)
                .contiguous()
                .requires_grad_(True)
            )
            X_fix = fix_features(X, fixed_features=fixed_features)
            loss = f(X_fix).sum()
            # compute gradient w.r.t. the inputs (does not accumulate in leaves)
            gradf = _arrayify(torch.autograd.grad(loss, X)[0].contiguous().view(-1))
            if np.isnan(gradf).any():
                msg = (
                    f"{np.isnan(gradf).sum()} elements of the {x.size} element "
                    "gradient array `gradf` are NaN. "
                    "This often indicates numerical issues."
                )
                if initial_conditions.dtype != torch.double:
                    msg += " Consider using `dtype=torch.double`."
                raise RuntimeError(msg)
            fval = loss.item()
            return fval, gradf

    else:

        def f_np_wrapper(x: np.ndarray, f: Callable):
            X = torch.from_numpy(x).to(initial_conditions).view(shapeX).contiguous()
            with torch.no_grad():
                X_fix = fix_features(X=X, fixed_features=fixed_features)
                loss = f(X_fix).sum()
            fval = loss.item()
            return fval

    if nonlinear_inequality_constraints:
        # Make sure `batch_limit` is 1 for now.
        if not (len(shapeX) == 3 and shapeX[:2] == torch.Size([1, 1])):
            raise ValueError(
                "`batch_limit` must be 1 when non-linear inequality constraints "
                "are given."
            )
        constraints += make_scipy_nonlinear_inequality_constraints(
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            f_np_wrapper=f_np_wrapper,
            x0=x0,
        )
    x0 = _arrayify(x0)

    def f(x):
        return -acquisition_function(x) if ('EIPU' not in acq_type) else -acquisition_function(x, delta, curr_iter)

    res = minimize_with_timeout(
        fun=f_np_wrapper,
        args=(f,),
        x0=x0,
        method=options.get("method", "SLSQP" if constraints else "L-BFGS-B"),
        jac=with_grad,
        bounds=bounds,
        constraints=constraints,
        callback=options.get("callback", None),
        options={
            k: v
            for k, v in options.items()
            if k not in ["method", "callback", "with_grad"]
        },
        timeout_sec=timeout_sec,
    )
    _process_scipy_result(res=res, options=options)

    candidates = fix_features(
        X=torch.from_numpy(res.x).to(initial_conditions).reshape(shapeX),
        fixed_features=fixed_features,
    )

    # SLSQP sometimes fails in the line search or may just fail to find a feasible
    # candidate in which case we just return the starting point. This happens rarely,
    # so it shouldn't be an issue given enough restarts.
    if nonlinear_inequality_constraints and any(
        nlc(candidates.view(-1)) < NLC_TOL for nlc in nonlinear_inequality_constraints
    ):
        candidates = torch.from_numpy(x0).to(candidates).reshape(shapeX)
        warnings.warn(
            "SLSQP failed to converge to a solution the satisfies the non-linear "
            "constraints. Returning the feasible starting point."
        )

    clamped_candidates = columnwise_clamp(
        X=candidates, lower=lower_bounds, upper=upper_bounds, raise_on_violation=True
    )
    with torch.no_grad():
        batch_acquisition = acquisition_function(clamped_candidates) if ('EIPU' not in acq_type) else acquisition_function(clamped_candidates, delta, curr_iter)
    
    return clamped_candidates, batch_acquisition
