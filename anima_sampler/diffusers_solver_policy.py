"""Conservative solver policy for Diffusers FlowMatch timestep grids."""

from __future__ import annotations

from dataclasses import dataclass

from .flow_math import _rf_lambda, _scalar_float

DIFFUSERS_GRID_TAIL_SIGMA = 0.10
DIFFUSERS_GRID_TAIL_GAP = 0.65
DIFFUSERS_GRID_TAIL_GAP_SIGMA = 0.25
DIFFUSERS_GRID_UNIPC_BODY_ORDER = 2


@dataclass(frozen=True)
class DiffusersGridStepPolicy:
    t: float
    t_next: float
    lambda_gap: float
    terminal: bool
    tail_interval: bool
    tail_corrector: bool


def diffusers_grid_step_policy(
    torch,
    t,
    t_next,
) -> DiffusersGridStepPolicy:
    """Classify RF step geometry that is risky on Diffusers-style grids."""

    t_value = _scalar_float(torch, t)
    t_next_value = _scalar_float(torch, t_next)
    lambda_gap = _scalar_float(torch, _rf_lambda(torch, t_next) - _rf_lambda(torch, t))
    terminal = t_next_value <= 0.0
    low_sigma_tail = t_next_value <= DIFFUSERS_GRID_TAIL_SIGMA
    large_tail_gap = (
        t_value <= DIFFUSERS_GRID_TAIL_GAP_SIGMA
        and lambda_gap >= DIFFUSERS_GRID_TAIL_GAP
    )
    tail_interval = terminal or low_sigma_tail or large_tail_gap
    tail_corrector = terminal or t_value <= DIFFUSERS_GRID_TAIL_SIGMA
    return DiffusersGridStepPolicy(
        t=t_value,
        t_next=t_next_value,
        lambda_gap=lambda_gap,
        terminal=terminal,
        tail_interval=tail_interval,
        tail_corrector=tail_corrector,
    )


def diffusers_grid_unipc_solver_order(
    requested_order: int,
    policy: DiffusersGridStepPolicy,
) -> int:
    """Cap UniPC order to the official order-2 body and first-order tail."""

    if policy.tail_interval:
        return 1
    return max(1, min(DIFFUSERS_GRID_UNIPC_BODY_ORDER, int(requested_order)))


def diffusers_grid_unipc_disable_correctors(
    *,
    step_index: int,
    disable_corrector_first: int,
    policy: DiffusersGridStepPolicy,
) -> tuple[int, ...]:
    disabled = set(range(max(0, int(disable_corrector_first))))
    if policy.tail_corrector and int(step_index) > 0:
        disabled.add(int(step_index) - 1)
    return tuple(sorted(disabled))


def diffusers_grid_pc3_predictor_max_order(
    max_order: int,
    policy: DiffusersGridStepPolicy,
) -> int:
    if policy.tail_interval:
        return 1
    return max(1, min(3, int(max_order)))


def diffusers_grid_pc3_skip_note(policy: DiffusersGridStepPolicy) -> str:
    if policy.terminal:
        return "pc3_terminal"
    if policy.tail_interval:
        return "pc3_diffusers_tail"
    return "pc3_endpoint_skipped"
