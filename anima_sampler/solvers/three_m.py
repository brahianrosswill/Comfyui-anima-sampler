"""One-eval RF 3M solver."""

from __future__ import annotations

import importlib
import math

from ..flow_math import _as_tensor_like, _rf_lambda, _rms, _scalar_float
from ..solver_types import Flow3MStepResult, FlowERState


def flow_3m_damped_step(
    x,
    denoised,
    t,
    t_next,
    *,
    state: FlowERState | None = None,
    max_gamma: float = 1.0,
    tolerance: float = 0.005,
    force_order_le_2: bool = False,
    eps: float = 1e-6,
) -> Flow3MStepResult:
    """One-eval lambda-native 2M/3M step with damped 3M extrapolation."""

    if state is None:
        state = FlowERState()

    torch = importlib.import_module("torch")
    current_lambda = _rf_lambda(torch, t, eps=eps)
    next_state = FlowERState(
        previous_denoised=denoised,
        previous_lambda=current_lambda,
        previous_previous_denoised=state.previous_denoised,
        previous_previous_lambda=state.previous_lambda,
    )

    if float(t_next) <= 0.0 or float(t) <= 0.0:
        zero = x.new_tensor(0.0)
        return Flow3MStepResult(denoised, next_state, denoised, denoised, zero, zero, 1, 0.0)

    t_current = torch.clamp(_as_tensor_like(torch, t, x), min=eps)
    t_next_tensor = torch.clamp(_as_tensor_like(torch, t_next, x), min=0.0)
    lambda_next = _rf_lambda(torch, t_next, eps=eps)
    h = lambda_next - current_lambda
    h_f = _scalar_float(torch, h)
    ratio = t_next_tensor / t_current
    if h_f <= eps:
        x_1m = ratio * x + (1.0 - ratio) * denoised
        zero = x.new_tensor(0.0)
        return Flow3MStepResult(x_1m, next_state, x_1m, x_1m, zero, zero, 1, 0.0)

    c = t_next_tensor * torch.exp(current_lambda)
    k0 = torch.expm1(h)
    k1 = torch.exp(h) * h - k0
    x_1m = ratio * x + c * k0 * denoised

    if state.previous_denoised is None or state.previous_lambda is None:
        zero = x.new_tensor(0.0)
        return Flow3MStepResult(x_1m, next_state, x_1m, x_1m, zero, zero, 1, 1.0)

    h_prev = current_lambda - state.previous_lambda
    h_prev_f = _scalar_float(torch, h_prev)
    if h_prev_f <= eps:
        zero = x.new_tensor(0.0)
        return Flow3MStepResult(x_1m, next_state, x_1m, x_1m, zero, zero, 1, 1.0)

    w0_2 = k0 + k1 / h_prev
    w1_2 = -k1 / h_prev
    x_2m = ratio * x + c * (w0_2 * denoised + w1_2 * state.previous_denoised)

    if (
        force_order_le_2
        or state.previous_previous_denoised is None
        or state.previous_previous_lambda is None
    ):
        zero = x.new_tensor(0.0)
        return Flow3MStepResult(x_2m, next_state, x_2m, x_2m, zero, zero, 2, 1.0)

    h_prevprev = state.previous_lambda - state.previous_previous_lambda
    h_prevprev_f = _scalar_float(torch, h_prevprev)
    if h_prevprev_f <= eps:
        zero = x.new_tensor(0.0)
        return Flow3MStepResult(x_2m, next_state, x_2m, x_2m, zero, zero, 2, 1.0)

    r1 = h_f / h_prev_f
    r2 = h_prev_f / h_prevprev_f
    k2 = torch.exp(h) * h * h - 2.0 * k1
    a = h_prev
    b = h_prev + h_prevprev
    w0_3 = k0 + ((a + b) / (a * b)) * k1 + k2 / (a * b)
    w1_3 = -(k2 + b * k1) / (a * (b - a))
    w2_3 = (k2 + a * k1) / (b * (b - a))
    coeff_l1 = _scalar_float(
        torch,
        (torch.abs(w0_3) + torch.abs(w1_3) + torch.abs(w2_3)) / (torch.abs(k0) + eps),
    )

    if not (0.50 <= r1 <= 2.00 and 0.50 <= r2 <= 2.00) or h_f > 0.55 or coeff_l1 > 5.0:
        zero = x.new_tensor(0.0)
        return Flow3MStepResult(x_2m, next_state, x_2m, x_2m, zero, zero, 2, coeff_l1)

    x_3m = ratio * x + c * (
        w0_3 * denoised
        + w1_3 * state.previous_denoised
        + w2_3 * state.previous_previous_denoised
    )
    e32 = _rms(torch, x_3m - x_2m) / (_rms(torch, x_2m) + eps)

    lambda_next_f = _scalar_float(torch, lambda_next)
    if lambda_next_f < -2.5:
        gamma_cap = 0.65
    elif lambda_next_f > 4.5:
        gamma_cap = 0.70
    else:
        gamma_cap = 0.85
    gamma_cap *= max(0.0, min(1.15, float(max_gamma)))
    gamma_cap = min(gamma_cap, 0.95)
    if not (0.65 <= r1 <= 1.60 and 0.65 <= r2 <= 1.60):
        gamma_cap *= 0.75

    sqrt_t = math.sqrt(max(0.0, min(1.0, _scalar_float(torch, t_next))))
    tau32 = max(0.0, float(tolerance)) * (0.50 + 0.80 * sqrt_t)
    if gamma_cap <= 0.0 or tau32 <= 0.0:
        zero = x.new_tensor(0.0)
        return Flow3MStepResult(x_2m, next_state, x_2m, x_3m, zero, e32, 2, coeff_l1)

    gamma3 = gamma_cap * min(1.0, math.sqrt(tau32 / (_scalar_float(torch, e32) + eps)))
    x_next = x_2m + gamma3 * (x_3m - x_2m)
    return Flow3MStepResult(x_next, next_state, x_2m, x_3m, x.new_tensor(gamma3), e32, 3, coeff_l1)
