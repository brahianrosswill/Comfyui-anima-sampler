"""Shared sampler option names."""

FLOW_SOLVERS = [
    "flow_euler",
    "flow_ab2",
    "flow_heun",
    "flow_pc3_damped",
    "flow_pc3_diffusers_damped",
    "flow_3m_damped",
    "flow_unipc2_x0",
    "flow_unipc2_diffusers_x0",
    "flow_er",
]

FLOW_PC3_SOLVERS = frozenset(
    {
        "flow_pc3_damped",
        "flow_pc3_diffusers_damped",
    }
)

FLOW_UNIPC_SOLVERS = frozenset(
    {
        "flow_unipc2_x0",
        "flow_unipc2_diffusers_x0",
    }
)

FLOW_TWO_MODEL_CALL_SOLVERS = frozenset(
    {
        "flow_heun",
        *FLOW_PC3_SOLVERS,
    }
)

FLOW_SCHEDULES = [
    "flow_diffusers_linear_shift",
    "flow_cosmos",
    "flow_cosmos_rf_tail",
    "flow_cosmos_lambda_biased_strong",
    "flow_cosmos_rho7",
    "flow_rf_linear_shift",
    "flow_rf_linear_s_tail_shift5",
    "simple",
]
