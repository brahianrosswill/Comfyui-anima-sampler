"""ComfyUI node registration for the Anima corrective sampler."""

from __future__ import annotations

from .corrective_sampler_node import AnimaFlowCorrectiveSampler
from .node_metadata import ANIMA_FLOW_SETTINGS, NODE_CATEGORY
from .settings_node import AnimaFlowSettings

NODE_CLASS_MAPPINGS = {
    "AnimaFlowSettings": AnimaFlowSettings,
    "AnimaFlowCorrectiveSampler": AnimaFlowCorrectiveSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaFlowSettings": "Anima Flow Settings",
    "AnimaFlowCorrectiveSampler": "Anima Flow Corrective Sampler",
}

__all__ = [
    "ANIMA_FLOW_SETTINGS",
    "NODE_CATEGORY",
    "AnimaFlowCorrectiveSampler",
    "AnimaFlowSettings",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
