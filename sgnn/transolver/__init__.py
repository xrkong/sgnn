"""Multi-scale Graph Neural Network implementation for physics simulations."""

from .multi_scale_gnn import MultiScaleGNN
from .multi_scale_graph import MultiScaleGraph, MultiScaleConfig
from .multi_scale_simulator import MultiScaleSimulator

__all__ = ['MultiScaleGNN', 'MultiScaleGraph', 'MultiScaleConfig', 'MultiScaleSimulator']