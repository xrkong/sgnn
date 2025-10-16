"""Single-scale Graph Neural Network implementation for physics simulations."""

from .graph_network import EncodeProcessDecode
from .learned_simulator import LearnedSimulator

__all__ = ['EncodeProcessDecode', 'LearnedSimulator']
