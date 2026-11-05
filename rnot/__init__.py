"""R-NOT: Riemannian optimal transport generator on the hypersphere, exactly two network calls per sample."""
from .geometry import angle, exp_step, tangent
from .model import RNOT, FourierField
from .transport import cells, fit_potentials

__version__ = '0.1.0'
__all__ = ['RNOT', 'FourierField', 'exp_step', 'tangent', 'angle', 'cells', 'fit_potentials']
