# merw_phate/kernels/__init__.py
from .registry import KERNEL_REGISTRY, register_kernel

# ── Import every kernel module so the @register_kernel decorators run ──
from . import gaussian      # triggers registration of "gaussian"
from . import alpha_decay   # triggers registration of "alpha_decay"
from . import cauchy        # triggers registration of "cauchy"

# Optional: expose the two functions directly if you want
#from .gaussian import build_affinity_fast
#from .alpha_decay import build_affinity_alpha_decay
#from .cauchy import build_affinity_cauchy

__all__ = [
    "KERNEL_REGISTRY",
    "register_kernel",
    "build_affinity_fast",
    "build_affinity_alpha_decay",
    "build_affinity_cauchy",
]