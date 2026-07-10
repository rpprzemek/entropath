import time
import logging
from contextlib import contextmanager
import numpy as np
from scipy.sparse import issparse


class IndentFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, "indent"):
            record.indent = 0
        record.tabs = "  " * record.indent  # 2 spaces per level
        return super().format(record)
    def _log(self, msg, level="info", indent=0):
        if self.verbose <= 0:
            return
        if self.logger is None:
            print("  " * indent + msg)
        else:
            log_fn = getattr(self.logger, level)
            log_fn(msg, extra={"indent": indent})


@contextmanager
def timed_step(step_name: str, indent_level: int = 0, logger=None, verbose: int = 1):
    if verbose <= 0:
        yield
        return
    if logger is None:
        logger = logging.getLogger(__name__)

    start = time.perf_counter()

    logger.info(f"Calculating {step_name}...", extra={"indent": indent_level})

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(
            f"Calculated {step_name} in {elapsed:.2f} seconds.",
            extra={"indent": indent_level}
        )


def _to_dense(M):
    return M.toarray() if issparse(M) else M

def _matmul(A, B):
    return A.dot(B) if issparse(A) else A @ B

def _compute_power(op, k):
    if issparse(op):
        op_k = op.copy()
        for _ in range(k - 1):
            op_k = op_k @ op
        op_k = op_k.toarray()
    else:
        op_k = np.linalg.matrix_power(op, k)
    return op_k

def gaussian_kernel_knn(distances, indices, sigma):
    d2 = distances ** 2
    sig_i = sigma[:, None]
    sig_j = sigma[indices]
    return np.exp(-d2 / (sig_i * sig_j))


def alpha_normalize(K, alpha=0.5):
    d = K.sum(axis=1)
    d = np.maximum(d, 1e-12)

    d_alpha = d ** (-alpha)
    K_alpha = K * d_alpha[:, None] * d_alpha[None, :]

    return K_alpha
