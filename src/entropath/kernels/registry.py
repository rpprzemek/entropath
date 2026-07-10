# kernels/registry.py
KERNEL_REGISTRY = {}

def register_kernel(name: str):
    def decorator(func):
        KERNEL_REGISTRY[name] = func
        return func
    return decorator