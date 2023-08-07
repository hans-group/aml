from aml.models.iap import InterAtomicPotential


def load_iap(path: str) -> InterAtomicPotential:
    return InterAtomicPotential.load(path)
