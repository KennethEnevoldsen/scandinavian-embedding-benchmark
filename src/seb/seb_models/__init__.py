try:
    from .fairseq_models import *
except ImportError:
    import warnings
    warnings.warn("Could not import fairseq_models. Make sure you have" +
                  "fairseq2 installed. This is currently only supported for " +
                  "Linux.")
from .hf_models import *