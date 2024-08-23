from enum import StrEnum

class SampledNetworkType(StrEnum):
    # Approximate SWIM, this is only approximate to the SWIM algorithm
    # because we are in an unsupervised setting
    # (we have no access to the true function values)
    A_SWIM = "A-SWIM"

    # this is SWIM with uniform data sampling, which does not require
    # true function values
    U_SWIM = "U-SWIM"

    # random-features sampled with a a-priori defined distributions
    ELM = "ELM"

    # This is "supervised" swim algorithm (A-SWIM tries to approximate this method!)
    SWIM = "SWIM"
