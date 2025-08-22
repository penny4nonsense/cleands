import numpy as np
from typing import Type, Dict
from ..base import distribution_model
from ..utils import *


## BOOTSTRAP THE FINITE SAMPLE DISTRIBUTION
def likelihood_ratio_test(null: distribution_model, alternative: distribution_model) -> Dict[str, float]:
    lrt_stat = null.log_likelihood - alternative.log_likelihood
    lrt_stat = np.exp(lrt_stat)
    p_value = sp.stats.chi2.cdf(lrt_stat, df=len(alternative.params) - len(null.params))
    return {'likelihood.ratio.statistic': lrt_stat, 'p.value': p_value}
