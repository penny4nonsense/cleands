from ..utils import *
from ..base import clustering_model

class simple_k_means(clustering_model):
    def __init__(self, x, k, max_iters=100, seed=None):
        super(simple_k_means, self).__init__(x)
        self.n_clusters = k
        if seed != None: np.random.seed(seed)
        outp = np.random.randint(k, size=(x.shape[0],))
        inpt = np.zeros((x.shape[0],))
        means = self._calc_means(outp)
        for j in range(max_iters):
            if (inpt == outp).all(): break
            inpt = outp.copy()
            means_new = self._calc_means(inpt)
            means = self._replace_means(means, means_new)
            outp = simple_k_means._get_groups(x, means)
        self.iters = j
        self._means = means

    @staticmethod
    def _get_groups(x, means):
        outp = [x - means[i, :] for i in range(means.shape[0])]
        outp = [(item ** 2).sum(1) for item in outp]
        return np.array(outp).argmin(0)

    def _replace_means(self, means, means_new):
        for i in range(self.n_clusters):
            if not np.isnan(means_new[i, :]).all():
                means[i, :] = means_new[i, :]
        return means

    def cluster(self, newx):
        return simple_k_means._get_groups(newx, self._means)


class k_means(simple_k_means):
    def __init__(self, x, k, max_iters=100, seed=None, n_start=10):
        if seed != None: np.random.seed(seed)
        clustering_model.__init__(self, x)
        outp = {}
        for i in range(n_start):
            model = simple_k_means(x, k=k, max_iters=max_iters, seed=None)
            outp[model.total_within_group_sum_of_squares] = model
        model = outp[min(outp.keys())]
        self.iters = model.iters
        self.means = model.means
        self.n_clusters = k



def total_within_group_sum_of_squares_for_different_k(x: np.ndarray, k_max: int = 10, *args, **kwargs):
    outp = np.full(k_max + 1, np.nan)
    for k in range(1, k_max + 1):
        model = k_means(x, k, *args, **kwargs)
        outp[k] = model.total_within_group_sum_of_squares
    return outp


def select_k(x: np.ndarray, k_max: int = 10, *args, **kwargs):
    twss = total_within_group_sum_of_squares_for_different_k(x, k_max=k_max, *args, **kwargs)
    dwss = -np.diff(twss)
    dwss[0] = np.nansum(dwss) / np.log(k_max)
    ratio = dwss[:-1] / dwss[1:]
    return ratio.argmax() + 1