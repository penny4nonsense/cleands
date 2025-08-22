# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import cleands as cds
from cleands import Prediction, Classification, Clustering, Distribution, DimensionReduction
from typing import Optional, Type, List, Tuple
from functools import partial
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum, auto

class DistType(Enum):
    MULTINOMIAL = auto()


@dataclass(eq=False, order=False)
class model_container(ABC):
    model_type: Type[cds.learning_model]
    n: int
    name: str
    statistic_name: str
    supervised: bool = False
    r: Optional[int] = None
    n_clusters: Optional[int] = None
    intercept: bool = False
    seed: Optional[int] = None
    test_ratio: float = 0.5
    tidy: bool = False
    dgp_type: Optional[DistType] = None
    model: cds.learning_model = field(init=False)
    statistic: float = field(init=False)

    def __post_init__(self) -> None:
        print(f'Creating {self.name}...')
        if self.seed != None: np.random.seed(self.seed)
        x, y = self.generate_data
        x_train, x_test, y_train, y_test = cds.test_train_split(x, y, test_ratio=self.test_ratio)
        if self.supervised:
            self.model = self.model_type(x_train, y_train)
        else:
            self.model = self.model_type(x_train)
        self.statistic = self.calculate_statistic(x_test, y_test)

    @property
    @abstractmethod
    def generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @abstractmethod
    def calculate_statistic(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        ...

    def print(self) -> None:
        print(self.name + ':')
        if self.tidy:
            print(self.model.tidy)
            print(self.model.glance.T)
        print(f'{self.statistic_name}={self.statistic}')

@dataclass(eq=False, order=False)
class prediction_container(model_container):
    statistic_name: str = 'rmse'
    supervised: bool = True

    @property
    def generate_data(self)->Tuple[np.ndarray, np.ndarray]:
        x = np.random.normal(size=(self.n, self.r - 1 if self.intercept else self.r))
        if self.intercept: x = cds.hstack(np.ones(self.n), x)
        y = x @ np.random.uniform(size=self.r) + np.random.normal(size=self.n)
        return x,y

    def calculate_statistic(self,x_test:np.ndarray,y_test:np.ndarray)->float:
        return self.model.out_of_sample_root_mean_squared_error(x_test, y_test)


@dataclass(eq=False, order=False)
class classification_container(model_container):
    statistic_name: str = 'accuracy'
    supervised: bool = True

    @property
    def generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        n_subsample = int(self.n / self.n_clusters)
        x = cds.vstack(*[
            np.random.normal(loc=np.random.uniform(low=-5, high=5, size=(self.r - 1 if self.intercept else self.r,)),
                             size=(n_subsample, self.r - 1 if self.intercept else self.r))
            for _ in range(self.n_clusters)])
        shuffle = np.random.permutation(x.shape[0])
        x = x[shuffle, :]
        y = np.digitize(shuffle, bins=range(n_subsample, self.n + 1, n_subsample))
        if self.intercept: x = cds.hstack(np.ones(self.n), x)
        return x,y

    def calculate_statistic(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        return self.model.out_of_sample_accuracy(x_test,y_test)


@dataclass(eq=False, order=False)
class clustering_container(model_container):
    statistic_name: str = 'accuracy'

    @property
    def generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        n_subsample = int(self.n / self.n_clusters)
        x = cds.vstack(
            *[np.random.normal(loc=np.random.uniform(low=-5, high=5, size=(self.r,)), size=(n_subsample, self.r)) for _
              in range(self.n_clusters)])
        shuffle = np.random.permutation(x.shape[0])
        x = x[shuffle, :]
        y = np.digitize(shuffle, bins=range(n_subsample, x.shape[0] + 1, n_subsample))
        return x,y

    def calculate_statistic(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        membership_ohe = cds.one_hot_encode(y_test)
        group_ohe = cds.one_hot_encode(self.model.cluster(x_test))
        return (membership_ohe.T @ group_ohe).max(0).sum() / x_test.shape[0]


@dataclass(eq=False, order=False)
class dimension_reduction_container(model_container):
    statistic_name: str = 'rmse'

    @property
    def generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.random.normal(size=(self.n, self.r))
        F = np.random.normal(size=(self.n, self.n_clusters))
        lam = np.random.normal(size=(self.r, self.n_clusters))
        x += F@lam.T
        return x, np.zeros(x.shape[0])

    def calculate_statistic(self, x_test: np.ndarray, _: np.ndarray) -> float:
        return self.model.out_of_sample_root_mean_squared_error(x_test)

@dataclass(eq=False, order=False)
class distribution_container(model_container):
    statistic_name: str = 'deviance'
    dgp_type: Optional[DistType] = None

    @property
    def generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        match self.dgp_type:
            case DistType.MULTINOMIAL:
                p = np.arange(1,self.n_clusters+1,dtype=float)
                p = p/p.sum()
                x = np.random.multinomial(self.n, p)
                x = [np.full(val,i) for i,val in enumerate(x)]
                x = cds.bind(*x)
            case _:
                raise ValueError(f'Invalid distribution type dgp_type={dgp_type}')
        return x.reshape(-1,1), np.zeros(x.shape[0])

    def calculate_statistic(self, x_test: np.ndarray, _: np.ndarray) -> float:
        return self.model.out_of_sample_deviance(x_test)

def prediction_models(n: int, r: int, seed: Optional[int] = None) -> List[model_container]:
    outp =[prediction_container(name='LS Regressor',
                           model_type=Prediction.least_squares_regressor,
                           tidy=True,
                           n = n,
                           r = r,
                           intercept = True,
                           seed = seed),
           prediction_container(name='White-Corrected Regressor',
                           model_type=partial(Prediction.least_squares_regressor, white=True),
                           tidy=True,
                           n=n,
                           r=r,
                           intercept=True,
                           seed=seed),
           prediction_container(name='L1 Regressor',
                           model_type=Prediction.l1_bootstrap_regressor,
                           tidy=True,
                           n=n,
                           r=r,
                           intercept=True,
                           seed=seed),
           prediction_container(name='RP Regressor',
                           model_type=Prediction.recursive_partitioning_regressor,
                           tidy=True,
                           n=n,
                           r=r,
                           intercept=True,
                           seed=seed),
           prediction_container(name='kNN Regressor',
                           model_type=Prediction.k_nearest_neighbors_cross_validation_regressor,
                           n=n,
                           r=r,
                           intercept=True,
                           seed=seed),
           prediction_container(name='Bagging Regressor',
                           model_type=partial(Prediction.bagging_recursive_partitioning_regressor,bootstraps=250),
                           n=n,
                           r=r,
                           intercept=True,
                           seed=seed),
           prediction_container(name='RF Regressor',
                           model_type=partial(Prediction.random_forest_regressor,bootstraps=250),
                           n=n,
                           r=r-1,
                           seed=seed)
           ]
    return outp

def classification_models(n: int, r: int, n_clusters: int, seed:Optional[int] = None) -> List[model_container]:
    outp = [classification_container(name='MN Classifier',
                            model_type=Classification.multinomial_classifier,
                            tidy=True,
                            n=n,
                            r=r,
                            n_clusters=n_clusters,
                            intercept=True,
                            seed=seed),
            classification_container(name='kNN Classifier',
                            model_type=Classification.k_nearest_neighbors_cross_validation_classifier,
                            n=n,
                            r=r,
                            n_clusters=n_clusters,
                            intercept=True,
                            seed=seed),
            classification_container(name='RP Classifier',
                            model_type=Classification.recursive_partitioning_classifier,
                            n=n,
                            r=r,
                            n_clusters=n_clusters,
                            intercept=True,
                            seed=seed),
            classification_container(name='LDA Classifier',
                            model_type=Classification.linear_discriminant_analysis,
                            n=n,
                            r=r-1,
                            n_clusters=n_clusters,
                            seed=seed),
            classification_container(name='QDA Classifier',
                            model_type=Classification.quadratic_discriminant_analysis,
                            n=n,
                            r=r-1,
                            n_clusters=n_clusters,
                            seed=seed),
            classification_container(name='Bagging Classifier',
                            model_type=Classification.bagging_recursive_partitioning_classifier,
                            n=n,
                            r=r,
                            n_clusters=n_clusters,
                            intercept=True,
                            seed=seed),
            classification_container(name='RF Classifier',
                            model_type=Classification.random_forest_classifier,
                            n=n,
                            r=r-1,
                            n_clusters=n_clusters,
                            intercept=False,
                            seed=seed),
            classification_container(name='Naive Bayes Classifier',
                             model_type=Classification.gaussian_naive_bayes,
                             n=n,
                             r=r-1,
                             n_clusters=n_clusters,
                             seed=seed)]
    return outp

def clustering_models(n: int, r: int, n_clusters: int, seed:Optional[int] = None) -> List[model_container]:
    outp = [clustering_container(name='k Means',
                                model_type=lambda x: Clustering.k_means(x, k = Clustering.select_k(x)),
                                n=n,
                                r=r,
                                n_clusters=n_clusters,
                                intercept=True,
                                seed=seed)]
    return outp

def distribution_models(n: int, n_clusters: int, seed:Optional[int] = None) -> List[model_container]:
    outp = [distribution_container(name='Multinomial',
                                model_type=Distribution.multinomial,
                                n=n,
                                n_clusters=n_clusters,
                                intercept=True,
                                seed=seed,
                                dgp_type=DistType.MULTINOMIAL)]
    return outp

def dimension_reduction_models(n: int, r:int, k: int, seed:Optional[int] = None) -> List[model_container]:
    outp = [dimension_reduction_container(name='PCA',
                                          model_type=partial(DimensionReduction.principal_components_analysis,k=k),
                                          n=n,
                                          r=r,
                                          n_clusters=k,
                                          seed=seed)]
    return outp

def main() -> None:
    # for model in prediction_models(3000, 5, seed = 90210):
    #     model.print()
    #     print('\n\n')
    # print('\n\n')
    # print('\n\n')
    # for model in classification_models(3000,5, n_clusters = 3, seed = 90210):
    #     model.print()
    #     print('\n\n')
    # print('\n\n')
    # print('\n\n')
    # for model in clustering_models(3000, 5, n_clusters=3, seed=90210):
    #     model.print()
    #     print(f'k={model.model.n_clusters}')
    #     print('\n\n')
    # print('\n\n')
    # print('\n\n')
    # for model in distribution_models(3000, n_clusters=4, seed=90210):
    #     model.print()
    #     print('\n\n')
    # print('\n\n')
    # print('\n\n')
    for model in dimension_reduction_models(100, 30, k=3, seed=90210):
        model.print()
        print('\n\n')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
