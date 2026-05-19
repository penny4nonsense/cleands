"""
Ensemble classification models.

This module provides ensemble classifiers for improved predictive accuracy
and stability, using bootstrap aggregation (bagging) and randomization
strategies.

Classes:
    bagging_logistic_classifier:
        Ensemble of logistic classifiers fit on bootstrap samples.
    bagging_recursive_partitioning_classifier:
        Ensemble of recursive partitioning classifiers (decision trees).

Factory Aliases:
    random_forest_classifier:
        Partially-applied constructor for randomized recursive partitioning
        (random forest-style) classifier.
    BaggingLogisticClassifier:
        Wrapper for constructing a bagged logistic classifier via ClassificationModel.
    RandomForestClassifier:
        Wrapper for constructing a random forest classifier via ClassificationModel.
    BaggingRecursivePartitioningClassifier:
        Wrapper for constructing a bagged tree classifier via ClassificationModel.
"""

from ..base import *
from ..utils import *
from .glm import logistic_classifier
from .tree import recursive_partitioning_classifier
from functools import partial


class bagging_logistic_classifier(logistic_classifier):
    """Bagged ensemble of logistic classifiers.

    Trains multiple logistic classifiers on bootstrap resamples of the data,
    then aggregates their predictions for improved robustness and variance
    reduction.

    Attributes:
        seed (int, optional): Random seed for reproducibility.
        n_boot (int): Number of bootstrap samples.
        bootstraps (list[logistic_classifier]): List of fitted base classifiers.
        bootstrap_params (np.ndarray): Matrix of parameter estimates across bootstraps.
        params (np.ndarray): Averaged parameter estimates across bootstraps.
        model (abstract_logistic_regressor): Aggregated logistic regression model.
    """

    def __init__(self, x, y, probability: float = 0.5, seed: int = None, bootstraps: int = 1000):
        """Initialize a bagged logistic classifier.

        Args:
            x (np.ndarray or pd.DataFrame): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Binary response vector of shape (n_samples,).
            probability (float, optional): Classification threshold. Defaults to 0.5.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            bootstraps (int, optional): Number of bootstrap resamples. Defaults to 1000.
        """
        super(bagging_logistic_classifier, self).__init__(x, y, probability=probability)
        self.seed = seed
        self.n_boot = bootstraps
        model = lambda x, y: logistic_classifier(x, y, probability=probability)
        self.bootstraps = bootstrap(model, x, y, seed=seed, bootstraps=bootstraps)
        self.bootstrap_params = np.array([item.model.params for item in self.bootstraps])
        self.params = self.bootstrap_params.mean(0)
        vcov_params = (self.bootstrap_params - self.params).T @ (self.bootstrap_params - self.params) / self.n_boot
        self.model = abstract_logistic_regressor(x, y, self.params, vcov_params)

    def predict_proba(self, newx: np.ndarray) -> np.ndarray:
        """Predict class probabilities by majority vote from bootstrapped models.

        Args:
            newx (np.ndarray or pd.DataFrame): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, 2).
                Column 0 = probability of class 0, Column 1 = probability of class 1.
        """
        outp = (np.array([item.classify(newx) for item in self.bootstraps]).mean(0) > 0.5).astype(int)
        return hstack(1 - outp, outp)


class bagging_recursive_partitioning_classifier(recursive_partitioning_classifier):
    """Bagged ensemble of recursive partitioning classifiers (decision trees).

    Builds multiple trees on bootstrap resamples and selects the best-performing
    model (closest to ensemble fit) for stability. Can also be randomized to
    emulate random forest behavior.

    Attributes:
        seed (int, optional): Random seed for reproducibility.
        n_boot (int): Number of bootstrap samples.
        bootstraps (list[recursive_partitioning_classifier]): List of fitted base trees.
    """

    def __init__(self, x, y, seed: int = None, bootstraps: int = 1000,
                 sign_level: float = 0.95, max_level: int = 2,
                 random_x: bool = False, weights: np.ndarray = None):
        """Initialize a bagged recursive partitioning classifier.

        Args:
            x (np.ndarray or pd.DataFrame): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Class label vector of shape (n_samples,).
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            bootstraps (int, optional): Number of bootstrap resamples. Defaults to 1000.
            sign_level (float, optional): Significance level for splitting. Defaults to 0.95.
            max_level (int, optional): Maximum tree depth. Defaults to 2.
            random_x (bool, optional): If True, randomly subset features at each split
                (random forest-style). Defaults to False.
            weights (np.ndarray, optional): Sample weights. Defaults to None.
        """
        super(bagging_recursive_partitioning_classifier, self).__init__(x, y, max_level=1)
        self.seed, self.n_boot = seed, bootstraps
        model = lambda x, y: recursive_partitioning_classifier(
            x, y, sign_level=sign_level, max_level=max_level,
            random_x=random_x, weights=weights
        )
        self.bootstraps = bootstrap(model, x, y, seed=seed, bootstraps=bootstraps)
        fit = self.fitted
        indx = np.array([np.mean(item.fitted == fit) for item in self.bootstraps]).argmax()
        model = self.bootstraps[indx]
        for key, value in vars(model).items():
            setattr(self, key, value)

    def predict_proba(self, target: np.ndarray, fitted: bool = False) -> np.ndarray:
        """Predict class probabilities by aggregating bootstrap trees.

        Args:
            target (np.ndarray or pd.DataFrame): Feature matrix of shape (n_samples, n_features).
            fitted (bool, optional): If True, use fitted model attributes. Defaults to False.

        Returns:
            np.ndarray: Predicted class probabilities of shape (n_samples, n_classes).
        """
        return itemprob(np.array([item.classify(target, fitted=fitted) for item in self.bootstraps]), axis=0)


class random_forest_classifier(bagging_recursive_partitioning_classifier):
    """Random Forest classifier using recursive partitioning trees.

    This class implements a random forest by combining multiple
    recursive partitioning classifiers with bootstrapping and random
    feature sub-sampling. At each split, only a random subset of
    predictors is considered, reducing correlation between trees.

    Args:
        x (np.ndarray): Training feature matrix of shape (n_obs, n_feat).
        y (np.ndarray): Training class labels of shape (n_obs,).
        seed (int, optional): Random seed for reproducibility.
        bootstraps (int, optional): Number of bootstrap resamples. Defaults to 1000.
        sign_level (float, optional): Significance level for splitting. Defaults to 0.95.
        max_level (int, optional): Maximum tree depth. Defaults to 2.
        weights (np.ndarray, optional): Optional observation weights.

    Inherits:
        bagging_recursive_partitioning_classifier: Provides the ensemble
        logic and base classification tree structure.
    """


class adaboost_classifier(classification_model):
    """Classic AdaBoost classifier using recursive partitioning weak learners.

    This class implements the original AdaBoost-style boosting procedure for
    binary classification. Weak learners are fit sequentially using observation
    weights, and each learner receives a vote proportional to its accuracy.

    By default, the weak learner is a shallow recursive partitioning classifier
    (typically a stump), which is the classic boosting choice.

    Args:
        x (np.ndarray): Training feature matrix of shape (n_obs, n_feat).
        y (np.ndarray): Binary class labels of shape (n_obs,), coded as 0/1.
        seed (int, optional): Random seed for reproducibility.
        iterations (int, optional): Number of boosting rounds. Defaults to 100.
        sign_level (float, optional): Significance level for splitting in each
            weak learner. Defaults to 0.95.
        max_level (int | None, optional): Maximum tree depth for each weak
            learner. Defaults to 2.

    Attributes:
        seed (int | None): Random seed used for reproducibility.
        n_boot (int): Number of boosting rounds.
        learners (list[recursive_partitioning_classifier]): Fitted weak learners.
        learner_weights (np.ndarray): Vote weights for each weak learner.
        observation_weights (np.ndarray): Final observation weights after the
            last boosting round.
    """

    def __init__(
            self,
            x,
            y,
            seed=None,
            iterations=100,
            sign_level=0.95,
            max_level=2,
    ):
        super().__init__(x, y)

        if np.unique(y).shape[0] != 2:
            raise ValueError("adaboost_classifier currently supports binary classification only.")
        if not np.array_equal(np.sort(np.unique(y)), np.array([0, 1])):
            raise ValueError("adaboost_classifier requires binary labels coded as 0 and 1.")

        self.seed = seed
        self.n_boot = iterations
        self.learners = []
        self.learner_weights = []

        n_obs = x.shape[0]
        weights = np.full(n_obs, 1 / n_obs, dtype=float)

        # Optional reproducibility hook for any downstream numpy randomness
        if seed is not None:
            np.random.seed(seed)

        for _ in range(iterations):
            learner = recursive_partitioning_classifier(
                x,
                y,
                sign_level=sign_level,
                max_level=max_level,
                random_x=False,
                weights=weights,
            )

            pred = learner.classify(x)
            incorrect = (pred != y).astype(float)

            # Weighted misclassification rate
            error = np.sum(weights * incorrect)

            # Stopping rules
            if error <= 0:
                self.learners.append(learner)
                self.learner_weights.append(1.0)
                weights = np.full(n_obs, 1 / n_obs, dtype=float)
                break

            if error >= 0.5:
                if len(self.learners) == 0:
                    self.learners.append(learner)
                    self.learner_weights.append(1.0)
                break

            alpha = 0.5 * np.log((1 - error) / error)

            self.learners.append(learner)
            self.learner_weights.append(alpha)

            # Convert labels/predictions from {0,1} to {-1,+1}
            y_signed = 2 * y - 1
            pred_signed = 2 * pred - 1

            weights *= np.exp(-alpha * y_signed * pred_signed)
            weights /= weights.sum()

        self.learner_weights = np.array(self.learner_weights, dtype=float)
        self.observation_weights = weights

    def decision_function(self, newx):
        """Compute the boosted decision score for each observation.

        Args:
            newx (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Signed decision scores. Positive values favor class 1,
            negative values favor class 0.
        """
        if len(self.learners) == 0:
            return np.zeros(newx.shape[0], dtype=float)

        scores = np.zeros(newx.shape[0], dtype=float)
        for alpha, learner in zip(self.learner_weights, self.learners):
            pred = learner.classify(newx)
            pred_signed = 2 * pred - 1
            scores += alpha * pred_signed
        return scores

    def classify(self, newx):
        """Predict hard class labels by weighted majority vote.

        Args:
            newx (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted binary class labels of shape (n_new,).
        """
        return (self.decision_function(newx) > 0).astype(int)

    def predict_proba(self, newx):
        """Predict class probabilities from boosted decision scores.

        Uses the logistic transform of the AdaBoost decision function to produce
        a probability-like output for binary classification.

        Args:
            newx (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted probabilities of shape (n_new, 2), with
            columns corresponding to class 0 and class 1.
        """
        scores = self.decision_function(newx)
        p1 = 1 / (1 + np.exp(-2 * scores))
        return np.column_stack((1 - p1, p1))

    @property
    def fitted(self):
        """Return hard labels for the training data.

        Returns:
            np.ndarray: Integer class labels of shape (n_obs,).
        """
        return self.classify(self.x)


class BaggingLogisticClassifier(ClassificationModel):
    """Convenience wrapper for bagging logistic classification.

    Applies the unified :class:`ClassificationModel` interface to the
    :class:`bagging_logistic_classifier`, enabling construction from
    formulas and DataFrames.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`bagging_logistic_classifier`.

    Example:
        >>> model = BaggingLogisticClassifier.from_formula("y ~ x1 + x2", data=df)
        >>> model.classify(df[["x1", "x2"]])
    """

    MODEL_TYPE = bagging_logistic_classifier


class BaggingRecursivePartitioningClassifier(ClassificationModel):
    """Convenience wrapper for bagging recursive partitioning classification.

    Provides a formula/DataFrame interface for the
    :class:`bagging_recursive_partitioning_classifier`.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`bagging_recursive_partitioning_classifier`.

    Example:
        >>> model = BaggingRecursivePartitioningClassifier.from_formula("y ~ x1 + x2", data=df, max_level=3)
        >>> model.classify(df[["x1", "x2"]])
    """

    MODEL_TYPE = bagging_recursive_partitioning_classifier


class RandomForestClassifier(ClassificationModel):
    """Convenience wrapper for random forest classification.

    Provides a formula/DataFrame interface for the
    :class:`random_forest_classifier`, which implements an ensemble of
    recursive partitioning classifiers with random feature sub-sampling.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`random_forest_classifier`.

    Example:
        >>> model = RandomForestClassifier.from_formula("y ~ x1 + x2 + x3", data=df)
        >>> model.classify(df[["x1", "x2", "x3"]])
    """

    MODEL_TYPE = random_forest_classifier


class AdaBoostClassifier(ClassificationModel):
    """Convenience wrapper for AdaBoost classification.

    Provides a formula/DataFrame interface for the
    :class:`adaboost_classifier`.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`adaboost_classifier`.

    Example:
        >>> model = AdaBoostClassifier.from_formula("y ~ x1 + x2 + x3", data=df)
        >>> model.classify(df[["x1", "x2", "x3"]])
        >>> model.predict_proba(df[["x1", "x2", "x3"]])
    """

    MODEL_TYPE = adaboost_classifier
