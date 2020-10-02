#!/usr/bin/env python
# coding: utf-8

# Don't get me wrong, I love Tensorflow 2.0 (TF), and I love PyTorch and for complex deep learning problems they are incredible. Still, having grown up using Scikit-Learn and loving the API, I really miss the pipelining and hyperparameter optimization features that come with that ecosystem. If you want to do deep learning in TF, it is great, and you can iterate fast, but not every problem is a deep learning problem and its very hard to mix and match between Scikit-learn and Tensorflow cause you can't just pickle Tensorflow models in the same way.  
# 
# Variational Autoencoders are a wonderful bread-and-butter for deep learning dimensionality reduction. While I really try and avoid overcomplicated my modelling, there is a time and place where these models can shine.  One thing I like about Scikit-learn models is how easy it is to extend on them and customize their brilliant suite of tools for your own applications.  Doing large model search and testing can be really fast and fun in Scikit-learn and is really easy for beginners getting used to modelling in Python.  
# 
# Variational Autoencoders are a wonderful case of where Deep Learning steals elegantly from Statistics.  In Autoencoders a model is trained to map a high-dimensional dataset to a low-dimensional space and reconstruct it.  With Variational Autoencoders, we try to contain the low-dimensional space using an 'activity regularizer' which limits how extreme the values of the latent space can be and how the latent space is shaped.  In other model architectures, this could be the l2 loss, but in the case of the VAE, this loss function measures the 'distance' between the latent data and a normal distribution. This means that if we penalize this latent space enough and our reconstruction loss is low enough, we can try to generate new 'never-seen-before' data by sampling from a normal distribution and decoding it into the high-dimensional space.  
# 
# This is by no means a definative implementation, but I would love your input and idea on Scikit-learn, where you use it and why you love it!

# In[ ]:


get_ipython().system(' pip install hvplot')


# In[ ]:


import pandas as pd
import holoviews as hv
import hvplot.pandas
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
X = X/X.max()


# In[ ]:


from typing import List, Tuple

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.neural_network._base import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
from sklearn.neural_network._multilayer_perceptron import BaseMultilayerPerceptron
from sklearn.utils import check_X_y
from sklearn.utils.extmath import check_array, safe_sparse_dot
from sklearn.utils.validation import column_or_1d


def kl_loss_delta(z_mean: np.ndarray, z_log_sigma: np.ndarray = 0.0) -> np.ndarray:
    """
    see: https://keras.io/examples/variational_autoencoder/
    
    
    """
    return np.hstack((-2 * z_mean, 1 - np.exp(z_log_sigma)))


class VAE(BaseMultilayerPerceptron, TransformerMixin):
    def __init__(
        self,
        encoder_layer_sizes: Tuple[int] = (100, 2),
        activation: str = "relu",
        latent_regularizer: str = "kl",
        elbo_weight: float = 0.01,
        out_activation: str = "identity",
        loss: str = "squared_loss",
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size: str = "auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        power_t: float = 0.5,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state=None,
        tol: float = 1e-4,
        verbose: bool = False,
        warm_start: bool = False,
        momentum: float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        n_iter_no_change: int = 10,
        max_fun: int = 15000,
    ):
        """See the documentation for
        https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
        for more information information on the parameters

        :param encoder_layer_sizes: The ith element represents the number of neurons
            in the ith hidden layer with the last representing the latent space, 
            defaults to (100, 2)
        :param latent_regularizer: This is the regularization schema on the latent space,
            defaults to "kl"
        :param elbo_weight: This is the weight on the latent_regularizer, defaults to 0.01
        :param out_activation: This is the output activation which should map to the domain 
            of the data, defaults to "identity"
        """
        self.encoder_layer_sizes = encoder_layer_sizes
        self.latent_regularizer = latent_regularizer
        self.elbo_weight = 0.01
        self.out_activation = out_activation
        self.out_activation_ = out_activation

        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.loss = loss
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun

    def fit(self, X: np.array):
        super().__init__(
            hidden_layer_sizes=[
                *[
                    *self.encoder_layer_sizes[:-1],
                    self.encoder_layer_sizes[-1] * 2,
                    *reversed(self.encoder_layer_sizes[:-1]),
                ],
                X.shape[1],
            ],
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            loss=self.loss,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            power_t=self.power_t,
            max_iter=self.max_iter,
            shuffle=self.shuffle,
            random_state=self.random_state,
            tol=self.tol,
            verbose=self.verbose,
            warm_start=self.warm_start,
            momentum=self.momentum,
            nesterovs_momentum=self.nesterovs_momentum,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            n_iter_no_change=self.n_iter_no_change,
            max_fun=self.max_fun,
        )

        super().fit(X=X, y=X)

    def _backprop(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: List[np.ndarray],
        deltas: List[np.ndarray],
        coef_grads: List[np.ndarray],
        intercept_grads: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Compute the MLP loss function and its corresponding derivatives
        with respect to each parameter: weights and bias vectors.
        Parameters
        
        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        :param y: ndarray of shape (n_samples,)
            The target values.
        :param activations: list, length = n_layers - 1
             The ith element of the list holds the values of the ith layer.
        :param deltas: list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function
        :param coef_grads: list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.
        intercept_grads : list, length = n_layers - 1
        :param intercept_grads: The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.
        :coef_grads: The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration
        :return: loss : float
                 coef_grads : list, length = n_layers - 1
                 intercept_grads : list, length = n_layers - 1
        """
        n_samples = X.shape[0]

        # Forward propagate
        activations = self._forward_pass(activations)

        # Get loss
        loss_func_name = self.loss
        if loss_func_name == "log_loss" and self.out_activation_ == "logistic":
            loss_func_name = "binary_log_loss"
        loss = LOSS_FUNCTIONS[loss_func_name](y, activations[-1])
        # Add L2 regularization term to loss
        values = np.sum(np.array([np.dot(s.ravel(), s.ravel()) for s in self.coefs_]))
        loss += (0.5 * self.alpha) * values / n_samples

        # Backward propagate
        last = self.n_layers_ - 2

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        deltas[last] = activations[-1] - y

        # Compute gradient for the last layer
        coef_grads, intercept_grads = self._compute_loss_grad(
            last, n_samples, activations, deltas, coef_grads, intercept_grads
        )

        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 2, 0, -1):

            # VAE
            # latent activity regulizer
            if i == (len(self.encoder_layer_sizes) - 1):
                assert (
                    deltas[len(self.encoder_layer_sizes) - 1].shape[1]
                    == self.encoder_layer_sizes[-1] * 2
                )
                if self.latent_regularizer == "kl":
                    z = activations[len(self.encoder_layer_sizes)]
                    deltas[len(self.encoder_layer_sizes) - 1] -= (
                        self.elbo_weight
                        * kl_loss_delta(
                            z[:, : self.encoder_layer_sizes[-1]],
                            z[:, -self.encoder_layer_sizes[-1] :],
                        )
                    )

                elif self.latent_regularizer == "mmd":
                    raise NotImplementedError(
                        "That is not a implemented activity regularizer,\
                                        reverting to l2."
                    )
                elif self.latent_regularizer == "l2":
                    deltas[len(self.encoder_layer_sizes) - 1] -= self.elbo_weight * (
                        -activations[len(self.encoder_layer_sizes)]
                    )
                else:
                    raise ValueError(
                        "That is not a supported, activity regularizer,\
                                        non is being applied. "
                    )

            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)

            if i == (len(self.encoder_layer_sizes) - 1):
                inplace_derivative = DERIVATIVES[self.activation]
                inplace_derivative(activations[i], deltas[i - 1])

            coef_grads, intercept_grads = self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads, intercept_grads
            )

        return loss, coef_grads, intercept_grads

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model
        Parameters
        
        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        :return: ndarray of (n_samples, n_outputs)
            The latent space of the samples
        """
        X = check_array(X, accept_sparse=["csr", "csc", "coo"])

        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        layer_units = [X.shape[1]] + hidden_layer_sizes + [self.n_outputs_]

        # Initialize layers
        activations = [X]

        for i in range(self.n_layers_ - 1):
            activations.append(np.empty((X.shape[0], layer_units[i + 1])))
        # forward propagate
        self._forward_pass(activations)

        # VAE
        y_transform = activations[len(self.encoder_layer_sizes)]
        assert y_transform.shape[1] == self.encoder_layer_sizes[-1] * 2

        return y_transform[:, : self.encoder_layer_sizes[-1]]

    def _forward_pass(self, activations: List[np.ndarray]) -> List[np.ndarray]:
        """Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.
        Parameters
        
        :param activations: list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        """
        hidden_activation = ACTIVATIONS[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 1):

            # VAE
            # ignore setop_gradient for latent variance weights on forward-pass
            if i == len(self.encoder_layer_sizes):
                assert self.coefs_[i].shape[0] == self.encoder_layer_sizes[-1] * 2
                self.coefs_[i][: self.encoder_layer_sizes[-1], :] = 0.0

            activations[i + 1] = safe_sparse_dot(activations[i], self.coefs_[i])
            activations[i + 1] += self.intercepts_[i]

            # VAE
            # For the hidden layers that are not latent Z layers
            if (i + 1) != (self.n_layers_ - 1):
                if (i + 1) != len(self.encoder_layer_sizes):
                    activations[i + 1] = hidden_activation(activations[i + 1])
                else:
                    assert (
                        activations[i + 1].shape[1] == self.encoder_layer_sizes[-1] * 2
                    )
                    activations[i + 1] = activations[i + 1]

        # For the last layer
        output_activation = ACTIVATIONS[self.out_activation_]
        activations[i + 1] = output_activation(activations[i + 1])

        return activations

    def _validate_input(
        self, X: np.ndarray, y: np.ndarray, incremental
    ) -> Tuple[np.ndarray]:
        X, y = check_X_y(
            X, y, accept_sparse=["csr", "csc", "coo"], multi_output=True, y_numeric=True
        )
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)
        return X, y


# In[ ]:


model = VAE(encoder_layer_sizes=(25,5, 2,), 
            learning_rate_init=0.0001,
            activation='tanh', 
            elbo_weight=1.5, 
            max_iter=1000,
            out_activation='sigmoid',
            loss='log_loss',
            latent_regularizer='kl')
model.fit(X)


# In[ ]:


hv.extension('bokeh')

(pd.DataFrame(model.transform(X), columns=['x','y'])
 .assign(digit=y.astype(str))
 .hvplot.scatter(x='x',y='y', c='digit'))


# In[ ]:




