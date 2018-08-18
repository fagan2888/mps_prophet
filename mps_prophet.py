import gpflow as gp
from functools import reduce
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MpsProphet(object):
    def __init__(self,
                 ls=3.0, var=1.0, input_dim=1,
                 ls_trainable=True, var_trainable=True,
                 use_variational=False, n_series=1, rank=1):
        """
        Time series interpolation and forecasting library based on additive Gaussian processes
        :param ls:
        :param var:
        :param input_dim:
        :param ls_trainable:
        :param var_trainable:
        :param use_variational: use a Gaussian approximation of the posterior, may speed up inference and be less
            biased than MAP
        :param n_series:
        :param rank
        """
        self.input_dim = input_dim
        self.components = {
            'trend': gp.kernels.RBF(self.input_dim, lengthscales=ls, variance=var, name='trend')
        }
        self.components['trend'].lengthscales.trainable = ls_trainable
        self.components['trend'].variance.trainable = var_trainable
        self.model = None
        self._is_trained = False
        self.model_kernel = None
        self.mean_func = None
        self.n_series = n_series
        self.coregion_kernel = None if n_series == 1 else \
            gp.kernels.Coregion(self.input_dim, n_series, rank=rank, name='coregion')

    def _add_cov_fun(self, name: str, cov_fun):
        if name in self.components:
            raise ValueError("component already exists!")
        self.components[name] = cov_fun
        return self

    def add_bias(self, c=0.0, c_trainable=False):
        '''
        Adds a constant bias
        :param c:
        :param c_trainable:
        :return:
        '''
        self.mean_func = gp.mean_functions.Constant(c=c, name='bias')
        self.mean_func.c.trainable = c_trainable
        return self

    def add_seasonality(self, name, period, var=1.0, ls=2.0,
                        period_trainable=False, var_trainable=True, ls_trainable=True):
        """
        Add a seasonal component to the model
        :param name:
        :param period:
        :param var:
        :param ls:
        :param period_trainable:
        :param var_trainable:
        :param ls_trainable:
        :return:
        """
        cov_fun = gp.kernels.Periodic(self.input_dim, period=float(period), lengthscales=ls, variance=var, name=name)
        cov_fun.period.trainable = period_trainable
        cov_fun.lengthscales.trainable = ls_trainable
        cov_fun.variance.trainable = var_trainable
        return self._add_cov_fun(name, cov_fun)

    def add_modulated_seasonality(self, name, period, var=1.0, ls1=2.0, ls2=12.0,
                                  period_trainable=False, var_trainable=True, ls1_trainable=True, ls2_trainable=True):
        '''
        Add a seasonal component that also varies across time, may be unstable
        :param name:
        :param period:
        :param var:
        :param ls1:
        :param ls2:
        :param period_trainable:
        :param var_trainable:
        :param ls1_trainable:
        :param ls2_trainable:
        :return:
        '''
        cov_fun1 = gp.kernels.Periodic(self.input_dim, period=float(period), lengthscales=ls1, variance=var, name=name)
        cov_fun1.period.trainable = period_trainable
        cov_fun1.lengthscales.trainable = ls1_trainable
        cov_fun1.variance.trainable = var_trainable

        cov_fun2 = gp.kernels.RBF(self.input_dim, variance=1.0, lengthscales=ls2)
        cov_fun2.lengthscales.trainable = ls2_trainable
        cov_fun2.variance.trainable = False

        cov_fun = cov_fun1 * cov_fun2
        return self._add_cov_fun(name, cov_fun)

    def add_trend(self, name, ls=1.0, var=1.0, ls_trainable=True, var_trainable=True):
        cov_fun = gp.kernels.RBF(self.input_dim, variance=var, lengthscales=ls, name=name)
        cov_fun.lengthscales.trainable = ls_trainable
        cov_fun.variance.trainable = var_trainable
        return self._add_cov_fun(name, cov_fun)

    def add_linear_growth(self, name, var=1.0, var_trainable=True):
        '''
        Add a linear growth component to the model, not recommended
        :param name:
        :param var:
        :param var_trainable:
        :return:
        '''
        cov_fun = gp.kernels.Linear(1, variance=var, name=name)
        cov_fun.variance.trainable = var_trainable
        return self._add_cov_fun(name, cov_fun)

    def _build_kernel(self):
        kernel = reduce(lambda k1, k2: k1 + k2, self.components.values())
        if self.coregion_kernel is not None:
            kernel *= self.coregion_kernel

        self.model_kernel = kernel

    def build_model(self, X, y):
        """
        Prepares the model for inference
        :param X:
        :param y:
        :return:
        """
        self._build_kernel()

        self.model = gp.models.GPR(X.reshape((-1, self.input_dim)), y[:,None],
                                   kern=self.model_kernel, mean_function=self.mean_func)

        self.model.compile()

    def fit(self, X, y, maxiter=2000, step_size=0.01):
        '''
        Use MAP inference to fit the model
        :param X:
        :param y:
        :param maxiter:
        :return:
        '''
        if X.ndim == 1:
            X = X[:,None]
        if self.model is None:
            self.build_model(X, y)
        optimizer = gp.train.RMSPropOptimizer(step_size)
        optimizer.minimize(self.model, maxiter=maxiter)
        self._is_trained = True

    def summary(self):
        '''
        Show a summary of the model parameters
        :return:
        '''
        if self.model is None:
            raise AssertionError('Build or train the model first!')
        return self.model.as_pandas_table()

    def fit_transform(self, X, y, maxiter=2000):
        '''
        Fit and transform X, result is a tuple of (mean, var) vectors
        :param X:
        :param y:
        :param maxiter:
        :return:
        '''
        self.fit(X, y, maxiter)
        return self.model.predict_y(X)

    def transform(self, X):
        '''
        After model has been fit, transforms new X. Result is a tuple of (mean, var) vectors
        :param X:
        :return:
        '''
        if X.ndim == 1:
            X = X[:, None]
        if not self._is_trained:
            raise AssertionError('Train model first!')
        mean, var = self.model.predict_y(X)
        return mean.flatten(), var.flatten()

    def separate_components(self, x_star):
        '''
        Calculates the posterior mean and covariance
        mean = K_partial_xstar_X * K_sum_XX ^ -1 * f_sum(X)
        cov = K_partial_xstar_xstar - K_partial_xstar_X * K_sum_XX ^ -1 * K_partial_X_xstar
        :param x_star:
        :return:
        '''
        if x_star.ndim == 1:
            x_star = x_star[:, None]
        K_sum_XX = self.model_kernel.compute_K_symm(self.model.X.value)
        f = self.model.predict_y(self.model.X.value)[0]

        def calc_single_kernel(kernel: gp.kernels.Kernel):
            K_xstar_X = kernel.compute_K( x_star, self.model.X.value)
            K_xstar_xstar = kernel.compute_K_symm(x_star)
            mean = K_xstar_X.dot(np.linalg.solve(K_sum_XX, f))
            cov = K_xstar_xstar - K_xstar_X.dot(np.linalg.solve(K_sum_XX, K_xstar_X.T))
            return { 'mean': mean.flatten(), 'sigma': np.sqrt(np.diag(cov)) }

        return {
          name: calc_single_kernel(kern)
          for name, kern in self.components.items()
        }

    def MCMC(self):
        assert self._is_trained
        raise NotImplementedError('MCMC inference has not been implemented')


    def predict(self, x_star, band_width=2.0):
        '''
        Creates a dataframe with all the components of the additive model
        :param x_star:
        :param band_width:
        :return:
        '''
        components = self.separate_components(x_star)

        data_holder = {}
        for name, estimate in components.items():
            mean_est = estimate['mean']
            if name == 'trend' and self.mean_func is not None:
                mean_est += self.mean_func.c.value
            sigma_est = estimate['sigma']
            data_holder[name] = mean_est
            data_holder[name+'_lower'] = mean_est - band_width * sigma_est
            data_holder[name+'_upper'] = mean_est + band_width * sigma_est

        yhat, yhat_var = self.transform(x_star)
        data_holder['yhat'] = yhat
        yhat_sigma = np.sqrt(yhat_var)
        data_holder['yhat_lower'] = yhat - band_width * yhat_sigma
        data_holder['yhat_upper'] = yhat + band_width * yhat_sigma

        return pd.DataFrame(data_holder, index=x_star.flatten())

    def plot(self, prediction_frame):
        '''
        Plots the mean and confidence interval of the posterior estimate
        :param prediction_frame:
        :return:
        '''
        assert 'yhat' in prediction_frame.columns
        fig, ax= plt.subplots(1,1,figsize=(12,5))
        ax.plot(self.model.X.value, self.model.Y.value, 'k.')
        prediction_frame.yhat.plot(ax=ax)

        if 'yhat_lower' in prediction_frame.columns and 'yhat_upper' in prediction_frame.columns:
            ax.fill_between(prediction_frame.index.values,
                                 prediction_frame.yhat_lower.values,
                                 prediction_frame.yhat_upper.values,
                                 alpha=0.5, facecolors='xkcd:baby blue')
        ax.set_ylabel('yhat')
        return ax

    def plot_components(self, prediction_frame):
        components = self.components.keys()
        for comp in components:
            assert comp in prediction_frame.columns
        figs = []
        for comp in components:
            fig, ax = plt.subplots(1,1, figsize=(12,5))
            prediction_frame[comp].plot(ax=ax)
            if comp + '_lower' in prediction_frame.columns and comp + '_upper' in prediction_frame.columns:
                ax.fill_between(prediction_frame.index.values,
                                prediction_frame[comp+'_lower'].values,
                                prediction_frame[comp+'_upper'].values,
                                alpha=0.5, facecolors='xkcd:baby blue')
            ax.set_ylabel(comp)
            figs.append(fig)
        return figs




