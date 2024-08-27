from math import ceil

from scikeras.wrappers import BaseEstimator, KerasRegressor
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import pyMAISE.settings as settings


class NeuralNetsWrapper(BaseEstimator):
    def __init__(
        self,
        num_layers,
        name,
        dropout,
        rate,
        loss,
        metrics,
        loss_weights,
        weighted_metrics,
        run_eagerly,
        steps_per_execution,
        batch_size,
        validation_batch_size,
        shuffle,
        callbacks,
        validation_split,
        epochs,
        warm_start,
        start_num_nodes,
        start_activation,
        start_use_bias,
        start_kernel_initializer,
        start_bias_initializer,
        start_kernel_regularizer,
        start_bias_regularizer,
        start_activity_regularizer,
        start_kernel_constraint,
        start_bias_constraint,
        input_dim,
        mid_num_node_strategy,
        mid_activation,
        mid_use_bias,
        mid_kernel_initializer,
        mid_bias_initializer,
        mid_kernel_regularizer,
        mid_bias_regularizer,
        mid_activity_regularizer,
        mid_kernel_constraint,
        mid_bias_constraint,
        end_num_nodes,
        end_activation,
        end_use_bias,
        end_kernel_initializer,
        end_bias_initializer,
        end_kernel_regularizer,
        end_bias_regularizer,
        end_activity_regularizer,
        end_kernel_constraint,
        end_bias_constraint,
        optimizer,
        learning_rate,
        beta_1,
        beta_2,
        epsilon,
        amsgrad,
        clipnorm,
        clipvalue,
        global_clipnorm,
        jit_compile,
    ):
        # Sequential
        self._num_layers = num_layers
        self._name = name
        self._dropout = dropout
        self._rate = rate
        self._loss = loss
        self._metrics = metrics
        self._loss_weights = loss_weights
        self._weighted_metrics = weighted_metrics
        self._run_eagerly = run_eagerly
        self._steps_per_execution = steps_per_execution
        self._jit_compile = jit_compile  # Used for both compile and adam
        self._batch_size = batch_size
        self._validation_batch_size = validation_batch_size
        self._shuffle = shuffle
        self._callbacks = callbacks
        self._validation_split = validation_split
        self._epochs = epochs
        self._warm_start = warm_start

        # Starting Layer
        self._start_num_nodes = start_num_nodes
        self._start_activation = start_activation
        self._start_use_bias = start_use_bias
        self._start_kernel_initializer = start_kernel_initializer
        self._start_bias_initializer = start_bias_initializer
        self._start_kernel_regularizer = start_kernel_regularizer
        self._start_bias_regularizer = start_bias_regularizer
        self._start_activity_regularizer = start_activity_regularizer
        self._start_kernel_constraint = start_kernel_constraint
        self._start_bias_constraint = start_bias_constraint
        self._input_dim = input_dim

        # Middle Layers
        self._mid_num_node_strategy = mid_num_node_strategy
        self._mid_activation = mid_activation
        self._mid_use_bias = mid_use_bias
        self._mid_kernel_initializer = mid_kernel_initializer
        self._mid_bias_initializer = mid_bias_initializer
        self._mid_kernel_regularizer = mid_kernel_regularizer
        self._mid_bias_regularizer = mid_bias_regularizer
        self._mid_activity_regularizer = mid_activity_regularizer
        self._mid_kernel_constraint = mid_kernel_constraint
        self._mid_bias_constraint = mid_bias_constraint

        # Ending Layer
        self._end_num_nodes = end_num_nodes
        self._end_activation = end_activation
        self._end_use_bias = end_use_bias
        self._end_kernel_initializer = end_kernel_initializer
        self._end_bias_initializer = end_bias_initializer
        self._end_kernel_regularizer = end_kernel_regularizer
        self._end_bias_regularizer = end_bias_regularizer
        self._end_activity_regularizer = end_activity_regularizer
        self._end_kernel_constraint = end_kernel_constraint
        self._end_bias_constraint = end_bias_constraint

        # Optimizer
        self._optimizer = optimizer

        # Adam
        self._learning_rate = learning_rate
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._amsgrad = amsgrad
        self._clipnorm = clipnorm
        self._clipvalue = clipvalue
        self._global_clipnorm = global_clipnorm

        self = self.NeuralNetsRegressor()

    def constant(self, num_layers: int, start_num_nodes: int, end_num_nodes: int):
        layers = [start_num_nodes] * num_layers
        layers[-1] = end_num_nodes
        return layers

    def linear(self, num_layers: int, start_num_nodes: int, end_num_nodes: int):
        layers = []

        # Determine slope
        increment = (end_num_nodes - start_num_nodes) / (num_layers - 1)

        # Starting number of nodes
        nodes = start_num_nodes

        # Fill layers with starting nodes and increment by slope
        for i in range(1, num_layers + 1):
            layers.append(ceil(nodes))
            nodes = nodes + increment

        return layers

    def fit(self, X, y):
        return self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)

    def score(self, X, y_true):
        return self._model.score(X, y_true)

    def get_params(self, deep=True):
        return {
            "num_layers": self._num_layers,
            "name": self._name,
            "dropout": self._dropout,
            "rate": self._rate,
            "loss": self._loss,
            "metrics": self._metrics,
            "loss_weights": self._loss_weights,
            "weighted_metrics": self._weighted_metrics,
            "run_eagerly": self._run_eagerly,
            "steps_per_execution": self._steps_per_execution,
            "batch_size": self._batch_size,
            "validation_batch_size": self._validation_batch_size,
            "shuffle": self._shuffle,
            "callbacks": self._callbacks,
            "validation_split": self._validation_split,
            "epochs": self._epochs,
            "warm_start": self._warm_start,
            "start_num_nodes": self._start_num_nodes,
            "start_activation": self._start_activation,
            "start_use_bias": self._start_use_bias,
            "start_kernel_initializer": self._start_kernel_initializer,
            "start_bias_initializer": self._start_bias_initializer,
            "start_kernel_regularizer": self._start_kernel_regularizer,
            "start_bias_regularizer": self._start_bias_regularizer,
            "start_activity_regularizer": self._start_activity_regularizer,
            "start_kernel_constraint": self._start_kernel_constraint,
            "start_bias_constraint": self._start_bias_constraint,
            "input_dim": self._input_dim,
            "mid_num_node_strategy": self._mid_num_node_strategy,
            "mid_activation": self._mid_activation,
            "mid_use_bias": self._mid_use_bias,
            "mid_kernel_initializer": self._mid_kernel_initializer,
            "mid_bias_initializer": self._mid_bias_initializer,
            "mid_kernel_regularizer": self._mid_kernel_regularizer,
            "mid_bias_regularizer": self._mid_bias_regularizer,
            "mid_activity_regularizer": self._mid_activity_regularizer,
            "mid_kernel_constraint": self._mid_kernel_constraint,
            "mid_bias_constraint": self._mid_bias_constraint,
            "end_num_nodes": self._end_num_nodes,
            "end_activation": self._end_activation,
            "end_use_bias": self._end_use_bias,
            "end_kernel_initializer": self._end_kernel_initializer,
            "end_bias_initializer": self._end_bias_initializer,
            "end_kernel_regularizer": self._end_kernel_regularizer,
            "end_bias_regularizer": self._end_bias_regularizer,
            "end_activity_regularizer": self._end_activity_regularizer,
            "end_kernel_constraint": self._end_kernel_constraint,
            "end_bias_constraint": self._end_bias_constraint,
            "optimizer": self._optimizer,
            "learning_rate": self._learning_rate,
            "beta_1": self._beta_1,
            "beta_2": self._beta_2,
            "epsilon": self._epsilon,
            "amsgrad": self._amsgrad,
            "clipnorm": self._clipnorm,
            "clipvalue": self._clipvalue,
            "global_clipnorm": self._global_clipnorm,
            "jit_compile": self._jit_compile,
        }

    def set_params(self, **parameters):
        # Change if user provided changes in dictionary
        if parameters is not None:
            for key, value in parameters.items():
                setattr(self, key, value)

        self = self.NeuralNetsRegressor()

        return self

    def NeuralNetsRegressor(self):
        # Determine number of nodes per layer through supported
        # functions or provided
        layers = []
        if isinstance(self._mid_num_node_strategy, str) and self._num_layers > 1:
            if self._mid_num_node_strategy == "constant":
                layers = self.constant(
                    self._num_layers, self._start_num_nodes, self._end_num_nodes
                )
            elif self._mid_num_node_strategy == "linear":
                layers = self.linear(
                    self._num_layers, self._start_num_nodes, self._end_num_nodes
                )
            else:
                raise Exception(
                    "Function ("
                    + self._mid_num_node_strategy
                    + ") is not supported, create your own callable"
                )
        else:
            layers = self._mid_num_node_strategy(
                self._num_layers, self._start_num_nodes, self._end_num_nodes
            )

        # Create the deep learning model
        model = Sequential(name=self._name)

        # Initial layer
        model.add(
            Dense(
                units=self._start_num_nodes,
                activation=self._start_activation,
                use_bias=self._start_use_bias,
                kernel_initializer=self._start_kernel_initializer,
                bias_initializer=self._start_bias_initializer,
                kernel_regularizer=self._start_kernel_regularizer,
                bias_regularizer=self._start_bias_regularizer,
                activity_regularizer=self._start_activity_regularizer,
                kernel_constraint=self._start_kernel_constraint,
                bias_constraint=self._start_bias_constraint,
                input_dim=self._input_dim,
            )
        )

        # Dropout
        if self._dropout:
            model.add(Dropout(rate=self._rate))

        # Hidden layers
        if self._num_layers > 2:
            for i in range(1, self._num_layers):
                model.add(
                    Dense(
                        units=layers[i],
                        activation=self._mid_activation,
                        use_bias=self._mid_use_bias,
                        kernel_initializer=self._mid_kernel_initializer,
                        bias_initializer=self._mid_bias_initializer,
                        kernel_regularizer=self._mid_kernel_regularizer,
                        bias_regularizer=self._mid_bias_regularizer,
                        activity_regularizer=self._mid_activity_regularizer,
                        kernel_constraint=self._mid_kernel_constraint,
                        bias_constraint=self._mid_bias_constraint,
                    )
                )

        # Ending Layer
        model.add(
            Dense(
                units=self._end_num_nodes,
                activation=self._end_activation,
                use_bias=self._end_use_bias,
                kernel_initializer=self._end_kernel_initializer,
                bias_initializer=self._end_bias_initializer,
                kernel_regularizer=self._end_kernel_regularizer,
                bias_regularizer=self._end_bias_regularizer,
                activity_regularizer=self._end_activity_regularizer,
                kernel_constraint=self._end_kernel_constraint,
                bias_constraint=self._end_bias_constraint,
            )
        )

        # Optimizer
        modelopt = None
        if self._optimizer == "adam":
            modelopt = Adam(
                learning_rate=self._learning_rate,
                beta_1=self._beta_1,
                beta_2=self._beta_2,
                epsilon=self._epsilon,
                amsgrad=self._amsgrad,
                clipnorm=self._clipnorm,
                clipvalue=self._clipvalue,
                global_clipnorm=self._global_clipnorm,
            )
        else:
            raise Exception("Optimizer (" + self._optimizer + ") is not supported")

        # Compile model
        model.compile(
            optimizer=modelopt,
            loss=self._loss,
            metrics=self._metrics,
            loss_weights=self._loss_weights,
            weighted_metrics=self._weighted_metrics,
            run_eagerly=self._run_eagerly,
        )

        # Set model to KerasRegressor from scikeras
        self._model = KerasRegressor(
            model=model,
            random_state=settings.values.random_state,
            warm_start=self._warm_start,
            optimizer=modelopt,
            loss=self._loss,
            metrics=self._metrics,
            batch_size=self._batch_size,
            validation_batch_size=self._validation_batch_size,
            verbose=settings.values.verbosity,
            callbacks=self._callbacks,
            validation_split=self._validation_split,
            shuffle=self._shuffle,
            run_eagerly=self._run_eagerly,
            epochs=self._epochs,
        )

        return self

    # ===========================================================
    # Getters
    @property
    def num_layers(self):
        return self._num_layers

    @property
    def name(self):
        return self._name

    @property
    def dropout(self):
        return self._dropout

    @property
    def rate(self):
        return self._rate

    @property
    def loss(self):
        return self._loss

    @property
    def metrics(self):
        return self._metrics

    @property
    def loss_weights(self):
        return self._loss_weights

    @property
    def run_eagerly(self):
        return self._run_eagerly

    @property
    def steps_per_execution(self):
        return self._steps_per_execution

    @property
    def jit_compile(self):
        return self._jit_compile

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def validation_batch_size(self):
        return self._validation_batch_size

    @property
    def shuffle(self):
        return self._shuffle

    @property
    def callbacks(self):
        return self._callbacks

    @property
    def validation_split(self):
        return self._validation_split

    @property
    def epochs(self):
        return self._epochs

    @property
    def warm_start(self):
        return self._warm_start

    @property
    def start_num_nodes(self):
        return self._start_num_nodes

    @property
    def start_activation(self):
        return self._start_activation

    @property
    def start_use_bias(self):
        return self._start_use_bias

    @property
    def start_kernel_initializer(self):
        return self._start_kernel_initializer

    @property
    def start_bias_initializer(self):
        return self._start_bias_initializer

    @property
    def start_kernel_regularizer(self):
        return self._start_kernel_regularizer

    @property
    def start_bias_regularizer(self):
        return self._start_bias_regularizer

    @property
    def start_activity_regularizer(self):
        return self._start_activity_regularizer

    @property
    def start_kernel_constraint(self):
        return self._start_kernel_constraint

    @property
    def start_bias_constraint(self):
        return self._start_bias_constraint

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def mid_num_node_strategy(self):
        return self._mid_num_node_strategy

    @property
    def mid_activation(self):
        return self._mid_activation

    @property
    def mid_use_bias(self):
        return self._mid_use_bias

    @property
    def mid_kernel_initializer(self):
        return self._mid_kernel_initializer

    @property
    def mid_bias_initializer(self):
        return self._mid_bias_initializer

    @property
    def mid_kernel_regularizer(self):
        return self._mid_kernel_regularizer

    @property
    def mid_bias_regularizer(self):
        return self._mid_bias_regularizer

    @property
    def mid_activity_regularizer(self):
        return self._mid_activity_regularizer

    @property
    def mid_kernel_constraint(self):
        return self._mid_kernel_constraint

    @property
    def mid_bias_constraint(self):
        return self._mid_bias_constraint

    @property
    def end_num_nodes(self):
        return self._end_num_nodes

    @property
    def end_activation(self):
        return self._end_activation

    @property
    def end_use_bias(self):
        return self._end_use_bias

    @property
    def end_kernel_initializer(self):
        return self._end_kernel_initializer

    @property
    def end_bias_initializer(self):
        return self._end_bias_initializer

    @property
    def end_kernel_regularizer(self):
        return self._end_kernel_regularizer

    @property
    def end_bias_regularizer(self):
        return self._end_bias_regularizer

    @property
    def end_activity_regularizer(self):
        return self._end_activity_regularizer

    @property
    def end_kernel_constraint(self):
        return self._start_end_constraint

    @property
    def end_bias_constraint(self):
        return self._end_bias_constraint

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def beta_1(self):
        return self._beta_1

    @property
    def beta_2(self):
        return self._beta_2

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def amsgrad(self):
        return self._amsgrad

    @property
    def clipnorm(self):
        return self._clipnorm

    @property
    def clipvalue(self):
        return self._clipvalue

    @property
    def global_clipnorm(self):
        return self._global_clipnorm

    # ===========================================================
    # Setters
    @num_layers.setter
    def num_layers(self, num_layers):
        self._num_layers = num_layers

    @name.setter
    def name(self, name):
        self._name = name

    @dropout.setter
    def dropout(self, dropout):
        self._dropout = dropout

    @rate.setter
    def rate(self, rate):
        self._rate = rate

    @loss.setter
    def loss(self, loss):
        self._loss = loss

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    @loss_weights.setter
    def loss_weights(self, loss_weights):
        self._loss_weights = loss_weights

    @run_eagerly.setter
    def run_eagerly(self, run_eagerly):
        self._run_eagerly = run_eagerly

    @steps_per_execution.setter
    def steps_per_execution(self, steps_per_execution):
        self._steps_per_execution = steps_per_execution

    @jit_compile.setter
    def jit_compile(self, jit_compile):
        self._jit_compile = jit_compile

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @validation_batch_size.setter
    def validation_batch_size(self, validation_batch_size):
        self._validation_batch_size = validation_batch_size

    @shuffle.setter
    def shuffle(self, shuffle):
        self._shuffle = shuffle

    @callbacks.setter
    def callbacks(self, callbacks):
        self._callbacks = callbacks

    @validation_split.setter
    def validation_split(self, validation_split):
        self._validation_split = validation_split

    @epochs.setter
    def epochs(self, epochs):
        self._epochs = epochs

    @warm_start.setter
    def warm_start(self, warm_start):
        self._warm_start = warm_start

    @start_num_nodes.setter
    def start_num_nodes(self, start_num_nodes):
        self._start_num_nodes = start_num_nodes

    @start_activation.setter
    def start_activation(self, start_activation):
        self._start_activation = start_activation

    @start_use_bias.setter
    def start_use_bias(self, start_use_bias):
        self._start_use_bias = start_use_bias

    @start_kernel_initializer.setter
    def start_kernel_initializer(self, start_kernel_initializer):
        self._start_kernel_initializer = start_kernel_initializer

    @start_bias_initializer.setter
    def start_bias_initializer(self, start_bias_initializer):
        self._start_bias_initializer = start_bias_initializer

    @start_kernel_regularizer.setter
    def start_kernel_regularizer(self, start_kernel_regularizer):
        self._start_kernel_regularizer = start_kernel_regularizer

    @start_bias_regularizer.setter
    def start_bias_regularizer(self, start_bias_regularizer):
        self._start_bias_regularizer = start_bias_regularizer

    @start_activity_regularizer.setter
    def start_activity_regularizer(self, start_activity_regularizer):
        self._start_activity_regularizer = start_activity_regularizer

    @start_kernel_constraint.setter
    def start_kernel_constraint(self, start_kernel_constraint):
        self._start_kernel_constraint = start_kernel_constraint

    @start_bias_constraint.setter
    def start_bias_constraint(self, start_bias_constraint):
        self._start_bias_constraint = start_bias_constraint

    @input_dim.setter
    def input_dim(self, input_dim):
        self._input_dim = input_dim

    @mid_num_node_strategy.setter
    def mid_num_node_strategy(self, mid_num_node_strategy):
        assert isinstance(mid_num_node_strategy, str) or isinstance(
            mid_num_node_strategy, callable
        )
        self._mid_num_node_strategy = mid_num_node_strategy

    @mid_activation.setter
    def mid_activation(self, mid_activation):
        self._mid_activation = mid_activation

    @mid_use_bias.setter
    def mid_use_bias(self, mid_use_bias):
        self._mid_use_bias = mid_use_bias

    @mid_kernel_initializer.setter
    def mid_kernel_initializer(self, mid_kernel_initializer):
        self._mid_kernel_initializer = mid_kernel_initializer

    @mid_bias_initializer.setter
    def mid_bias_initializer(self, mid_bias_initializer):
        self._mid_bias_initializer = mid_bias_initializer

    @mid_kernel_regularizer.setter
    def mid_kernel_regularizer(self, mid_kernel_regularizer):
        self._mid_kernel_regularizer = mid_kernel_regularizer

    @mid_bias_regularizer.setter
    def mid_bias_regularizer(self, mid_bias_regularizer):
        self._mid_bias_regularizer = mid_bias_regularizer

    @mid_activity_regularizer.setter
    def mid_activity_regularizer(self, mid_activity_regularizer):
        self._mid_activity_regularizer = mid_activity_regularizer

    @mid_kernel_constraint.setter
    def mid_kernel_constraint(self, mid_kernel_constraint):
        self._mid_kernel_constraint = mid_kernel_constraint

    @mid_bias_constraint.setter
    def mid_bias_constraint(self, mid_bias_constraint):
        self._mid_bias_constraint = mid_bias_constraint

    @end_num_nodes.setter
    def end_num_nodes(self, end_num_nodes):
        self._end_num_nodes = end_num_nodes

    @end_activation.setter
    def end_activation(self, end_activation):
        self._end_activation = end_activation

    @end_use_bias.setter
    def end_use_bias(self, end_use_bias):
        self._end_use_bias = end_use_bias

    @end_kernel_initializer.setter
    def end_kernel_initializer(self, end_kernel_initializer):
        self._end_kernel_initializer = end_kernel_initializer

    @end_bias_initializer.setter
    def end_bias_initializer(self, end_bias_initializer):
        self._end_bias_initializer = end_bias_initializer

    @end_kernel_regularizer.setter
    def end_kernel_regularizer(self, end_kernel_regularizer):
        self._end_kernel_regularizer = end_kernel_regularizer

    @end_bias_regularizer.setter
    def end_bias_regularizer(self, end_bias_regularizer):
        self._end_bias_regularizer = end_bias_regularizer

    @end_activity_regularizer.setter
    def end_activity_regularizer(self, end_activity_regularizer):
        self._end_activity_regularizer = end_activity_regularizer

    @end_kernel_constraint.setter
    def end_kernel_constraint(self, end_kernel_constraint):
        self._start_end_constraint = end_kernel_constraint

    @end_bias_constraint.setter
    def end_bias_constraint(self, end_bias_constraint):
        self._end_bias_constraint = end_bias_constraint

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    @beta_1.setter
    def beta_1(self, beta_1):
        self._beta_1 = beta_1

    @beta_2.setter
    def beta_2(self, beta_2):
        self._beta_2 = beta_2

    @epsilon.setter
    def epsilon(self, epsilon):
        self._epsilon = epsilon

    @amsgrad.setter
    def amsgrad(self, amsgrad):
        self._amsgrad = amsgrad

    @clipnorm.setter
    def clipnorm(self, clipnorm):
        self._clipnorm = clipnorm

    @clipvalue.setter
    def clipvalue(self, clipvalue):
        self._clipvalue = clipvalue

    @global_clipnorm.setter
    def global_clipnorm(self, global_clipnorm):
        self._global_clipnorm = global_clipnorm
