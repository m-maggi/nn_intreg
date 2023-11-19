import numpy as np
from scipy.stats import gamma
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import polars as pl
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras import Model
from tensorflow.keras import initializers

tfd = tfp.distributions


# custom Loss as in https://stackoverflow.com/questions/64223840/use-additional-trainable-variables-in-keras-tensorflow-custom-loss-function
class MyLoss(Layer):
    def __init__(
        self,
        shape_par,
    ):
        super(MyLoss, self).__init__()
        self.a_hat = K.variable(shape_par)
        # self.a_hat = tf.cast(shape_par, tf.float32)

    def get_vars(self):
        return self.a_hat

    def custom_loss(self, y_true, y_pred):
        # y_pred is the mean
        shape_hat = tf.exp(self.a_hat)
        scales_hat = y_pred / shape_hat
        gamma_dist = tfd.Gamma(shape_hat, 1 / scales_hat)
        left = y_true[:, 0, tf.newaxis]
        right = y_true[:, 1, tf.newaxis]
        probability = gamma_dist.cdf(right) - gamma_dist.cdf(left)
        probability = tf.math.log(probability)
        return tf.reduce_mean(-probability)

    def call(self, y_true, y_pred):
        self.add_loss(self.custom_loss(y_true, y_pred))
        return y_pred


if __name__ == "__main__":
    # set seed for reproducibility of the synthetic data
    np.random.seed(500)
    # number of simulations for the synthetic data
    NSIM = 10000
    SHAPE_PAR = 1.6
    N_BINS = 30
    # define the covariates
    intercept_labels = ["intercept"]
    x1_labels = ["foo", "bar"]
    x2_labels = ["apple", "orange", "lemon"]
    # define the relativities
    intercept_factors = [16]
    x1_factors = [1.0, 1.07]
    x2_factors = [1.0, 0.95, 1.02]

    # function to collect covariates information
    def _helper_pl(cov_name, labels, factors):
        return pl.DataFrame({cov_name: labels, f"factor_{cov_name}": factors})

    sample = []
    # sampling of the labels and the factors
    for el in [
        ("intercept", intercept_labels, intercept_factors),
        ("X1", x1_labels, x1_factors),
        ("X2", x2_labels, x2_factors),
    ]:
        cov_name, labels, factors = el
        _df = _helper_pl(cov_name, labels, factors).sample(NSIM, with_replacement=True)
        sample.append(_df)
    sample = pl.concat(sample, how="horizontal")
    sample = (
        sample.with_columns(l=pl.concat_list((pl.col("^factor_.*$"))))
        .with_columns(
            factor_total=pl.col("l").list.eval(pl.col("*").product(), parallel=True)
        )
        .explode("factor_total")
        .select(["X1", "X2", "factor_total"])
    )

    y = gamma.rvs(size=NSIM, a=SHAPE_PAR, scale=sample["factor_total"] / SHAPE_PAR)
    sample = (
        sample.with_columns(y=pl.lit(y))
        .with_columns(breaks=pl.col("y").qcut(N_BINS, include_breaks=True))
        .unnest("breaks")
    )

    intervals = (
        sample["y_bin"]
        .cast(str)
        .str.strip_chars('( ] "')
        .str.split(", ")
        .list.to_struct()
        .struct.rename_fields(["left_break", "right_break"])
        .struct.unnest()
        .cast(pl.Float32)
    )

    sample = pl.concat([sample, intervals], how="horizontal").select(
        ["X1", "X2", "left_break", "right_break", "factor_total"]
    )
    X_y_train = sample.to_dummies(["X1", "X2"], drop_first=True)
    X_train = X_y_train.select(
        pl.all().exclude(["left_break", "right_break", "factor_total"])
    ).to_numpy()
    y_train = X_y_train.select(["left_break", "right_break"]).to_numpy()
    y_train = np.clip(y_train, 0.000001, 99999999)
    inputs = Input(shape=(X_y_train.shape[1] - 3,))
    y_input = Input(shape=(2,))
    # hidden1 = Dense(10)(inputs)
    constraint = tf.keras.constraints.MinMaxNorm(
        min_value=0.0, max_value=1.0, rate=1.0, axis=0
    )
    output = Dense(
        1,
        activation="exponential",
        kernel_initializer=initializers.Zeros(),
        bias_initializer=initializers.constant(np.log(20.0)),
        # kernel_constraint=constraint
    )(inputs)
    my_loss = MyLoss(
        np.log(1.6),
    )(
        y_input,
        output,
    )  # here can also initialize those var1, var2
    model = Model(inputs=[inputs, y_input], outputs=my_loss)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.4)
    model.compile(optimizer=optimizer, run_eagerly=False)
    print(model.summary())
    history = model.fit(
        [X_train, y_train],
        None,
        batch_size=NSIM,
        epochs=1000,
        validation_split=0.2,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=50)],
    )

    scales_hat = model.predict([X_train, y_train])
    results = (
        sample.with_columns(yhat=pl.lit(scales_hat.flatten()))
        .group_by(["X1", "X2"])
        .agg(truth=pl.col("factor_total").mean(), model=pl.col("yhat").mean())
    )
    print("Found shape:")
    print(np.exp(model.layers[-1].get_vars().numpy()))
    print("True shape:")
    print(SHAPE_PAR)
    print("E(Y|X) check:")
    print(results)
