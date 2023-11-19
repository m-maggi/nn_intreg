import numpy as np
from scipy.stats import gamma
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import polars as pl
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras import Model


# custom Loss as in https://stackoverflow.com/questions/64223840/use-additional-trainable-variables-in-keras-tensorflow-custom-loss-function
class MyLoss(Layer):
    def __init__(self, var1, var2):
        super(MyLoss, self).__init__()
        self.var1 = K.variable(var1)  # or tf.Variable(var1) etc.
        self.var2 = K.variable(var2)

    def get_vars(self):
        return self.var1, self.var2

    def custom_loss(self, y_true, y_pred):
        return self.var1 * K.mean(K.square(y_true - y_pred)) + self.var2**2

    def call(self, y_true, y_pred):
        self.add_loss(self.custom_loss(y_true, y_pred))
        return y_pred


if __name__ == "__main__":
    # set seed for reproducibility of the synthetic data
    np.random.seed(500)
    # number of simulations for the synthetic data
    NSIM = 100
    SHAPE_PAR = 1.5
    N_BINS = 10
    # define the covariates
    intercept_labels = ["intercept"]
    x1_labels = ["foo", "bar"]
    x2_labels = ["apple", "orange", "lemon"]
    # define the relativities
    intercept_factors = [20.]
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
    sample = sample.with_columns(
        l=pl.concat_list((pl.col("^factor_.*$")))
    ).with_columns(
        factor_total=pl.col("l").list.eval(pl.col("*").product(), parallel=True)
    ).explode(
        "factor_total"
    ).select(
        ["X1", "X2", "factor_total"]
    )

    y = gamma.rvs(size=NSIM, a=SHAPE_PAR, scale=sample["factor_total"] / SHAPE_PAR)
    sample = sample.with_columns(
        y=pl.lit(y)
    ).with_columns(
        breaks=pl.col("y").qcut(N_BINS, include_breaks=True)
    ).unnest("breaks")
    
    intervals = sample['y_bin'].cast(str).str.strip_chars('( ] "').str.split(", ").list.to_struct().struct.unnest().cast(pl.Float32)
   
    X_train = np.zeros((10, 5))
    inputs = Input(shape=(X_train.shape[1],))
    y_input = Input(shape=(1,))
    hidden1 = Dense(10)(inputs)
    output = Dense(1)(hidden1)
    my_loss = MyLoss(0.5, 0.5)(
        y_input, output
    )  # here can also initialize those var1, var2
    model = Model(inputs=[inputs, y_input], outputs=my_loss)

    model.compile(optimizer="adam")
