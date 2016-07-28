import keras.backend as K
from diktya.regularizers import WeightOrthRegularizer

def test_weight_orth_regularizer():
    reg = WeightOrthRegularizer(weight=1.)
    loss = K.variable(0.)

    normal_filters = K.random_normal((32, 3, 3))
    uniform_filters = K.random_uniform((32, 3, 3))

    reg.set_param(normal_filters)
    loss_function = K.function([K.learning_phase()], reg(loss))
    normal_loss = loss_function((1,))

    reg.set_param(uniform_filters)
    loss_function = K.function([K.learning_phase()], reg(loss))
    uniform_loss = loss_function((1,))

    assert(normal_loss < uniform_loss)