from monai.networks.nets import Regressor


def get_regressor_model():
    model = Regressor(
        in_shape=[1, 197, 233, 189],
        out_shape=1,
        channels=(16, 32, 64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
    )

    return model
