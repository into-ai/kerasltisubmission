import typing
import unittest.mock

import keras
import numpy as np

JSONType = typing.Dict[str, typing.Any]


class MockRequestsResponse:
    def __init__(self, json_data: JSONType, status_code: int) -> None:
        self.json_data = json_data
        self.status_code = status_code

    def json(self) -> JSONType:
        return self.json_data


def mocked_keras_model(
    predicts: np.ndarray,
    input_shape: typing.Optional[typing.Tuple[int]] = None,
    output_shape: typing.Optional[typing.Tuple[int]] = None,
) -> unittest.mock.MagicMock:
    def mock_predict(inputs: np.ndarray) -> np.ndarray:
        return [predicts for _ in range(len(inputs))]

    model = unittest.mock.Mock(spec=keras.Model)
    model.predict.side_effect = mock_predict
    model.input_shape = input_shape or (None, 28, 28)
    model.output_shape = output_shape or (None, len(predicts))
    return model
