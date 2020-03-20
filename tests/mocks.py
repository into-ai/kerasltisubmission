import typing

import numpy as np

JSONType = typing.Dict[str, typing.Any]


class MockRequestsResponse:
    def __init__(self, json_data: JSONType, status_code: int) -> None:
        self.json_data = json_data
        self.status_code = status_code

    def json(self) -> JSONType:
        return self.json_data


class MockKerasModel:
    def __init__(
        self,
        predicts: np.ndarray,
        input_shape: typing.Optional[typing.Tuple[int]] = None,
        output_shape: typing.Optional[typing.Tuple[int]] = None,
    ) -> None:
        self.predicts = predicts
        self.input_shape = input_shape or (None, 28, 28)
        self.output_shape = output_shape or (None, 4)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return [self.predicts for _ in range(len(inputs))]
