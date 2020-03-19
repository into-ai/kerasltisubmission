# -*- coding: utf-8 -*-

"""Main module."""

import typing
from typing import TYPE_CHECKING

import kerasltisubmission.provider as provider

if TYPE_CHECKING:  # pragma: no cover
    import keras.models


class Submission:
    def __init__(
        self, assignment_id: provider.AnyIDType, model: "keras.models.Model"
    ) -> None:
        self.assignment_id = assignment_id
        self.model = model

    def submit(self, server: provider.LTIProvider) -> None:
        if(model.input_shape != (None, 28, 28)):
            print("Wrong input shape. Expected: (None, 28, 28), Found: ", 
                  model.input_shape)
            return
        if(model.output_shape != (None, 10)):
            print("Wrong output shape. Expected: (None, 28, 28), Found: ", 
            model.output_shape)
            return

        # Convenience method, it is preferred to use the server interface in the first place
        server.submit(self)

    def __eq__(self, other: typing.Any) -> bool:
        if not isinstance(other, Submission):
            return NotImplemented

        return self.assignment_id == other.assignment_id and self.model == other.model
