# -*- coding: utf-8 -*-

"""Main module."""

import typing

import keras
from tensorflow import keras as tfkeras

import kerasltisubmission.provider as provider
from kerasltisubmission.exceptions import KerasLTISubmissionBadModelException

# from typing import TYPE_CHECKING


# if TYPE_CHECKING:  # pragma: no cover
#    import keras


class Submission:
    def __init__(
        self,
        assignment_id: provider.AnyIDType,
        model: typing.Union[keras.Model, tfkeras.Model],
    ) -> None:
        self.assignment_id = assignment_id
        if not (isinstance(model, keras.Model) or isinstance(model, tfkeras.Model)):
            raise KerasLTISubmissionBadModelException("Model must be a keras model!")
        self.model = model

    def submit(
        self, server: provider.LTIProvider, verbose: bool = True, reshape: bool = True,
    ) -> typing.Dict[str, typing.Dict[str, float]]:
        # Convenience method, it is preferred to use the server interface in the first place
        return server.submit(self, verbose=verbose, reshape=reshape)

    def __eq__(self, other: typing.Any) -> bool:
        if not isinstance(other, Submission):
            return NotImplemented

        return self.assignment_id == other.assignment_id and self.model == other.model
