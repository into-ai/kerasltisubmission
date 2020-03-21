# -*- coding: utf-8 -*-

"""Main module."""

import typing
from typing import TYPE_CHECKING

import kerasltisubmission.provider as provider
from kerasltisubmission.exceptions import KerasLTISubmissionBadModelException

if TYPE_CHECKING:  # pragma: no cover
    from tensorflow import keras as _tfkeras  # noqa: F401
    import keras as _keras  # noqa: F401


class Submission:
    def __init__(
        self,
        assignment_id: provider.AnyIDType,
        model: typing.Union["_keras.Model", "_tfkeras.Model"],
    ) -> None:
        self.assignment_id = assignment_id
        valid_model = True
        try:
            import keras
            from tensorflow import keras as tfkeras

            valid_model = isinstance(model, keras.Model) or isinstance(
                model, tfkeras.Model
            )
        except ImportError:
            pass

        if not valid_model:
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
