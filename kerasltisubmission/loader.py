import abc
import typing
from typing import TYPE_CHECKING

import requests

from kerasltisubmission.exceptions import (
    KerasLTISubmissionBadResponseException,
    KerasLTISubmissionConnectionFailedException,
)

if TYPE_CHECKING:  # pragma: no cover
    from kerasltisubmission.provider import SingleInputType  # noqa: F401
    from kerasltisubmission.provider import (
        AnyIDType,  # noqa: F401
        InputsType,  # noqa: F401
    )  # noqa: F401


class InputLoader(abc.ABC):
    def __init__(self, assignment_id: "AnyIDType", input_api_endpoint: str) -> None:
        self.assignment_id = assignment_id
        self.input_api_endpoint = input_api_endpoint

    def load_next(self) -> typing.Optional["SingleInputType"]:
        pass

    def is_empty(self) -> bool:
        pass


class PartialLoader(InputLoader):
    def __init__(self, assignment_id: "AnyIDType", input_api_endpoint: str) -> None:
        super().__init__(assignment_id, input_api_endpoint)
        self.input_api_endpoint = input_api_endpoint
        self.currentIndex = 0
        self.batchIndex = 0
        self.batched: "InputsType" = list()

    def load_batch(self, input_id: int) -> "InputsType":
        try:
            r = requests.get(
                f"{self.input_api_endpoint}/assignment/{self.assignment_id}/inputs/{input_id}"
            )
            rr = r.json()
        except Exception as e:
            raise KerasLTISubmissionConnectionFailedException(
                self.input_api_endpoint, e
            ) from None
        if r.status_code == 200 and rr.get("success", True) is True:
            prediction_inputs: "InputsType" = rr.get("predict")
            return prediction_inputs
        else:
            raise KerasLTISubmissionBadResponseException(
                api_endpoint=self.input_api_endpoint,
                return_code=r.status_code,
                assignment_id=self.assignment_id,
                message=rr.get("error"),
            )

    def load_next(self) -> typing.Optional["SingleInputType"]:
        if self.currentIndex >= len(self.batched):
            # Grow batched
            self.batched += self.load_batch(self.batchIndex)
            self.batchIndex += 1
        n = (
            None
            if self.currentIndex >= len(self.batched)
            else self.batched[self.currentIndex]
        )
        self.currentIndex += 1
        return n

    def is_empty(self) -> bool:
        try:
            return len(self.load_batch(0)) < 1
        except (
            KerasLTISubmissionConnectionFailedException,
            KerasLTISubmissionBadResponseException,
        ):
            pass
        return False


class TotalLoader(InputLoader):
    def __init__(self, assignment_id: "AnyIDType", input_api_endpoint: str) -> None:
        try:
            r = requests.get(f"{input_api_endpoint}/assignment/{assignment_id}/inputs")
            rr = r.json()
        except Exception as e:
            raise KerasLTISubmissionConnectionFailedException(
                input_api_endpoint, e
            ) from None
        if r.status_code == 200 and rr.get("success", True) is True:
            self.inputs = rr.get("predict")
            self.currentIndex = 0
        else:
            raise KerasLTISubmissionBadResponseException(
                api_endpoint=input_api_endpoint,
                return_code=r.status_code,
                assignment_id=assignment_id,
                message=rr.get("error"),
            )

    def load_next(self) -> typing.Optional["SingleInputType"]:
        n = (
            None
            if self.currentIndex >= len(self.inputs)
            else self.inputs[self.currentIndex]
        )
        self.currentIndex += 1
        return n

    def is_empty(self) -> bool:
        return len(self.inputs) < 1
