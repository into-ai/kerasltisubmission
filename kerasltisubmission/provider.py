import json
import logging
import typing
from typing import TYPE_CHECKING

import numpy as np
import progressbar
import requests

from kerasltisubmission import loader
from kerasltisubmission.exceptions import (
    KerasLTISubmissionBadResponseException,
    KerasLTISubmissionConnectionFailedException,
    KerasLTISubmissionException,
    KerasLTISubmissionInputException,
    KerasLTISubmissionInvalidSubmissionException,
    KerasLTISubmissionNoInputException,
)

if TYPE_CHECKING:  # pragma: no cover
    from kerasltisubmission import Submission  # noqa: F401
    from kerasltisubmission.kerasltisubmission import ModelType  # noqa: F401


log = logging.getLogger("kerasltisubmission")
log.addHandler(logging.NullHandler())

AnyIDType = typing.Union[str, int]
SingleInputType = typing.Dict[str, typing.Any]
InputsType = typing.List[SingleInputType]
PredictionsType = typing.Dict[str, typing.Any]


class LTIProvider:
    def __init__(
        self,
        input_api_endpoint: str,
        submission_api_endpoint: str,
        user_token: AnyIDType,
    ) -> None:
        self.user_token = user_token
        self.input_api_endpoint = input_api_endpoint
        self.submission_api_endpoint = submission_api_endpoint

    def expects_partial_inputs(
        self, assignment_id: AnyIDType
    ) -> typing.Tuple[bool, typing.Optional[int]]:
        try:
            r = requests.get(f"{self.input_api_endpoint}/assignments")
            rr = r.json()
        except Exception as e:
            raise KerasLTISubmissionConnectionFailedException(
                self.input_api_endpoint, e
            ) from None
        is_partial = False
        validation_set_size = None
        if r.status_code == 200:
            all_assignments = rr.get("assignments")
            assignments = [
                a
                for a in all_assignments
                if str(a.get("identifier")) == str(assignment_id)
            ]
            if len(assignments) > 0:
                try:
                    validation_set_size = int(assignments[0].get("validation_set_size"))
                except ValueError:
                    pass
                if validation_set_size:
                    is_partial = assignments[0].get("partial_loading", False)
        return is_partial, validation_set_size

    def guess(
        self, assignment_id: AnyIDType, predictions: PredictionsType
    ) -> typing.Tuple[float, float]:
        log.debug(
            f"Submitting {len(predictions)} predictions to the provider for grading"
        )
        headers = {"content-type": "application/json"}
        if not len(predictions) > 0:
            raise KerasLTISubmissionInvalidSubmissionException(predictions)
        try:
            r = requests.post(
                self.submission_api_endpoint,
                data=json.dumps(
                    dict(
                        predictions=predictions,
                        user_token=self.user_token,
                        assignment_id=assignment_id,
                    )
                ),
                headers=headers,
            )
            rr = r.json()
        except Exception as e:
            log.error(e)
            raise KerasLTISubmissionConnectionFailedException(
                self.submission_api_endpoint, e
            ) from None
        try:
            assert r.status_code == 200 and rr.get("error") is None
            log.debug(
                f"Sent {len(predictions)} predictions to the provider for grading"
            )
            log.info(f"Successfully submitted assignment {assignment_id} for grading")
            return (
                round(rr.get("accuracy"), ndigits=2),
                round(rr.get("grade"), ndigits=2),
            )
        except (AssertionError, KeyError, ValueError, TypeError):
            raise KerasLTISubmissionBadResponseException(
                api_endpoint=self.submission_api_endpoint,
                return_code=r.status_code,
                assignment_id=assignment_id,
                message=rr.get("error"),
            )

    @classmethod
    def perform_reshape(
        cls,
        model: "ModelType",
        input_matrix: np.ndarray,
        reshape: typing.Optional[bool] = True,
    ) -> np.ndarray:
        input_shape = input_matrix.shape
        expected_input_shape = (None, *input_shape[1:])
        if model.input_shape != expected_input_shape:
            output_shape_mismatch = f"Input shape mismatch: Got {model.input_shape} but expected {expected_input_shape}"
            if reshape is not True:
                raise KerasLTISubmissionInputException(output_shape_mismatch)
            # Try to reshape
            log.warning(output_shape_mismatch)
            return input_matrix.reshape(cls.safe_shape(model.input_shape))

    @classmethod
    def safe_shape(
        cls, shape: typing.Tuple[typing.Optional[typing.Any], ...]
    ) -> typing.Tuple[int, ...]:
        escaped = []
        for dim in shape:
            escaped.append(-1 if not dim else dim)
        return tuple(escaped)

    def submit(
        self,
        s: typing.Union["Submission", typing.List["Submission"]],
        verbose: bool = True,
        strict: bool = False,
        reshape: bool = False,
        expected_output_shape: typing.Optional[
            typing.Tuple[typing.Optional[typing.Any], ...]
        ] = None,
    ) -> typing.Dict[str, typing.Dict[str, float]]:
        results = dict()
        if isinstance(s, list):
            submissions = s
        else:
            submissions = [s]
        for sub in submissions:

            if (
                strict
                and expected_output_shape
                and not sub.model.output_shape == expected_output_shape
            ):
                raise KerasLTISubmissionInputException(
                    f"Model has invalid output shape: Got {sub.model.output_shape} but expected {expected_output_shape}"
                )

            # Get assignment inputs and propagate errors
            is_partial, validation_set_size = self.expects_partial_inputs(
                sub.assignment_id
            )
            loader_cls = loader.PartialLoader if is_partial else loader.TotalLoader
            assignment_loader = loader_cls(sub.assignment_id, self.input_api_endpoint)
            if assignment_loader.is_empty():
                raise KerasLTISubmissionNoInputException(
                    self.input_api_endpoint, sub.assignment_id
                )

            predictions: PredictionsType = dict()

            if not verbose or validation_set_size is None:
                # Collect all input matrices
                collected: "InputsType" = []
                while True:
                    if (
                        validation_set_size is not None
                        and len(collected) >= validation_set_size
                    ):
                        break
                    try:
                        loaded_input = assignment_loader.load_next()
                    except (
                        KerasLTISubmissionConnectionFailedException,
                        KerasLTISubmissionBadResponseException,
                    ):
                        break
                    if loaded_input is None:
                        break
                    collected.append(loaded_input)
                net_out = sub.model.predict(
                    np.array([np.asarray(c.get("matrix")) for c in collected])
                )
                predictions = {
                    str(c.get("hash")): int(np.argmax(pred))
                    for c, pred in zip(collected, net_out)
                }
            else:
                errors: typing.List[Exception] = []
                for i in progressbar.progressbar(
                    range(validation_set_size), redirect_stdout=True
                ):
                    loaded_input = assignment_loader.load_next()
                    if loaded_input is None:
                        raise KerasLTISubmissionInputException(f"Missing input {i}")
                    try:
                        input_matrix = loaded_input.get("matrix")
                        input_hash = loaded_input.get("hash")
                        probabilities = sub.model.predict(
                            np.expand_dims(np.asarray(input_matrix), axis=0)
                        )
                        prediction = np.argmax(probabilities)
                        if input_hash:
                            predictions[input_hash] = int(prediction)
                    except Exception as e:
                        raise e
                        if e not in errors:
                            errors.append(e)
                if len(errors) > 0:
                    raise KerasLTISubmissionException()

            accuracy, grade = self.guess(sub.assignment_id, predictions)
            results[str(sub.assignment_id)] = dict(accuracy=accuracy, grade=grade)
        return results
