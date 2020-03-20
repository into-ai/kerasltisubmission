#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `kerasltisubmission` package."""

import json
import typing
import unittest.mock
from contextlib import contextmanager

import numpy as np
import pytest
import requests

from kerasltisubmission import exceptions
from kerasltisubmission.kerasltisubmission import Submission
from kerasltisubmission.provider import InputsType, LTIProvider, PredictionsType
from tests.mocks import JSONType, MockKerasModel, MockRequestsResponse


@pytest.fixture  # type: ignore
def provider() -> LTIProvider:
    """Sample provider"""
    return LTIProvider(
        input_api_endpoint="http://localhost:8080",
        submission_api_endpoint="http://localhost:8080/submit",
        user_token="7dd7367c-40c2-43cb-a052-bb04e1d0a858",
    )


@pytest.fixture  # type: ignore
def submission() -> Submission:
    """Sample submission"""
    # model = tf.keras.models.load_model(str((Path(__file__).parent / 'mnist.h5').absolute()))
    model = MockKerasModel(predicts=np.array([0, 2, 4, 6]))
    return Submission(assignment_id=12, model=model)


@pytest.fixture  # type: ignore
def prediction_input() -> InputsType:
    """Sample input matrix"""
    return [
        dict(
            hash="408358c06df48a3ade194e09db7b8113a272e072402650c1e283433cd75c9953",
            matrix=[
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.28627450980392155,
                    0.8,
                    1.0,
                    0.41568627450980394,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.06666666666666667,
                    0.2980392156862745,
                    0.5137254901960784,
                    0.7215686274509804,
                    0.9411764705882353,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.5882352941176471,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.00784313725490196,
                    0.16862745098039217,
                    0.4745098039215686,
                    0.8980392156862745,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.9803921568627451,
                    0.6274509803921569,
                    0.23529411764705882,
                    0.9921568627450981,
                    0.7725490196078432,
                    0.00784313725490196,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.027450980392156862,
                    0.6705882352941176,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.8862745098039215,
                    0.6705882352941176,
                    0.4,
                    0.13725490196078433,
                    0.0,
                    0.10980392156862745,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.0196078431372549,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.10588235294117647,
                    0.5294117647058824,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.807843137254902,
                    0.09019607843137255,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.10980392156862745,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.0196078431372549,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.00784313725490196,
                    0.5294117647058824,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.6352941176470588,
                    0.09019607843137255,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.12549019607843137,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.0196078431372549,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.43529411764705883,
                    0.9921568627450981,
                    0.984313725490196,
                    0.7215686274509804,
                    0.09019607843137255,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.07450980392156863,
                    0.8,
                    0.9921568627450981,
                    0.6627450980392157,
                    0.00784313725490196,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.592156862745098,
                    0.9921568627450981,
                    0.9529411764705882,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.22745098039215686,
                    0.7725490196078432,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.16862745098039217,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.592156862745098,
                    0.9921568627450981,
                    0.9529411764705882,
                    0.027450980392156862,
                    0.0,
                    0.0,
                    0.0,
                    0.023529411764705882,
                    0.5019607843137255,
                    0.9647058823529412,
                    0.9921568627450981,
                    0.9411764705882353,
                    0.4,
                    0.01568627450980392,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.2901960784313726,
                    0.9411764705882353,
                    0.9921568627450981,
                    0.7725490196078432,
                    0.14901960784313725,
                    0.01568627450980392,
                    0.3058823529411765,
                    0.7647058823529411,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.9137254901960784,
                    0.33725490196078434,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.4117647058823529,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.8352941176470589,
                    0.8,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.9647058823529412,
                    0.35294117647058826,
                    0.12941176470588237,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.1568627450980392,
                    0.9058823529411765,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.9764705882352941,
                    0.9098039215686274,
                    0.10588235294117647,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0784313725490196,
                    0.5882352941176471,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.7803921568627451,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.06666666666666667,
                    0.5764705882352941,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.7803921568627451,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.00392156862745098,
                    0.6196078431372549,
                    0.9921568627450981,
                    0.9254901960784314,
                    0.4666666666666667,
                    0.8705882352941177,
                    0.9921568627450981,
                    0.984313725490196,
                    0.43137254901960786,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.023529411764705882,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.4196078431372549,
                    0.0,
                    0.38823529411764707,
                    0.9725490196078431,
                    0.9921568627450981,
                    0.8784313725490196,
                    0.043137254901960784,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.023529411764705882,
                    0.9921568627450981,
                    0.8627450980392157,
                    0.06274509803921569,
                    0.0,
                    0.0,
                    0.9137254901960784,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.06274509803921569,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.023529411764705882,
                    0.9921568627450981,
                    0.7137254901960784,
                    0.01568627450980392,
                    0.0,
                    0.0,
                    0.8588235294117647,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.06274509803921569,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.023529411764705882,
                    0.9921568627450981,
                    0.9921568627450981,
                    0.5176470588235295,
                    0.0,
                    0.21568627450980393,
                    0.8117647058823529,
                    0.9921568627450981,
                    0.7372549019607844,
                    0.0196078431372549,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.01568627450980392,
                    0.5686274509803921,
                    0.5686274509803921,
                    0.5607843137254902,
                    0.16470588235294117,
                    0.4392156862745098,
                    0.9607843137254902,
                    0.9921568627450981,
                    0.4549019607843137,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ],
        )
    ]


@contextmanager
def assignment_input(
    json_data: JSONType = None,
    post_json_response: JSONType = None,
    status_code: int = 200,
    post_json_status_code: int = 200,
) -> typing.Iterator[typing.Tuple[unittest.mock.Mock, unittest.mock.Mock]]:
    with unittest.mock.patch("requests.get", autospec=True) as mocked_get:
        mocked_get.return_value = MockRequestsResponse(
            json_data=json_data or dict(), status_code=status_code
        )
        with unittest.mock.patch("requests.post", autospec=True) as mocked_post:
            mocked_post.return_value = MockRequestsResponse(
                json_data=post_json_response or dict(),
                status_code=post_json_status_code,
            )
            yield mocked_get, mocked_post


def test_input_request(
    provider: LTIProvider, submission: Submission, prediction_input: PredictionsType
) -> None:
    verbosity = [True, False]
    for v in verbosity:
        with assignment_input(
            json_data=dict(success=True, predict=prediction_input),
            post_json_response=dict(success=True, grade=0.5, accuracy=0.5),
        ) as (_, mocked_post):
            provider.submit(submission, verbose=v)
            posted = mocked_post.call_args_list
            assert len(posted) == 1
            mocked_post.assert_called_with(
                unittest.mock.ANY,
                data=json.dumps(
                    dict(
                        predictions={
                            "408358c06df48a3ade194e09db7b8113a272e072402650c1e283433cd75c9953": 3
                        },
                        user_token=provider.user_token,
                        assignment_id=submission.assignment_id,
                    )
                ),
                headers=unittest.mock.ANY,
            )


def test_no_input_raises(provider: LTIProvider, submission: Submission) -> None:
    with pytest.raises(exceptions.KerasLTISubmissionNoInputException):
        with assignment_input(
            json_data=dict(success=True, predict=dict()), status_code=200
        ) as (_, _):
            provider.submit(submission)


def test_bad_status_raises(
    provider: LTIProvider, submission: Submission, prediction_input: PredictionsType
) -> None:
    with pytest.raises(exceptions.KerasLTISubmissionBadResponseException):
        with assignment_input(
            json_data=dict(success=True, predict=prediction_input), status_code=500
        ) as (_, _):
            provider.submit(submission)


def test_exception_metadata(
    provider: LTIProvider, submission: Submission, prediction_input: PredictionsType
) -> None:
    try:
        with assignment_input(
            json_data=dict(success=True, predict=prediction_input),
            post_json_response=dict(success=False, grade=0.5, accuracy=0.5),
        ) as (_, _):
            submission.submit(provider)
    except exceptions.KerasLTISubmissionBadResponseException as e:
        assert e.return_code == 200
        assert e.api_endpoint == provider.submission_api_endpoint

    try:
        with assignment_input(
            json_data=dict(success=True, predict=prediction_input),
            post_json_response=dict(success=False, grade=0.5, accuracy=0.5),
        ) as (_, mocked_post):

            def simulate_connection_error(
                *args: typing.Any, **kwargs: typing.Any
            ) -> None:
                raise requests.exceptions.ConnectionError

            mocked_post.side_effect = simulate_connection_error
            submission.submit(provider)
    except exceptions.KerasLTISubmissionConnectionFailedException as e:
        assert isinstance(e.exc, requests.exceptions.ConnectionError)
        assert e.api_endpoint == provider.submission_api_endpoint
