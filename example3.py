from __future__ import print_function
import tensorflow.keras as keras

from kerasltisubmission import LTIProvider, Submission
from kerasltisubmission.exceptions import KerasLTISubmissionBadResponseException

model = keras.models.load_model('mymodel.h5')

# See https://github.com/into-ai/kerasltiprovider
provider = LTIProvider(
    input_api_endpoint="https://neuralnet.xopic.de/ltiprovider",
    submission_api_endpoint="https://neuralnet.xopic.de/ltiprovider/submit",
    user_token="ce85eb5d110faafc9798cf95aad8b7f8",
)

submission = Submission(assignment_id=3, model=model)

try:
    results = provider.submit(submission, verbose=True)
    for assignment_id, result in results.items():
        print(f"Submission was successful for assignment {assignment_id}!")
        print(f"    Your model has an accuracy of {result.get('accuracy') * 100}% on our validation data.")
        print(f"    You received a score of {result.get('grade') * 100}%")
except KerasLTISubmissionBadResponseException as e:
    raise e
except Exception as e:
    raise e
