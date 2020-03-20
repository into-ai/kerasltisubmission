from tensorflow import keras
from kerasltisubmission import LTIProvider, Submission
from kerasltisubmission.exceptions import KerasLTISubmissionBadResponseException

data = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

total_classes = 10
train_vec_labels = keras.utils.to_categorical(train_labels, total_classes)
test_vec_labels = keras.utils.to_categorical(test_labels, total_classes)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(
    optimizer='sgd',
    loss='mean_squared_error',
    metrics=['accuracy'])

model.fit(train_images, train_vec_labels, epochs=2, verbose=True)

eval_loss, eval_accuracy = model.evaluate(test_images, test_vec_labels, verbose=False)
print("Model accuracy: %.2f" % eval_accuracy)

# See https://github.com/into-ai/kerasltiprovider
provider = LTIProvider(
    input_api_endpoint="http://localhost:8080",
    submission_api_endpoint="http://localhost:8080/submit",
    user_token="<your-token>",
)

submission = Submission(assignment_id=2, model=model)

try:
    provider.submit(submission, verbose=False)
    print("Submission was successful!")
except KerasLTISubmissionBadResponseException as e:
    print(e.message)
except Exception as e:
    print(str(e))

