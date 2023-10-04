import mlflow
import mlflow.sagemaker

import sys

# Load the traced model
model = mlflow.pyfunc.load_model(sys.argv[1])

# Register the model in MLflow
mlflow.register_model(model, "pd_model")

# Deploy the model to AWS SageMaker
mlflow.sagemaker.deploy("pd_model", "my_endpoint")

# Make a prediction on a new image
image = ...
prediction = model.predict(image)

# Print the prediction
print(prediction)
