# Pneumonia Detection

This project uses PyTorch to develop a deep learning model to detect pneumonia in chest X-ray images.


## Requirements

To install the required packages, run the following command:

`pip3 install -r requirements.txt`

## Dataset

The dataset used for this project is the Chest X-ray Pneumonia Detection Challenge dataset from Kaggle. You can download the dataset from [here](https://www.kaggle.com/datasets/lasaljaywardena/pneumonia-chest-x-ray-dataset)

## Training the Model

To train the model, run the following command:

`python3 train.py`

This will train the model on the Chest X-ray Pneumonia Detection Challenge dataset.

## Evaluating the Model

To evaluate the model, run the following command:

`python3 evaluate.py`

This will evaluate the model on a held-out test set of images.

## Deploying the Model

To deploy the model, you can use the torch.jit.trace() function to trace the model and save it to a file. You can then load this file and use it to make predictions on new images.
Usage

To use the model to make a prediction on a new image, run the following command:

`python3 predict.py model_path image_path`

This will print the probability of pneumonia in the image.

## License

This project is licensed under the MIT License.

## Contact

If you have any questions or suggestions, please feel free to contact me at zqazi004@ucr.edu.