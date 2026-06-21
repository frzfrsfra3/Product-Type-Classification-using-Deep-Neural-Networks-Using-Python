# Product Type Classification using Deep Neural Networks

An end-to-end deep learning solution to classify e-commerce customer reviews into specific product categories. The model uses Natural Language Processing (NLP) techniques and a neural network built with TensorFlow/Keras to predict whether a product is a Movie DVD, Electronics, or Kitchen Appliance based on the text description.

## 📋 Project Overview

In this project, we analyze customer product reviews from an e-commerce platform. The goal is to build and optimize a deep learning model that accurately categorizes the product type mentioned in each review. The model is evaluated on the **Accuracy** metric, as defined by the standard classification accuracy formula (ratio of correct predictions to total samples).

## 📊 Dataset

The dataset consists of two CSV files:
- **`train.csv`**: Used to train the model. Contains exactly 2 columns:
  - `desc` (str): The full text description/review of the product.
  - `label` (int): The target class. (0 = Movie DVD, 1 = Electronics, 2 = Kitchen Appliances).
- **`test.csv`**: Used for final prediction. Contains only the `desc` column.

## 🏗️ Methodology & Model Architecture

1. **Text Preprocessing**:
   - We use the `Tokenizer` from TensorFlow to build a vocabulary of the 10,000 most common words.
   - The text sequences are converted to integer arrays and padded to a uniform length of `150` tokens (`pad_sequences`) to feed into the neural network.
2. **Model Architecture**:
   - **Embedding Layer**: Maps integer-encoded vocabulary to dense vectors of size 16.
   - **Global Average Pooling 1D**: Reduces the dimensionality of the embedding output, making it easier for the model to learn while avoiding overfitting compared to Flatten.
   - **Dense Layer (64 units)**: Fully connected layer with `ReLU` activation.
   - **Dropout Layer (0.5 rate)**: Regularization technique to randomly drop 50% of the neurons during training to prevent overfitting.
   - **Output Layer (3 units)**: Dense layer with `Softmax` activation to output a probability distribution over the 3 product classes.
3. **Training**:
   - **Optimizer**: Adam.
   - **Loss Function**: Sparse Categorical Crossentropy (suitable for integer-labeled multiclass classification).
   - **Early Stopping**: Included to monitor validation loss. If the validation loss does not improve for 3 consecutive epochs, training stops and the best model weights are restored, ensuring the model generalizes well to unseen data.

## 🛠️ Requirements

The notebook requires the following Python packages. If you are running the provided Jupyter Notebook in the target environment, install them using:

```bash
pip install pandas numpy matplotlib tensorflow scikit-learn
