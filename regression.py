import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import Sequential

pd.options.display.float_format = "{:.1f}".format
tf.keras.backend.set_floatx('float32')

# Define the functions that build and train a model
def create_model(my_learning_rate, feature_layer):
  """Create and compile a simple linear regression model."""
  # Most simple tf.keras models are sequential.
  model = tf.keras.models.Sequential()

  # Add the layer containing the feature columns to the model.
  model.add(feature_layer)

  # Add one linear layer to the model to yield a simple linear regressor.
  model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

  # Construct the layers into a model that TensorFlow can execute.
  model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

  return model 


def train_model(model, dataset, epochs, batch_size, label_name):
  """Feed a dataset into the model in order to train it."""

  features = {name:np.array(value) for name, value in dataset.items()}
  label = np.array(features.pop(label_name))
  history = model.fit(x=features, y=label, batch_size=batch_size,
                      epochs=epochs, shuffle=True)

  # The list of epochs is stored separately from the rest of history.
  epochs = history.epoch
  
  # Isolate the mean absolute error for each epoch.
  hist = pd.DataFrame(history.history)
  rmse = hist["root_mean_squared_error"]

  return epochs, rmse 

# Define the plotting function
def plot_the_loss_curve(epochs, rmse):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs, rmse, label="Loss")
  plt.legend()
  plt.ylim([rmse.min()*0.94, rmse.max()* 1.05])
  plt.show() 





train_df = pd.read_csv("./NSPT_01.csv",  sep=";", decimal=",")
train_df = train_df.dropna()  # eliminamos las filas sin datos
test_df = pd.read_csv("./NSPT_01_test.csv",  sep=";", decimal=",")
# Mezclamos las muestras
train_df = train_df.reindex(np.random.permutation(train_df.index))

resolution_in_degrees = 0.005

# Create a new empty list that will eventually hold the generated feature column.
feature_columns = []

# Create a bucket feature column for latitude.
latitude_as_a_numeric_column = tf.feature_column.numeric_column("W")
latitude_boundaries = list(np.arange(min(train_df['W']), 
                                     max(train_df['W']), 
                                     resolution_in_degrees))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column, 
                                               latitude_boundaries)
feature_columns.append(latitude)

# Create a bucket feature column for longitude.
longitude_as_a_numeric_column = tf.feature_column.numeric_column("S")
longitude_boundaries = list(np.arange(min(train_df['S']), 
                                      max(train_df['S']), 
                                      resolution_in_degrees))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column, 
                                                longitude_boundaries)
feature_columns.append(longitude)

# Create a feature cross of latitude and longitude.
latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
feature_columns.append(crossed_feature)

# Convert the list of feature columns into a layer that will ultimately become
# part of the model. Understanding layers is not important right now.
feature_cross_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# Hyperparameters.
learning_rate = 0.04
epochs = 100
batch_size = 100
label_name = 'N_01'

my_model = create_model(learning_rate, feature_cross_feature_layer)

# Train the model on the training set.
epochs, rmse = train_model(my_model, train_df, epochs, batch_size, label_name)

plot_the_loss_curve(epochs, rmse)

print("\n: Evaluate the new model against the test set:")
test_features = {name:np.array(value) for name, value in test_df.items()}
test_label = np.array(test_features.pop(label_name))
my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size) 

predicted_values = my_model.predict(x=feature_cross_feature_layer)

# print("feature   label          predicted")
# print("  value   value          value")
# print("--------------------------------------")
# for i in range(n):
#   print ("%5.0f %6.0f %15.0f" % (training_df[feature][10000 + i],
#                                   training_df[label][10000 + i],
#                                   predicted_values[i][0] ))