# ==============================================================================
import os
from absl import app
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from stock_prediction_numpy import StockData
from datetime import date
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def main(argv):
    print(tf.version.VERSION)
    inference_folder = os.path.join(os.getcwd(), 'GOOG_20200704_b5f47746c83698528343678663ac3c96')

    # load future data
    data = StockData()
    min_max = MinMaxScaler(feature_range=(0, 1))
    x_test, y_test = data.generate_future_data(TIME_STEPS, min_max, date(2020, 7, 5), date(2021, 7, 5))

    # load the weights from our best model
    model = tf.keras.models.load_model(os.path.join(inference_folder, 'model_weights.h5'))
    model.summary()

    # display the content of the model
    baseline_results = model.evaluate(x_test, y_test, verbose=2)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()

    # perform a prediction
    test_predictions_baseline = model.predict(x_test)
    test_predictions_baseline = min_max.inverse_transform(test_predictions_baseline)
    test_predictions_baseline = pd.DataFrame(test_predictions_baseline)
    test_predictions_baseline.to_csv(os.path.join(inference_folder, 'inference.csv'))
    print(test_predictions_baseline)


if __name__ == '__main__':
    TIME_STEPS = 60
    app.run(main)