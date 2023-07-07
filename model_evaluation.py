from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load the trained model
model = load_model('LotkaVolterra_LSTM.h5')

# Load test data
# Assuming test_X and test_y are your test input and output data
test_X = np.load('test_X.npy')
test_y = np.load('test_y.npy')

# Predict on the test data
pred_y = model.predict(test_X)

# Calculate metrics
mae = mean_absolute_error(test_y, pred_y)
rmse = np.sqrt(mean_squared_error(test_y, pred_y))

print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}')
