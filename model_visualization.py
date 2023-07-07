import matplotlib.pyplot as plt
import numpy as np

# Load test data and predictions
# Assuming test_y and pred_y are your test output data and predictions
test_y = np.load('test_y.npy')
pred_y = np.load('pred_y.npy')

# Plot actual vs prediction
plt.plot(test_y, label='Actual')
plt.plot(pred_y, label='Predicted')
plt.legend()
plt.show()
