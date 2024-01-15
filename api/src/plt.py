import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
# Given data
data = [{'eval_loss': 1.4694, 'learning_rate': 1.0000000000000002e-06, 'epoch': 0.41, 'step': 5}, {'loss': 1.4543410539627075, 'eval_runtime': 28.1169, 'eval_samples_per_second': 15.4, 'eval_steps_per_second': 1.956, 'epoch': 0.41, 'step': 5}, {'eval_loss': 1.4578, 'learning_rate': 2.0000000000000003e-06, 'epoch': 0.81, 'step': 10}, {'loss': 1.4477379322052002, 'eval_runtime': 28.1167, 'eval_samples_per_second': 15.4, 'eval_steps_per_second': 1.956, 'epoch': 0.81, 'step': 10}, {'eval_loss': 1.4368, 'learning_rate': 3e-06, 'epoch': 1.22, 'step': 15}, {'loss': 1.4320191354751587, 'eval_runtime': 28.1178, 'eval_samples_per_second': 15.399, 'eval_steps_per_second': 1.956, 'epoch': 1.22, 'step': 15}, {'eval_loss': 1.4286, 'learning_rate': 4.000000000000001e-06, 'epoch': 1.62, 'step': 20}, {'loss': 1.4168320894241333, 'eval_runtime': 28.1181, 'eval_samples_per_second': 15.399, 'eval_steps_per_second': 1.956, 'epoch': 1.62, 'step': 20}, {'eval_loss': 1.4183, 'learning_rate': 5e-06, 'epoch': 2.03, 'step': 25}, {'loss': 1.3926141262054443, 'eval_runtime': 28.1168, 'eval_samples_per_second': 15.4, 'eval_steps_per_second': 1.956, 'epoch': 2.03, 'step': 25}, {'eval_loss': 1.3894, 'learning_rate': 6e-06, 'epoch': 2.43, 'step': 30}, {'loss': 1.3611570596694946, 'eval_runtime': 28.1176, 'eval_samples_per_second': 15.4, 'eval_steps_per_second': 1.956, 'epoch': 2.43, 'step': 30}, {'eval_loss': 1.36, 'learning_rate': 7e-06, 'epoch': 2.84, 'step': 35}, {'loss': 1.3213225603103638, 'eval_runtime': 28.1177, 'eval_samples_per_second': 15.4, 'eval_steps_per_second': 1.956, 'epoch': 2.84, 'step': 35}, {'eval_loss': 1.3053, 'learning_rate': 8.000000000000001e-06, 'epoch': 3.24, 'step': 40}, {'loss': 1.2736258506774902, 'eval_runtime': 28.1168, 'eval_samples_per_second': 15.4, 'eval_steps_per_second': 1.956, 'epoch': 3.24, 'step': 40}, {'eval_loss': 1.2242, 'learning_rate': 9e-06, 'epoch': 3.65, 'step': 45}, {'loss': 1.2171036005020142, 'eval_runtime': 28.1162, 'eval_samples_per_second': 15.4, 'eval_steps_per_second': 1.956, 'epoch': 3.65, 'step': 45}, {'train_runtime': 588.8614, 'train_samples_per_second': 2.941, 'train_steps_per_second': 0.082, 'total_flos': 2.31784695595008e+16, 'train_loss': 1.377326230208079, 'epoch': 3.89, 'step': 48}]


# Extract relevant information
epochs = np.unique([entry['epoch'] for entry in data[:-1] if 'epoch' in entry])
training_losses = [entry['loss'] for entry in data if 'loss' in entry]
evaluation_losses = [entry['eval_loss'] for entry in data if 'eval_loss' in entry]

# Interpolate data for a smoother curve
interp_epochs = np.linspace(min(epochs), max(epochs), 1000)
interp_training_losses = interp1d(epochs, training_losses, kind='cubic', fill_value="extrapolate")(interp_epochs)
interp_evaluation_losses = interp1d(epochs, evaluation_losses, kind='cubic', fill_value="extrapolate")(interp_epochs)

# Plotting
plt.figure(figsize=(10, 6))

# Plot training losses as a smooth curve
plt.plot(interp_epochs, interp_training_losses, label='Training Loss')

# Plot evaluation losses as a smooth curve
plt.plot(interp_epochs, interp_evaluation_losses, label='Evaluation Loss')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss Over Epochs')
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('smooth_loss_curve.png')

# Show the plot
plt.show()