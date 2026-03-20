# forecasting_multivariate/ETTh1.json

data = {

"forecasting_multivariate_ETTh1_24": {
    "valid_loss": [ 0.488, 0.472, 0.468, 0.467, 0.461, 0.472, 0.448, 0.451, 0.444, 0.456, 0.448, 0.443, 0.437, 0.443, 0.435, 0.449, 0.433, 0.434, 0.438, 0.441, 0.431, 0.438, 0.431, 0.432, 0.429, 0.430, 0.429, 0.429, 0.432, 0.428 ],
    "valid_mae": [ 0.477, 0.467, 0.463, 0.464, 0.463, 0.473, 0.457, 0.459, 0.453, 0.467, 0.459, 0.454, 0.450, 0.454, 0.451, 0.457, 0.449, 0.451, 0.450, 0.452, 0.448, 0.450, 0.447, 0.448, 0.448, 0.447, 0.448, 0.446, 0.449, 0.446 ],
    "test_loss": [ 0.338, 0.330, 0.329, 0.331, 0.327, 0.337, 0.328, 0.328, 0.321, 0.335, 0.327, 0.321, 0.325, 0.322, 0.326, 0.321, 0.323, 0.324, 0.319, 0.323, 0.321, 0.323, 0.319, 0.319, 0.319, 0.318, 0.319, 0.318, 0.320, 0.316 ],
    "test_mae": [ 0.385, 0.379, 0.379, 0.379, 0.377, 0.385, 0.376, 0.378, 0.372, 0.382, 0.376, 0.372, 0.374, 0.377, 0.376, 0.372, 0.373, 0.374, 0.370, 0.373, 0.371, 0.372, 0.370, 0.370, 0.371, 0.369, 0.370, 0.369, 0.370, 0.369 ]
}

}



seleted_dict = "forecasting_multivariate_ETTh1_24"

import matplotlib.pyplot as plt

valid_loss = data[seleted_dict]["valid_loss"]
valid_mae = data[seleted_dict]["valid_mae"]
test_loss = data[seleted_dict]["test_loss"]
test_mae = data[seleted_dict]["test_mae"]

epochs = list(range(1, len(valid_loss) + 1))

def get_best(values):
    best_value = min(values)
    best_epoch = values.index(best_value) + 1
    return best_epoch, best_value

# Legjobb pontok
best_valid_loss_epoch, best_valid_loss = get_best(valid_loss)
best_test_loss_epoch, best_test_loss = get_best(test_loss)
best_valid_mae_epoch, best_valid_mae = get_best(valid_mae)
best_test_mae_epoch, best_test_mae = get_best(test_mae)

# -------- LOSS FIGURE --------
plt.figure(figsize=(10, 5))
plt.plot(epochs, valid_loss, marker='o', label='Validation Loss', color="blue")
plt.plot(epochs, test_loss, marker='s', label='Test Loss', color="orange")

plt.scatter(best_valid_loss_epoch, best_valid_loss, s=100,
            label=f'Best Val Loss ({best_valid_loss:.3f})', color="blue")
plt.scatter(best_test_loss_epoch, best_test_loss, s=100,
            label=f'Best Test Loss ({best_test_loss:.3f})', color="orange")

plt.axvline(best_valid_loss_epoch, linestyle='--', alpha=0.6, color="blue")
plt.axvline(best_test_loss_epoch, linestyle='--', alpha=0.6, color="orange")

plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs[::2])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# -------- MAE FIGURE --------
plt.figure(figsize=(10, 5))
plt.plot(epochs, valid_mae, marker='o', label='Validation MAE', color="blue")
plt.plot(epochs, test_mae, marker='s', label='Test MAE', color="orange")

plt.scatter(best_valid_mae_epoch, best_valid_mae, s=100,
            label=f'Best Val MAE ({best_valid_mae:.3f})', color="blue")
plt.scatter(best_test_mae_epoch, best_test_mae, s=100,
            label=f'Best Test MAE ({best_test_mae:.3f})', color="orange")

plt.axvline(best_valid_mae_epoch, linestyle='--', alpha=0.6, color="blue")
plt.axvline(best_test_mae_epoch, linestyle='--', alpha=0.6, color="orange")

plt.title('MAE over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.xticks(epochs[::2])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# -------- SUMMARY --------
print("Best metrics:")
print(f"Validation Loss: epoch {best_valid_loss_epoch}, value = {best_valid_loss:.3f}")
print(f"Test Loss:       epoch {best_test_loss_epoch}, value = {best_test_loss:.3f}")
print(f"Validation MAE:  epoch {best_valid_mae_epoch}, value = {best_valid_mae:.3f}")
print(f"Test MAE:        epoch {best_test_mae_epoch}, value = {best_test_mae:.3f}")

