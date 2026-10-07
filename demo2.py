import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from NeuralNetwork import NeuralNetwork

# =====================================================================
# STEP 1: Load All 3 Iris Species
# =====================================================================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
raw_data = urllib.request.urlopen(url).read().decode('utf-8').strip().split('\n')

features, labels = [], []
species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

for line in raw_data:
    if not line:
        continue
    parts = line.split(',')
    species = parts[4]
    if species in species_map:
        # Petal Length (index 2) and Petal Width (index 3)
        features.append([float(parts[2]), float(parts[3])])
        labels.append(species_map[species])

features = np.array(features)
labels = np.array(labels)

# Feature Standardization (mean=0, std=1)
features[:, 0] = (features[:, 0] - features[:, 0].mean()) / features[:, 0].std()
features[:, 1] = (features[:, 1] - features[:, 1].mean()) / features[:, 1].std()

# One-Hot Encode Labels for 3 Output Neurons
def one_hot(labels, num_classes=3):
    one_hot_labels = np.zeros((len(labels), num_classes))
    one_hot_labels[np.arange(len(labels)), labels] = 1.0
    return one_hot_labels

labels_encoded = one_hot(labels, 3)

# =====================================================================
# STEP 2: Multi-Layer Neural Network Built From Scratch
# play with the hiddent dimensions, learning rate, and epochs to see
# how it changes the model
# =====================================================================

nn = NeuralNetwork(input_dim=2, hidden_dim=4, output_dim=3, learning_rate=0.08)
nn.train(features, labels_encoded, labels, epochs=300)

# =====================================================================
# STEP 3: Animated Decision Boundary Regions
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 7))

# Create 2D Meshgrid for decision contour plotting
x_min, x_max = features[:, 0].min() - 0.5, features[:, 0].max() + 0.5
y_min, y_max = features[:, 1].min() - 0.5, features[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Plot Data Points
colors = ['red', 'blue', 'green']
species_names = ['Setosa (0)', 'Versicolor (1)', 'Virginica (2)']

for class_val, color, name in zip([0, 1, 2], colors, species_names):
    ax.scatter(features[labels == class_val][:, 0], features[labels == class_val][:, 1], 
               c=color, label=name, edgecolor='k', s=50, zorder=3)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("Petal Length (Standardized)")
ax.set_ylabel("Petal Width (Standardized)")
ax.legend(loc="upper left")
ax.grid(True, linestyle=':', alpha=0.5)

# Function to compute predictions for the grid given specific weights
def predict_grid(grid, W1, b1, W2, b2):
    Z1 = np.dot(grid, W1) + b1
    A1 = np.maximum(0, Z1)
    Z2 = np.dot(A1, W2) + b2
    exp_Z = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
    probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    return np.argmax(probs, axis=1).reshape(xx.shape)

# Draw initial background contour mesh
initial_W1, initial_b1, initial_W2, initial_b2, _, _ = nn.history[0]
Z_grid = predict_grid(grid_points, initial_W1, initial_b1, initial_W2, initial_b2)
contour = ax.contourf(xx, yy, Z_grid, alpha=0.3, levels=[-0.5, 0.5, 1.5, 2.5], colors=colors)

def update(frame):
    global contour
    W1, b1, W2, b2, epoch, errors = nn.history[frame]
    
    # Cleanly remove the previous epoch's contour fill (Matplotlib 3.8+)
    contour.remove()
        
    # Re-calculate and draw the updated decision regions
    Z_grid = predict_grid(grid_points, W1, b1, W2, b2)
    contour = ax.contourf(xx, yy, Z_grid, alpha=0.3, levels=[-0.5, 0.5, 1.5, 2.5], colors=colors)
    
    accuracy = ((150 - errors) / 150) * 100
    ax.set_title(f"Class 2: Neural Network | Epoch {epoch}\nErrors: {errors}/150 | Accuracy: {accuracy:.1f}%")
    return contour.collections if hasattr(contour, 'collections') else []

ani = FuncAnimation(
    fig, 
    update, 
    frames=len(nn.history), 
    interval=300, 
    repeat=False
)

plt.show()