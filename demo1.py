import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Perceptron import Perceptron

# =====================================================================
# STEP 1: Load Data (Versicolor vs. Virginica)
# =====================================================================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
raw_data = urllib.request.urlopen(url).read().decode('utf-8').strip().split('\n')

features, labels = [], []
for line in raw_data:
    if not line:
        continue
    parts = line.split(',')
    species = parts[4]
    
    if species == 'Iris-versicolor':
        features.append([float(parts[2]), float(parts[3])])
        labels.append(-1)
    elif species == 'Iris-virginica':
        features.append([float(parts[2]), float(parts[3])])
        labels.append(1)

features = np.array(features)
labels = np.array(labels)

# Feature Standardization
features[:, 0] = (features[:, 0] - features[:, 0].mean()) / features[:, 0].std()
features[:, 1] = (features[:, 1] - features[:, 1].mean()) / features[:, 1].std()


# =====================================================================
# STEP 1: Load Data (Versicolor vs. Virginica)
# =====================================================================
model = Perceptron(2, learning_rate=0.01, epochs=20)
model.train(features, labels)

# =====================================================================
# STEP 3: Setup Matplotlib Plot and Animation
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 6))

# Static Data Points (Plotted once)
ax.scatter(features[labels == -1][:, 0], features[labels == -1][:, 1], color='blue', label='Versicolor (-1)', edgecolor='k', s=50)
ax.scatter(features[labels == 1][:, 0], features[labels == 1][:, 1], color='green', label='Virginica (+1)', edgecolor='k', s=50)

# Create an empty line object that FuncAnimation will update frame-by-frame
line, = ax.plot([], [], color='black', linestyle='--', linewidth=2, label='Decision Boundary')

x1_min, x1_max = features[:, 0].min() - 0.5, features[:, 0].max() + 0.5
x1_line = np.linspace(x1_min, x1_max, 100)

ax.set_xlim(x1_min, x1_max)
ax.set_ylim(features[:, 1].min() - 0.5, features[:, 1].max() + 0.5)
ax.set_xlabel("Petal Length (Standardized)")
ax.set_ylabel("Petal Width (Standardized)")
ax.legend(loc="upper left")
ax.grid(True, linestyle=':', alpha=0.6)

def update(frame):
    """Update function called automatically by FuncAnimation for each epoch."""
    weights, bias, epoch, errors = model.history[frame]
    
    if abs(weights[1]) > 1e-5:
        x2_line = -(weights[0] * x1_line + bias) / weights[1]
        line.set_data(x1_line, x2_line)  # Efficiently update line position
    
    ax.set_title(f"Epoch {epoch} | Weights: [{weights[0]:.3f}, {weights[1]:.3f}] | Bias: {bias:.3f}")
    return line,

# FuncAnimation handles the frame updates seamlessly
# interval=800 means 800 milliseconds (0.8 seconds) per epoch
ani = FuncAnimation(
    fig, 
    update, 
    frames=len(model.history), 
    interval=800, 
    repeat=False
)

plt.show()