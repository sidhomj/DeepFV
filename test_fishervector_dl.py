import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fishervector import FisherVectorDL
import matplotlib
matplotlib.use("MacOSX", force=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("Testing FisherVectorDL with 2D Visualizable Data")
print("=" * 60)

# Parameters
n_mixtures = 3  # True number of Gaussian components in data
n_samples_per_mixture = 500
feature_dim = 2  # 2D for visualization

# Generate ground truth data from 3 Gaussian mixtures with different covariances (non-spherical)
print(f"\nGenerating data from {n_mixtures} Gaussian mixtures with elliptical shapes...")
true_means = [
    np.array([-3.0, -3.0]),
    np.array([0.0, 4.0]),
    np.array([3.0, -2.0])
]

# Define covariance matrices for elliptical clusters
true_covs = [
    np.array([[2.0, 1.5],    # Elongated diagonal cluster
              [1.5, 0.5]]),
    np.array([[0.4, 0.0],    # Vertical ellipse
              [0.0, 1.8]]),
    np.array([[1.5, -1.0],   # Tilted ellipse
              [-1.0, 1.2]])
]
colors = ['red', 'blue', 'green']

# Generate samples from each mixture with non-spherical covariances
data_list = []
labels = []
for i in range(n_mixtures):
    samples = np.random.multivariate_normal(true_means[i], true_covs[i], size=n_samples_per_mixture)
    data_list.append(samples)
    labels.extend([i] * n_samples_per_mixture)

# Combine all data
X = np.vstack(data_list).astype(np.float32)
labels = np.array(labels)

print(f"✓ Generated {X.shape[0]} samples with {X.shape[1]} features")
print(f"  True mixture means: {[list(m) for m in true_means]}")

# Visualize the generated data
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
for i in range(n_mixtures):
    mask = labels == i
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.5, s=10, label=f'Mixture {i+1}')
plt.title('Ground Truth Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

# Test 1: Fit with correct number of kernels using FULL covariance
print(f"\n[Test 1] Fitting GMM with {n_mixtures} kernels (FULL covariance)")
print("-" * 60)

fv_dl = FisherVectorDL(n_kernels=n_mixtures, feature_dim=feature_dim, covariance_type='full')
fv_dl.fit_minibatch(X, epochs=100, batch_size=256, learning_rate=0.01, verbose=True)

# Get learned parameters
learned_means = fv_dl.mu_layer.numpy().T  # (n_kernels, feature_dim)
learned_pi = tf.nn.softmax(fv_dl.pi_layer).numpy()

# Get covariance matrices
L = fv_dl.cov_layer.numpy()  # Cholesky factors
learned_covs = np.array([L[k] @ L[k].T for k in range(n_mixtures)])

print(f"\n✓ Learned mixture weights: {learned_pi}")
print(f"✓ Learned means:\n{learned_means}")
print(f"✓ Learned covariance matrices (showing first component):\n{learned_covs[0]}")

# Visualize learned clusters with ellipses
plt.subplot(1, 3, 2)
for i in range(n_mixtures):
    mask = labels == i
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.3, s=10)

# Plot learned means and covariance ellipses
from matplotlib.patches import Ellipse

for i in range(n_mixtures):
    # Plot mean
    plt.scatter(learned_means[i, 0], learned_means[i, 1],
               marker='x', s=200, c='black', linewidths=3,
               label=f'Learned μ{i+1}')

    # Plot covariance ellipse (2 standard deviations)
    eigenvalues, eigenvectors = np.linalg.eigh(learned_covs[i])
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    width, height = 2 * 2 * np.sqrt(eigenvalues)  # 2 std devs
    ellipse = Ellipse(learned_means[i], width, height, angle=angle,
                     facecolor='none', edgecolor='black', linewidth=2, linestyle='--')
    plt.gca().add_patch(ellipse)

plt.title(f'Learned GMM (Full Cov, {n_mixtures} kernels)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)


# Test 2: BIC-based model selection with FULL covariance
print(f"\n[Test 2] BIC-based model selection (FULL covariance)")
print("-" * 60)

fv_dl_bic = FisherVectorDL(feature_dim=feature_dim, covariance_type='full')
fv_dl_bic.fit_by_bic(
    X.reshape(1, -1, feature_dim),  # Reshape to 3D (1, n_samples, feature_dim)
    choices_n_kernels=[2, 3, 4, 5],
    epochs=80,
    batch_size=256,
    learning_rate=0.01,
    verbose=True
)

print(f"\n✓ BIC selected {fv_dl_bic.n_kernels} kernels (true: {n_mixtures})")

# Visualize BIC-selected model
plt.subplot(1, 3, 3)
for i in range(n_mixtures):
    mask = labels == i
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.3, s=10)

# Plot BIC-selected means and ellipses
bic_means = fv_dl_bic.mu_layer.numpy().T
bic_pi = tf.nn.softmax(fv_dl_bic.pi_layer).numpy()

# Get BIC covariances
L_bic = fv_dl_bic.cov_layer.numpy()
bic_covs = np.array([L_bic[k] @ L_bic[k].T for k in range(fv_dl_bic.n_kernels)])

for i in range(fv_dl_bic.n_kernels):
    # Plot mean
    plt.scatter(bic_means[i, 0], bic_means[i, 1],
               marker='*', s=300, c='orange', edgecolors='black', linewidths=2,
               label=f'BIC μ{i+1} (π={bic_pi[i]:.2f})')

    # Plot covariance ellipse
    eigenvalues, eigenvectors = np.linalg.eigh(bic_covs[i])
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    width, height = 2 * 2 * np.sqrt(eigenvalues)
    ellipse = Ellipse(bic_means[i], width, height, angle=angle,
                     facecolor='none', edgecolor='orange', linewidth=2, linestyle='-.')
    plt.gca().add_patch(ellipse)

plt.title(f'BIC Selection (Full Cov, {fv_dl_bic.n_kernels} kernels)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fishervector_dl_test.png', dpi=150)
print(f"\n✓ Visualization saved to 'fishervector_dl_test.png'")
plt.show()

print("\n" + "=" * 60)
print("Test completed successfully! ✓")
print("=" * 60)
