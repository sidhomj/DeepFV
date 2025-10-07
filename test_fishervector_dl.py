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

# Visualize learned clusters
plt.subplot(1, 3, 2)
for i in range(n_mixtures):
    mask = labels == i
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.3, s=10)

# Plot learned means
for i in range(n_mixtures):
    plt.scatter(learned_means[i, 0], learned_means[i, 1],
               marker='x', s=200, c='black', linewidths=3,
               label=f'Learned μ{i+1}')

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

# Plot BIC-selected means
bic_means = fv_dl_bic.mu_layer.numpy().T
bic_pi = tf.nn.softmax(fv_dl_bic.pi_layer).numpy()

for i in range(fv_dl_bic.n_kernels):
    plt.scatter(bic_means[i, 0], bic_means[i, 1],
               marker='*', s=300, c='orange', edgecolors='black', linewidths=2,
               label=f'BIC μ{i+1} (π={bic_pi[i]:.2f})')

plt.title(f'BIC Selection (Full Cov, {fv_dl_bic.n_kernels} kernels)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fishervector_dl_test.png', dpi=150)
print(f"\n✓ Visualization saved to 'fishervector_dl_test.png'")
plt.show()


# Test 3: Compute and visualize Fisher Vectors
print("\n" + "=" * 60)
print("[Test 3] Computing and visualizing Fisher Vectors")
print("=" * 60)

# Sample evenly from each cluster to ensure all clusters are represented
n_samples_per_cluster = 50
sample_indices = []
for i in range(n_mixtures):
    cluster_indices = np.where(labels == i)[0]
    sampled = np.random.choice(cluster_indices, size=min(n_samples_per_cluster, len(cluster_indices)), replace=False)
    sample_indices.extend(sampled)

sample_indices = np.array(sample_indices)
X_sampled = X[sample_indices]
labels_sampled = labels[sample_indices]

print(f"Sampled {len(sample_indices)} points ({n_samples_per_cluster} per cluster)")
print(f"  Cluster distribution: {[np.sum(labels_sampled == i) for i in range(n_mixtures)]}")

# Reshape data to 3D for Fisher Vector computation (n_images, n_features_per_image, feature_dim)
# Let's treat each sample as an "image" with 1 "feature"
X_for_fv = X_sampled.reshape(len(sample_indices), 1, feature_dim)

# Compute Fisher Vectors
print(f"\nComputing Fisher Vectors for {X_for_fv.shape[0]} samples...")
fisher_vectors = fv_dl.predict_fisher_vector(X_for_fv, normalized=True)
print(f"✓ Fisher vector shape: {fisher_vectors.shape}")
print(f"  Expected: ({X_for_fv.shape[0]}, {2 * n_mixtures}, {feature_dim})")

# Flatten Fisher vectors for visualization
fv_flattened = fisher_vectors.reshape(fisher_vectors.shape[0], -1)  # (n_samples, 2*n_kernels*feature_dim)
print(f"✓ Flattened Fisher vector shape: {fv_flattened.shape}")

# Visualize Fisher Vectors using PCA
from sklearn.decomposition import PCA

print("\nReducing Fisher Vector dimensionality with PCA for visualization...")
pca = PCA(n_components=2)
fv_2d = pca.fit_transform(fv_flattened)
print(f"✓ PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# Plot Fisher Vectors in 2D space
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
# Color by true labels
for i in range(n_mixtures):
    mask = labels_sampled == i
    plt.scatter(fv_2d[mask, 0], fv_2d[mask, 1], c=colors[i], alpha=0.6, s=30, label=f'Cluster {i+1}')
plt.title('Fisher Vectors (colored by true cluster)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Color by original features
plt.scatter(fv_2d[:, 0], fv_2d[:, 1], c=X_sampled[:, 0], cmap='viridis', alpha=0.6, s=30)
plt.colorbar(label='Original Feature 1 Value')
plt.title('Fisher Vectors (colored by feature value)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fishervector_visualization.png', dpi=150)
print(f"\n✓ Fisher Vector visualization saved to 'fishervector_visualization.png'")
plt.show()

print("\n" + "=" * 60)
print("All tests completed successfully! ✓")
print("=" * 60)
