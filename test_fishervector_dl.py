import numpy as np
import tensorflow as tf
from fishervector import FisherVectorDL

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("Testing FisherVectorDL with TensorFlow")
print("=" * 60)

# Test 1: Basic fitting and prediction with 3D data (images)
print("\n[Test 1] Basic fitting and prediction (3D data)")
print("-" * 60)
shape_3d = [200, 10, 30]  # (n_images, n_features, feature_dim)
test_data_3d = np.concatenate([
    np.random.normal(np.zeros(30), size=shape_3d),
    np.random.normal(np.ones(30), size=shape_3d)
], axis=0)

n_kernels = 2
fv_dl = FisherVectorDL(n_kernels=n_kernels)
print(f"Fitting GMM with {n_kernels} kernels...")
fv_dl.fit_minibatch(test_data_3d, epochs=50, batch_size=1024, verbose=True)

# Predict Fisher Vectors
n_test_images = 20
fv = fv_dl.predict_fisher_vector(test_data_3d[:n_test_images])
print(f"✓ Fisher vector shape: {fv.shape}")
print(f"  Expected: ({n_test_images}, {2 * n_kernels}, {30})")
assert fv.shape == (n_test_images, 2 * n_kernels, 30), "Shape mismatch for 3D data"
print("✓ Test 1 passed!\n")


# Test 2: Fitting and prediction with 4D data (videos)
print("\n[Test 2] Fitting and prediction (4D data)")
print("-" * 60)
shape_4d = [100, 15, 10, 30]  # (n_videos, n_frames, n_features, feature_dim)
test_data_4d = np.concatenate([
    np.random.normal(np.zeros(30), size=shape_4d),
    np.random.normal(np.ones(30), size=shape_4d)
], axis=0)

n_kernels = 3
fv_dl_4d = FisherVectorDL(n_kernels=n_kernels)
print(f"Fitting GMM with {n_kernels} kernels...")
fv_dl_4d.fit_minibatch(test_data_4d, epochs=50, batch_size=2048, verbose=True)

# Predict Fisher Vectors
n_test_videos = 10
fv_4d = fv_dl_4d.predict_fisher_vector(test_data_4d[:n_test_videos])
print(f"✓ Fisher vector shape: {fv_4d.shape}")
print(f"  Expected: ({n_test_videos}, 15, {2 * n_kernels}, 30)")
assert fv_4d.shape == (n_test_videos, 15, 2 * n_kernels, 30), "Shape mismatch for 4D data"
print("✓ Test 2 passed!\n")


# Test 3: BIC-based model selection
print("\n[Test 3] BIC-based model selection")
print("-" * 60)
shape_bic = [150, 10, 30]
test_data_bic = np.concatenate([
    np.random.normal(-np.ones(30), size=shape_bic),
    np.random.normal(np.ones(30), size=shape_bic)
], axis=0)

fv_dl_bic = FisherVectorDL()
print("Testing multiple kernel counts with BIC selection...")
fv_dl_bic.fit_by_bic(
    test_data_bic,
    choices_n_kernels=[2, 5, 10],
    epochs=30,
    batch_size=1024,
    verbose=True
)

print(f"✓ Selected n_kernels: {fv_dl_bic.n_kernels}")

# Test prediction with selected model
n_test = 10
fv_bic = fv_dl_bic.predict_fisher_vector(test_data_bic[:n_test])
print(f"✓ Fisher vector shape: {fv_bic.shape}")
print(f"  Expected: ({n_test}, {2 * fv_dl_bic.n_kernels}, 30)")
assert fv_bic.shape == (n_test, 2 * fv_dl_bic.n_kernels, 30), "Shape mismatch for BIC model"
print("✓ Test 3 passed!\n")


# Test 4: Save and load model
print("\n[Test 4] Save and load model")
print("-" * 60)
save_path = "./test_fv_dl_model.pkl"

# Create and fit a model
fv_dl_save = FisherVectorDL(n_kernels=2)
small_data = test_data_3d[:50]
fv_dl_save.fit_minibatch(small_data, epochs=20, batch_size=512, verbose=False)

# Save model
fv_dl_save.save_model(save_path)
print(f"✓ Model saved to {save_path}")

# Load model
fv_dl_load = FisherVectorDL.load_model(save_path)
print(f"✓ Model loaded from {save_path}")

# Compare parameters
pi_match = np.allclose(fv_dl_save.pi_layer.numpy(), fv_dl_load.pi_layer.numpy())
mu_match = np.allclose(fv_dl_save.mu_layer.numpy(), fv_dl_load.mu_layer.numpy())
sd_match = np.allclose(fv_dl_save.sd_layer.numpy(), fv_dl_load.sd_layer.numpy())

print(f"✓ Parameters match: pi={pi_match}, mu={mu_match}, sd={sd_match}")
assert pi_match and mu_match and sd_match, "Loaded parameters don't match saved model"

# Test prediction with loaded model
test_samples = small_data[:5]
fv_original = fv_dl_save.predict_fisher_vector(test_samples)
fv_loaded = fv_dl_load.predict_fisher_vector(test_samples)
prediction_match = np.allclose(fv_original, fv_loaded, rtol=1e-5)
print(f"✓ Predictions match: {prediction_match}")
assert prediction_match, "Predictions from loaded model don't match"

# Clean up
import os
os.remove(save_path)
print(f"✓ Cleaned up test file")
print("✓ Test 4 passed!\n")


# Test 5: Initialize method
print("\n[Test 5] Initialize method with MiniBatchKMeans")
print("-" * 60)
init_data = test_data_3d[:100].reshape(-1, 30)
fv_dl_init = FisherVectorDL(n_kernels=4, feature_dim=30)
fv_dl_init.initialize(init_data, init='MiniBatchKmeans')
print(f"✓ Model initialized with {fv_dl_init.n_kernels} kernels")
print(f"✓ Parameters shape: pi={fv_dl_init.pi_layer.shape}, mu={fv_dl_init.mu_layer.shape}, sd={fv_dl_init.sd_layer.shape}")
assert fv_dl_init.fitted, "Model should be marked as fitted after initialization"
print("✓ Test 5 passed!\n")


print("=" * 60)
print("All tests passed! ✓")
print("=" * 60)
