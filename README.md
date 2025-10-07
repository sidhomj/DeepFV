
# DeepFV - Fisher Vectors with Deep Learning

A TensorFlow-based implementation of Improved Fisher Vectors as described in [1]. This package provides a modern, scalable approach to computing Fisher Vectors using deep learning techniques. For a concise description of Fisher Vectors see [2].

## Features

- **Full & Diagonal Covariance Support**: Model complex elliptical clusters with full covariance matrices, or use diagonal covariance for faster training
- **Mini-batch Training**: Scalable to large datasets with mini-batch gradient descent
- **BIC-based Model Selection**: Automatically determine optimal number of GMM components
- **GPU Acceleration**: Built on TensorFlow 2.x for fast training on GPUs
- **MiniBatchKMeans Initialization**: Smart initialization using scikit-learn's MiniBatchKMeans
- **Save/Load Models**: Persist trained models for reuse
- **Normalized Fisher Vectors**: Implements improved Fisher Vector normalization

## Installation

Install from PyPI:
```bash
pip install DeepFV
```

Or install from source:
```bash
git clone https://github.com/sidhomj/DeepFV.git
cd DeepFV
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Prepare your data
```python
import numpy as np

# Example: SIFT image features
shape = [300, 20, 32]  # (n_images, n_features_per_image, feature_dim)
image_data = np.concatenate([
    np.random.normal(-np.ones(30), size=shape),
    np.random.normal(np.ones(30), size=shape)
], axis=0)
```

### 2. Train with mini-batch gradient descent
```python
from DeepFV import FisherVectorDL

# Create model with FULL covariance support
fv_dl = FisherVectorDL(
    n_kernels=10,
    feature_dim=32,
    covariance_type='full'  # or 'diag' for diagonal covariance
)

# Fit with mini-batch training
fv_dl.fit_minibatch(
    image_data,
    epochs=100,
    batch_size=1024*6,
    learning_rate=0.001,
    verbose=True
)
```

### 3. BIC-based model selection
```python
# Automatically select optimal number of components
fv_dl = FisherVectorDL(feature_dim=32, covariance_type='full')
fv_dl.fit_by_bic(
    image_data,
    choices_n_kernels=[2, 5, 10, 20],
    epochs=80,
    batch_size=1024,
    verbose=True
)

print(f"Selected {fv_dl.n_kernels} components")
```

### 4. Compute Fisher Vectors
```python
# Compute normalized Fisher Vectors
image_data_test = image_data[:20]
fisher_vectors = fv_dl.predict_fisher_vector(image_data_test, normalized=True)

# Output shape: (n_images, 2*n_kernels, feature_dim)
print(f"Fisher vector shape: {fisher_vectors.shape}")
```

### 5. Save and load models
```python
# Save trained model
fv_dl.save_model('my_model.pkl')

# Load model later
from DeepFV import FisherVectorDL
fv_dl_loaded = FisherVectorDL.load_model('my_model.pkl')
```

## Why FisherVectorDL?

### Advantages over traditional GMM implementations:

1. **Full Covariance Support**: Model rotated/tilted elliptical clusters, not just axis-aligned ones
2. **Scalability**: Mini-batch training handles datasets too large to fit in memory
3. **Speed**: GPU acceleration via TensorFlow for faster training
4. **Flexibility**: Customizable learning rate, batch size, and number of epochs
5. **Modern Stack**: Built on TensorFlow 2.x with eager execution
6. **Smart Initialization**: Uses MiniBatchKMeans for better starting parameters

## Testing

Run the test script to see a 2D visualization:
```bash
python test_fishervector_dl.py
```

This will:
- Generate 3 elliptical Gaussian clusters
- Train a GMM with full covariance
- Use BIC to select optimal number of components
- Compute and visualize Fisher Vectors
- Save visualizations as PNG files

## Contributors

* **John-William Sidhom** (https://github.com/sidhomj/) - Main contributor, TensorFlow implementation with full covariance support

### Original Contributors:
* Jonas Rothfuss (https://github.com/jonasrothfuss/) - Original implementation
* Fabio Ferreira (https://github.com/ferreirafabio/) - Original implementation

## References

- [1] Perronnin, F., Sánchez, J., & Mensink, T. (2010). Improving the fisher kernel for large-scale image classification. In European conference on computer vision (pp. 143-156). Springer, Berlin, Heidelberg. https://www.robots.ox.ac.uk/~vgg/rg/papers/peronnin_etal_ECCV10.pdf
- [2] Fisher Vector Fundamentals - VLFeat Documentation: http://www.vlfeat.org/api/fisher-fundamentals.html

## Development

### Building and Publishing

1. **Build the package:**
   ```bash
   ./build_package.sh
   ```

2. **Test on TestPyPI (recommended):**
   ```bash
   ./publish_to_testpypi.sh
   ```

   Then test installation:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple DeepFV
   ```

3. **Publish to PyPI:**
   ```bash
   ./publish_to_pypi.sh
   ```

### PyPI Credentials Setup

Before publishing, you need PyPI credentials:

1. Create accounts on [PyPI](https://pypi.org/account/register/) and [TestPyPI](https://test.pypi.org/account/register/)
2. Create API tokens:
   - PyPI: https://pypi.org/manage/account/token/
   - TestPyPI: https://test.pypi.org/manage/account/token/
3. Create `~/.pypirc`:
   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = pypi-your-api-token-here

   [testpypi]
   username = __token__
   password = pypi-your-testpypi-token-here
   ```

## License

MIT License - see LICENSE file for details
