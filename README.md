[![Build Status](https://travis-ci.org/jonasrothfuss/fishervector.svg?branch=master)](https://travis-ci.org/jonasrothfuss/fishervector)

# DeepFV - Fisher Vectors with Deep Learning

This package implements Improved Fisher Vectors as described in [1], with both traditional scikit-learn and modern TensorFlow implementations. For a more concise description of Fisher Vectors see [2].

## Features

### Classic Implementation (FisherVectorGMM)
- Fitting a Gaussian Mixture Model (GMM) using scikit-learn
- Determining the number of GMM components via BIC
- Saving and loading the fitted GMM
- Computing the (Improved) Fisher Vectors based on the fitted GMM
- Supports diagonal covariance

### Deep Learning Implementation (FisherVectorDL) - **NEW!**
- TensorFlow/Keras-based GMM with mini-batch gradient descent training
- **Supports both diagonal and full covariance matrices**
- Scalable to large datasets with mini-batch training
- BIC-based model selection
- Fisher Vector computation compatible with the classic implementation
- GPU acceleration support via TensorFlow

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

Or install directly:
```bash
pip install numpy scipy scikit-learn tensorflow matplotlib
```

## Quick Start

### Classic Implementation (FisherVectorGMM)

##### 1. Prepare your data
```python
import numpy as np
shape = [300, 20, 32]  # e.g. SIFT image features
image_data = np.concatenate([
    np.random.normal(-np.ones(30), size=shape),
    np.random.normal(np.ones(30), size=shape)
], axis=0)
```

##### 2. Train the GMM
```python
from fishervector import FisherVectorGMM

# Fixed number of components
fv_gmm = FisherVectorGMM(n_kernels=10).fit(image_data)

# Or use BIC to select optimal number of components
fv_gmm = FisherVectorGMM().fit_by_bic(image_data, choices_n_kernels=[2,5,10,20])
```

##### 3. Compute Fisher Vectors
```python
image_data_test = image_data[:20]
fv = fv_gmm.predict(image_data_test)
```

### Deep Learning Implementation (FisherVectorDL)

##### 1. Train with mini-batch gradient descent
```python
from fishervector import FisherVectorDL

# Create model with FULL covariance support
fv_dl = FisherVectorDL(
    n_kernels=10,
    feature_dim=32,
    covariance_type='full'  # or 'diag'
)

# Fit with mini-batch training
fv_dl.fit_minibatch(
    image_data,
    epochs=100,
    batch_size=1024*6,
    learning_rate=0.001
)
```

##### 2. BIC-based model selection
```python
fv_dl = FisherVectorDL(covariance_type='full')
fv_dl.fit_by_bic(
    image_data,
    choices_n_kernels=[2, 5, 10, 20],
    epochs=80,
    batch_size=1024
)
```

##### 3. Compute Fisher Vectors
```python
fisher_vectors = fv_dl.predict_fisher_vector(image_data_test, normalized=True)
```

##### 4. Save and load models
```python
# Save
fv_dl.save_model('my_model.pkl')

# Load
fv_dl_loaded = FisherVectorDL.load_model('my_model.pkl')
```

## Key Advantages of FisherVectorDL

1. **Full Covariance Support**: Can model rotated/tilted elliptical clusters, not just axis-aligned ones
2. **Scalability**: Mini-batch training handles datasets too large for memory
3. **Speed**: GPU acceleration via TensorFlow for faster training
4. **Flexibility**: Customizable learning rate, batch size, and epochs
5. **Modern Stack**: Built on TensorFlow 2.x with eager execution

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

Contributors:
* Jonas Rothfuss (https://github.com/jonasrothfuss/)
* Fabio Ferreira (https://github.com/ferreirafabio/)

References:
- [1] https://www.robots.ox.ac.uk/~vgg/rg/papers/peronnin_etal_ECCV10.pdf
- [2] http://www.vlfeat.org/api/fisher-fundamentals.html
