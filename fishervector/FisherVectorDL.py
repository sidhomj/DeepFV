import numpy as np
import tensorflow as tf
from sklearn.cluster import MiniBatchKMeans
import pickle
import os

N_Kernel_Choices = [5, 20, 60, 100, 200, 500]


class FisherVectorDL(tf.keras.Model):
    """
    TensorFlow/Keras implementation of Fisher Vector with Gaussian Mixture Model.
    Uses mini-batch training for improved speed on large datasets.
    """

    def __init__(self, n_kernels=1, feature_dim=None, covariance_type='diag'):
        """
        Initialize Fisher Vector GMM model.

        Args:
            n_kernels: Number of Gaussian components
            feature_dim: Dimensionality of input features
            covariance_type: Type of covariance ('diag' only supported currently)
        """
        super(FisherVectorDL, self).__init__()

        assert covariance_type == 'diag', "Only diagonal covariance is currently supported"
        assert n_kernels > 0

        self.n_kernels = n_kernels
        self.feature_dim = feature_dim
        self.covariance_type = covariance_type
        self.fitted = False

        # These will be initialized in build() or initialize()
        self.pi_layer = None
        self.mu_layer = None
        self.sd_layer = None

    def build(self, input_shape):
        """Build the model layers."""
        if self.feature_dim is None:
            self.feature_dim = input_shape[-1]

        # Initialize trainable parameters
        # pi: mixture weights (n_kernels,)
        self.pi_layer = self.add_weight(
            name='pi',
            shape=(self.n_kernels,),
            initializer=tf.keras.initializers.Constant(1.0 / self.n_kernels),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg()
        )

        # mu: means (feature_dim, n_kernels)
        self.mu_layer = self.add_weight(
            name='mu',
            shape=(self.feature_dim, self.n_kernels),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            trainable=True
        )

        # sd: standard deviations (feature_dim, n_kernels)
        self.sd_layer = self.add_weight(
            name='sd',
            shape=(self.feature_dim, self.n_kernels),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg()
        )

        super(FisherVectorDL, self).build(input_shape)

    def initialize(self, X, init='MiniBatchKmeans'):
        """
        Initialize GMM parameters using KMeans clustering.

        Args:
            X: Training data (n_samples, feature_dim)
            init: Initialization method ('MiniBatchKmeans' supported)
        """
        # Build the model first
        self(tf.keras.layers.Input(shape=(X.shape[1],), dtype=tf.float32))

        if init == 'MiniBatchKmeans':
            # MiniBatchKMeans for initialization
            mb_kmeans = MiniBatchKMeans(
                n_clusters=self.n_kernels,
                init='k-means++',
                max_iter=500,
                batch_size=1024 * 6,
                random_state=42
            )
            mb_kmeans.fit(X)

            # Calculate initial GMM parameters from kmeans solution
            mu = mb_kmeans.cluster_centers_.T  # (feature_dim, n_kernels)

            # Assign samples to clusters
            labels = mb_kmeans.predict(X)
            sd = np.eye(self.n_kernels)[labels]  # One-hot encoding
            pi = sd.sum(axis=0, keepdims=True)  # Count per cluster

            # Calculate standard deviation for each cluster
            sd_values = np.zeros((self.feature_dim, self.n_kernels))
            for k in range(self.n_kernels):
                cluster_mask = labels == k
                if cluster_mask.sum() > 1:
                    cluster_data = X[cluster_mask]
                    sd_values[:, k] = np.sqrt(
                        np.sum((cluster_data - mu[:, k])**2, axis=0) / (pi[0, k] - 1)
                    )
                else:
                    sd_values[:, k] = 1.0  # Default value for empty clusters

            # Normalize pi
            pi = pi / pi.sum()

            # Set model weights
            self.set_params(pi, mu, sd_values)

        self.fitted = True
        return self

    def set_params(self, pi, mu, sd):
        """
        Set GMM parameters.

        Args:
            pi: Mixture weights (1, n_kernels) or (n_kernels,)
            mu: Means (feature_dim, n_kernels)
            sd: Standard deviations (feature_dim, n_kernels)
        """
        if pi.ndim == 2:
            pi = pi.flatten()

        self.pi_layer.assign(pi)
        self.mu_layer.assign(mu)
        self.sd_layer.assign(sd)

    def call(self, inputs):
        """
        Forward pass - compute negative log likelihood for training.

        Args:
            inputs: Input features (batch_size, feature_dim)

        Returns:
            Negative log likelihood
        """
        # Normalize pi to ensure it sums to 1
        pi = tf.nn.softmax(self.pi_layer)

        # Add small epsilon to avoid numerical issues
        sd = self.sd_layer + 1e-6

        # Compute Gaussian likelihood for each component
        # inputs: (batch_size, feature_dim)
        # mu: (feature_dim, n_kernels)
        # Expand dimensions for broadcasting
        x_expanded = tf.expand_dims(inputs, axis=-1)  # (batch_size, feature_dim, 1)
        mu_expanded = tf.expand_dims(self.mu_layer, axis=0)  # (1, feature_dim, n_kernels)
        sd_expanded = tf.expand_dims(sd, axis=0)  # (1, feature_dim, n_kernels)

        # Compute normalized squared distance
        diff = (x_expanded - mu_expanded) / sd_expanded  # (batch_size, feature_dim, n_kernels)

        # Log probability for diagonal Gaussian
        log_prob = -0.5 * tf.reduce_sum(diff**2, axis=1)  # (batch_size, n_kernels)
        log_prob = log_prob - tf.reduce_sum(tf.math.log(sd_expanded), axis=1)  # (batch_size, n_kernels)
        log_prob = log_prob - 0.5 * self.feature_dim * tf.math.log(2 * np.pi)

        # Add mixture weights
        log_prob = log_prob + tf.math.log(pi + 1e-10)

        # Log-sum-exp for numerical stability
        log_likelihood = tf.reduce_logsumexp(log_prob, axis=1)

        return -tf.reduce_mean(log_likelihood)

    def fit_minibatch(self, X, epochs=100, batch_size=1024*6, learning_rate=0.001,
                      verbose=True, model_dump_path=None):
        """
        Fit GMM using mini-batch gradient descent.

        Args:
            X: Training data (n_samples, feature_dim) or with higher dimensions
            epochs: Number of training epochs
            batch_size: Mini-batch size
            learning_rate: Learning rate for optimizer
            verbose: Print training progress
            model_dump_path: Path to save fitted model

        Returns:
            self
        """
        # Handle different input dimensions
        if X.ndim == 4:
            self.ndim = 4
            original_shape = X.shape
            X = X.reshape(-1, X.shape[-1])
        elif X.ndim == 3:
            self.ndim = 3
            original_shape = X.shape
            X = np.reshape(X, [1] + list(X.shape))
            X = X.reshape(-1, X.shape[-1])
        else:
            self.ndim = 2
            original_shape = X.shape

        self.feature_dim = X.shape[-1]

        # Initialize if not already done
        if not self.fitted:
            if verbose:
                print(f'Initializing GMM with MiniBatchKMeans ({self.n_kernels} kernels)...')
            self.initialize(X, init='MiniBatchKmeans')

        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
        dataset = dataset.shuffle(buffer_size=min(10000, len(X)))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Training loop
        if verbose:
            print(f'Training GMM with {self.n_kernels} kernels for {epochs} epochs...')

        for epoch in range(epochs):
            epoch_loss = []

            for batch in dataset:
                with tf.GradientTape() as tape:
                    loss = self(batch, training=True)

                gradients = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                epoch_loss.append(loss.numpy())

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                avg_loss = np.mean(epoch_loss)
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

        self.fitted = True

        if model_dump_path:
            self.save_model(model_dump_path)
            if verbose:
                print(f'Saved model to {model_dump_path}')

        return self

    def compute_bic(self, X):
        """
        Compute Bayesian Information Criterion (BIC) for the model.

        Args:
            X: Input data (n_samples, feature_dim) or higher dimensions

        Returns:
            BIC score (lower is better)
        """
        # Handle different input dimensions
        if X.ndim == 4:
            X = X.reshape(-1, X.shape[-1])
        elif X.ndim == 3:
            X = np.reshape(X, [1] + list(X.shape))
            X = X.reshape(-1, X.shape[-1])

        n_samples = X.shape[0]

        # Compute log likelihood
        X_tf = tf.constant(X.astype(np.float32), dtype=tf.float32)
        neg_log_likelihood = self(X_tf, training=False).numpy()
        log_likelihood = -neg_log_likelihood * n_samples

        # Number of parameters:
        # - pi: n_kernels - 1 (constraint: sum to 1)
        # - mu: feature_dim * n_kernels
        # - sd: feature_dim * n_kernels (diagonal covariance)
        n_params = (self.n_kernels - 1) + (self.feature_dim * self.n_kernels * 2)

        # BIC = -2 * log_likelihood + n_params * log(n_samples)
        bic = -2 * log_likelihood + n_params * np.log(n_samples)

        return bic

    def fit_by_bic(self, X, choices_n_kernels=N_Kernel_Choices, epochs=100,
                   batch_size=1024*6, learning_rate=0.001,
                   model_dump_path=None, verbose=True):
        """
        Fit GMM with various n_kernels and select model with lowest BIC.

        Args:
            X: Training data with 3 or 4 dimensions
            choices_n_kernels: List of kernel numbers to try
            epochs: Number of training epochs per model
            batch_size: Mini-batch size
            learning_rate: Learning rate for optimizer
            model_dump_path: Path to save the best model
            verbose: Print training progress

        Returns:
            self (fitted with best n_kernels)
        """
        # Store original n_kernels
        original_n_kernels = self.n_kernels

        # Handle input dimensions
        if X.ndim == 4:
            ndim = 4
        elif X.ndim == 3:
            ndim = 3
            X = np.reshape(X, [1] + list(X.shape))
        else:
            raise AssertionError("X must be an ndarray with 3 or 4 dimensions")

        bic_scores = []
        best_params = None
        best_bic = float('inf')
        best_n_kernels = choices_n_kernels[0]

        for n_kernels in choices_n_kernels:
            # Create new model with different n_kernels
            temp_model = FisherVectorDL(
                n_kernels=n_kernels,
                feature_dim=self.feature_dim,
                covariance_type=self.covariance_type
            )

            # Fit the model
            temp_model.fit_minibatch(
                X,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                verbose=False
            )

            # Compute BIC
            bic_score = temp_model.compute_bic(X)
            bic_scores.append(bic_score)

            if verbose:
                print(f'Fitted GMM with {n_kernels} kernels - BIC = {bic_score:.4f}')

            # Keep track of best model
            if bic_score < best_bic:
                best_bic = bic_score
                best_n_kernels = n_kernels
                best_params = {
                    'pi': temp_model.pi_layer.numpy(),
                    'mu': temp_model.mu_layer.numpy(),
                    'sd': temp_model.sd_layer.numpy()
                }

        # Update current model with best parameters
        self.n_kernels = best_n_kernels
        self.ndim = ndim

        if verbose:
            print(f'Selected GMM with {best_n_kernels} kernels')

        # Rebuild model with best n_kernels
        self.__init__(
            n_kernels=best_n_kernels,
            feature_dim=self.feature_dim,
            covariance_type=self.covariance_type
        )

        # Build and set best parameters
        dummy_input = tf.keras.layers.Input(shape=(self.feature_dim,), dtype=tf.float32)
        self(dummy_input)
        self.set_params(best_params['pi'], best_params['mu'], best_params['sd'])
        self.fitted = True

        if model_dump_path:
            self.save_model(model_dump_path)
            if verbose:
                print(f'Saved model to {model_dump_path}')

        return self

    def predict_fisher_vector(self, X, normalized=True):
        """
        Compute Fisher Vectors for input data.

        Args:
            X: Input features with 3 or 4 dimensions
            normalized: Apply improved Fisher Vector normalization

        Returns:
            Fisher vectors
        """
        if X.ndim == 4:
            return self._predict(X, normalized=normalized)
        elif X.ndim == 3:
            orig_shape = X.shape
            X = np.reshape(X, [1] + list(X.shape))
            result = self._predict(X, normalized=normalized)
            return np.reshape(result, (orig_shape[0], 2 * self.n_kernels, orig_shape[-1]))
        else:
            raise AssertionError("X must be an ndarray with 3 or 4 dimensions")

    def _predict(self, X, normalized=True):
        """
        Internal method to compute Fisher Vectors.

        Args:
            X: Input features (n_videos, n_frames, n_features, feature_dim)
            normalized: Apply improved Fisher Vector normalization

        Returns:
            Fisher vectors (n_videos, n_frames, 2*n_kernels, feature_dim)
        """
        assert self.fitted, "Model must be fitted first"
        assert X.ndim == 4
        assert X.shape[-1] == self.feature_dim

        n_videos, n_frames = X.shape[0], X.shape[1]

        # Reshape for processing
        X = X.reshape((-1, X.shape[-2], X.shape[-1]))  # (n_images, n_features, feature_dim)
        X_matrix = X.reshape(-1, X.shape[-1])  # (n_images * n_features, feature_dim)

        # Get GMM parameters
        pi = tf.nn.softmax(self.pi_layer).numpy()
        mu = self.mu_layer.numpy()  # (feature_dim, n_kernels)
        sd = (self.sd_layer + 1e-6).numpy()  # (feature_dim, n_kernels)

        # Compute posterior probabilities (responsibilities)
        # Use uniform weights for likelihood ratio as in original implementation
        equal_weights = np.ones(self.n_kernels) / self.n_kernels

        # Compute log probabilities
        X_tf = tf.constant(X_matrix, dtype=tf.float32)
        x_expanded = tf.expand_dims(X_tf, axis=-1)  # (n_samples, feature_dim, 1)
        mu_expanded = tf.expand_dims(mu, axis=0)  # (1, feature_dim, n_kernels)
        sd_expanded = tf.expand_dims(sd, axis=0)  # (1, feature_dim, n_kernels)

        diff = (x_expanded - mu_expanded) / sd_expanded
        log_prob = -0.5 * tf.reduce_sum(diff**2, axis=1)
        log_prob = log_prob - tf.reduce_sum(tf.math.log(sd_expanded), axis=1)
        log_prob = log_prob - 0.5 * self.feature_dim * tf.math.log(2 * np.pi)
        log_prob = log_prob + tf.math.log(equal_weights + 1e-10)

        # Convert to probabilities
        log_prob_normalized = log_prob - tf.reduce_logsumexp(log_prob, axis=1, keepdims=True)
        likelihood_ratio = tf.exp(log_prob_normalized).numpy()
        likelihood_ratio = likelihood_ratio.reshape(X.shape[0], X.shape[1], self.n_kernels)

        # Compute normalized deviation from modes
        var = sd ** 2
        norm_dev_from_modes = np.tile(X[:, :, None, :], (1, 1, self.n_kernels, 1))
        np.subtract(norm_dev_from_modes, mu.T[None, None, :, :], out=norm_dev_from_modes)
        np.divide(norm_dev_from_modes, var.T[None, None, :, :], out=norm_dev_from_modes)

        # Mean deviation
        mean_dev = np.multiply(
            likelihood_ratio[:, :, :, None],
            norm_dev_from_modes
        ).mean(axis=1)  # (n_images, n_kernels, feature_dim)
        mean_dev = np.multiply(1 / np.sqrt(pi[None, :, None]), mean_dev)

        # Covariance deviation
        cov_dev = np.multiply(
            likelihood_ratio[:, :, :, None],
            norm_dev_from_modes**2 - 1
        ).mean(axis=1)
        cov_dev = np.multiply(1 / np.sqrt(2 * pi[None, :, None]), cov_dev)

        # Concatenate mean and covariance deviations
        fisher_vectors = np.concatenate([mean_dev, cov_dev], axis=1)

        # Reshape to separate videos and frames
        fisher_vectors = fisher_vectors.reshape(
            (n_videos, n_frames, fisher_vectors.shape[1], fisher_vectors.shape[2])
        )

        # Apply normalization if requested
        if normalized:
            # Power normalization
            fisher_vectors = np.sqrt(np.abs(fisher_vectors)) * np.sign(fisher_vectors)
            # L2 normalization
            norms = np.linalg.norm(fisher_vectors, axis=(2, 3))[:, :, None, None]
            fisher_vectors = fisher_vectors / (norms + 1e-10)

        # Threshold small values
        fisher_vectors[fisher_vectors < 1e-4] = 0

        return fisher_vectors

    def save_model(self, path):
        """Save model parameters to file."""
        params = {
            'n_kernels': self.n_kernels,
            'feature_dim': self.feature_dim,
            'covariance_type': self.covariance_type,
            'pi': self.pi_layer.numpy(),
            'mu': self.mu_layer.numpy(),
            'sd': self.sd_layer.numpy(),
            'fitted': self.fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(params, f, protocol=4)

    @staticmethod
    def load_model(path):
        """Load model parameters from file."""
        assert os.path.isfile(path), 'Path must be an existing file'

        with open(path, 'rb') as f:
            params = pickle.load(f)

        model = FisherVectorDL(
            n_kernels=params['n_kernels'],
            feature_dim=params['feature_dim'],
            covariance_type=params['covariance_type']
        )

        # Build model
        dummy_input = tf.keras.layers.Input(shape=(params['feature_dim'],), dtype=tf.float32)
        model(dummy_input)

        # Set parameters
        model.set_params(params['pi'], params['mu'], params['sd'])
        model.fitted = params['fitted']

        return model
