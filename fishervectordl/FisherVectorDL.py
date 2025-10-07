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
            covariance_type: Type of covariance ('diag' or 'full')
        """
        super(FisherVectorDL, self).__init__()

        assert covariance_type in ['diag', 'full'], "covariance_type must be 'diag' or 'full'"
        assert n_kernels > 0

        self.n_kernels = n_kernels
        self.feature_dim = feature_dim
        self.covariance_type = covariance_type
        self.fitted = False

        # These will be initialized in build() or initialize()
        self.pi_layer = None
        self.mu_layer = None
        self.sd_layer = None  # For diagonal covariance
        self.cov_layer = None  # For full covariance (lower triangular Cholesky factors)

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

        if self.covariance_type == 'diag':
            # sd: standard deviations (feature_dim, n_kernels)
            self.sd_layer = self.add_weight(
                name='sd',
                shape=(self.feature_dim, self.n_kernels),
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=True,
                constraint=tf.keras.constraints.NonNeg()
            )
        else:  # full covariance
            # Store Cholesky decomposition of covariance matrices (lower triangular)
            # Shape: (n_kernels, feature_dim, feature_dim)
            # Initialize as identity matrices
            init_cov = np.tile(np.eye(self.feature_dim)[None, :, :], (self.n_kernels, 1, 1))
            self.cov_layer = self.add_weight(
                name='cov_cholesky',
                shape=(self.n_kernels, self.feature_dim, self.feature_dim),
                initializer=tf.keras.initializers.Constant(init_cov),
                trainable=True
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
            one_hot = np.eye(self.n_kernels)[labels]  # One-hot encoding
            pi = one_hot.sum(axis=0, keepdims=True)  # Count per cluster

            if self.covariance_type == 'diag':
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

            else:  # full covariance
                # Calculate full covariance matrix for each cluster
                cov_matrices = np.zeros((self.n_kernels, self.feature_dim, self.feature_dim))
                for k in range(self.n_kernels):
                    cluster_mask = labels == k
                    if cluster_mask.sum() > 1:
                        cluster_data = X[cluster_mask]
                        centered = cluster_data - mu[:, k]
                        cov_matrices[k] = (centered.T @ centered) / (pi[0, k] - 1)
                        # Add small regularization to diagonal for numerical stability
                        cov_matrices[k] += np.eye(self.feature_dim) * 1e-6
                    else:
                        cov_matrices[k] = np.eye(self.feature_dim)

                # Normalize pi
                pi = pi / pi.sum()

                # Set model weights
                self.set_params(pi, mu, cov_matrices)

        self.fitted = True
        return self

    def set_params(self, pi, mu, sd_or_cov):
        """
        Set GMM parameters.

        Args:
            pi: Mixture weights (1, n_kernels) or (n_kernels,)
            mu: Means (feature_dim, n_kernels)
            sd_or_cov: For 'diag': Standard deviations (feature_dim, n_kernels)
                      For 'full': Covariance matrices (n_kernels, feature_dim, feature_dim)
        """
        if pi.ndim == 2:
            pi = pi.flatten()

        self.pi_layer.assign(pi)
        self.mu_layer.assign(mu)

        if self.covariance_type == 'diag':
            self.sd_layer.assign(sd_or_cov)
        else:  # full covariance
            # Compute Cholesky decomposition of covariance matrices
            cov_cholesky = np.zeros_like(sd_or_cov)
            for k in range(self.n_kernels):
                try:
                    cov_cholesky[k] = np.linalg.cholesky(sd_or_cov[k])
                except np.linalg.LinAlgError:
                    # If Cholesky fails, add more regularization
                    cov_reg = sd_or_cov[k] + np.eye(self.feature_dim) * 1e-4
                    cov_cholesky[k] = np.linalg.cholesky(cov_reg)
            self.cov_layer.assign(cov_cholesky)

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

        if self.covariance_type == 'diag':
            # Diagonal covariance implementation
            sd = self.sd_layer + 1e-6

            # Compute Gaussian likelihood for each component
            x_expanded = tf.expand_dims(inputs, axis=-1)  # (batch_size, feature_dim, 1)
            mu_expanded = tf.expand_dims(self.mu_layer, axis=0)  # (1, feature_dim, n_kernels)
            sd_expanded = tf.expand_dims(sd, axis=0)  # (1, feature_dim, n_kernels)

            # Compute normalized squared distance
            diff = (x_expanded - mu_expanded) / sd_expanded  # (batch_size, feature_dim, n_kernels)

            # Log probability for diagonal Gaussian
            log_prob = -0.5 * tf.reduce_sum(diff**2, axis=1)  # (batch_size, n_kernels)
            log_prob = log_prob - tf.reduce_sum(tf.math.log(sd_expanded), axis=1)
            log_prob = log_prob - 0.5 * self.feature_dim * tf.math.log(2 * np.pi)

        else:  # full covariance
            # Get Cholesky decomposition L where Cov = L @ L^T
            L = self.cov_layer  # (n_kernels, feature_dim, feature_dim)

            # Compute log determinant: log|Cov| = 2 * sum(log(diag(L)))
            log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(L)) + 1e-10), axis=1)  # (n_kernels,)

            # Compute Mahalanobis distance for each component
            # inputs: (batch_size, feature_dim)
            # mu: (feature_dim, n_kernels)
            x_expanded = tf.expand_dims(inputs, axis=1)  # (batch_size, 1, feature_dim)
            mu_expanded = tf.transpose(self.mu_layer)[tf.newaxis, :, :]  # (1, n_kernels, feature_dim)

            # Centered data
            centered = x_expanded - mu_expanded  # (batch_size, n_kernels, feature_dim)

            # Solve L @ v = centered^T for v, then compute ||v||^2
            # centered: (batch_size, n_kernels, feature_dim)
            # L: (n_kernels, feature_dim, feature_dim)
            # We need to solve for each kernel separately
            log_prob_list = []
            for k in range(self.n_kernels):
                # centered_k: (batch_size, feature_dim)
                centered_k = centered[:, k, :]  # (batch_size, feature_dim)
                L_k = L[k]  # (feature_dim, feature_dim)

                # Solve L @ v = centered_k^T
                v = tf.linalg.triangular_solve(L_k, tf.transpose(centered_k), lower=True)  # (feature_dim, batch_size)
                mahalanobis_sq = tf.reduce_sum(v**2, axis=0)  # (batch_size,)

                # Log probability
                log_prob_k = -0.5 * (mahalanobis_sq + log_det[k] + self.feature_dim * tf.math.log(2 * np.pi))
                log_prob_list.append(log_prob_k)

            log_prob = tf.stack(log_prob_list, axis=1)  # (batch_size, n_kernels)

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
                    'mu': temp_model.mu_layer.numpy()
                }

                if self.covariance_type == 'diag':
                    best_params['sd'] = temp_model.sd_layer.numpy()
                else:
                    # Save covariance matrices (not Cholesky)
                    L = temp_model.cov_layer.numpy()
                    best_params['cov'] = np.array([L[k] @ L[k].T for k in range(n_kernels)])

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

        if self.covariance_type == 'diag':
            self.set_params(best_params['pi'], best_params['mu'], best_params['sd'])
        else:
            self.set_params(best_params['pi'], best_params['mu'], best_params['cov'])

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

        # Compute posterior probabilities (responsibilities)
        # Use uniform weights for likelihood ratio as in original implementation
        equal_weights = np.ones(self.n_kernels) / self.n_kernels

        X_tf = tf.constant(X_matrix, dtype=tf.float32)

        if self.covariance_type == 'diag':
            sd = (self.sd_layer + 1e-6).numpy()  # (feature_dim, n_kernels)

            # Compute log probabilities
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

            # Compute normalized deviation from modes (for diagonal covariance)
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

        else:  # full covariance
            # Get Cholesky decomposition and compute covariance matrices
            L = self.cov_layer.numpy()  # (n_kernels, feature_dim, feature_dim)
            cov_matrices = np.array([L[k] @ L[k].T for k in range(self.n_kernels)])  # (n_kernels, feature_dim, feature_dim)
            cov_inv = np.array([np.linalg.inv(cov_matrices[k]) for k in range(self.n_kernels)])  # (n_kernels, feature_dim, feature_dim)

            # Compute log determinants
            log_det = np.array([np.linalg.slogdet(cov_matrices[k])[1] for k in range(self.n_kernels)])

            # Compute log probabilities for each component
            log_prob_list = []
            for k in range(self.n_kernels):
                centered = X_matrix - mu[:, k]  # (n_samples, feature_dim)
                mahalanobis_sq = np.sum(centered @ cov_inv[k] * centered, axis=1)  # (n_samples,)
                log_prob_k = -0.5 * (mahalanobis_sq + log_det[k] + self.feature_dim * np.log(2 * np.pi))
                log_prob_list.append(log_prob_k)

            log_prob = np.stack(log_prob_list, axis=1) + np.log(equal_weights + 1e-10)  # (n_samples, n_kernels)

            # Convert to probabilities
            log_prob_normalized = log_prob - np.max(log_prob, axis=1, keepdims=True)
            prob = np.exp(log_prob_normalized)
            likelihood_ratio = prob / (prob.sum(axis=1, keepdims=True) + 1e-10)
            likelihood_ratio = likelihood_ratio.reshape(X.shape[0], X.shape[1], self.n_kernels)

            # Compute Fisher vector gradients for full covariance
            mean_dev = np.zeros((X.shape[0], self.n_kernels, self.feature_dim))
            cov_dev = np.zeros((X.shape[0], self.n_kernels, self.feature_dim))

            for k in range(self.n_kernels):
                # Mean gradient
                centered = X - mu[:, k]  # (n_images, n_features, feature_dim)
                weighted_centered = likelihood_ratio[:, :, k:k+1] * (centered @ cov_inv[k])  # (n_images, n_features, feature_dim)
                mean_dev[:, k, :] = weighted_centered.mean(axis=1)  # (n_images, feature_dim)
                mean_dev[:, k, :] *= 1 / np.sqrt(pi[k])

                # Covariance gradient (diagonal elements only for compatibility)
                # For full covariance, we compute the gradient w.r.t. diagonal of precision matrix
                quad_form = np.einsum('...i,ij,...j->...', centered, cov_inv[k], centered)  # (n_images, n_features)
                cov_grad = likelihood_ratio[:, :, k] * (quad_form - self.feature_dim)  # (n_images, n_features)

                # Average over features and use diagonal variance as approximation
                diag_var = np.diag(cov_matrices[k])
                cov_dev[:, k, :] = cov_grad.mean(axis=1, keepdims=True) * np.ones((1, self.feature_dim)) / diag_var
                cov_dev[:, k, :] *= 1 / np.sqrt(2 * pi[k])

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
            'fitted': self.fitted
        }

        if self.covariance_type == 'diag':
            params['sd'] = self.sd_layer.numpy()
        else:
            # For full covariance, save the Cholesky factors
            params['cov_cholesky'] = self.cov_layer.numpy()

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
        if params['covariance_type'] == 'diag':
            # Reconstruct covariance from standard deviations
            model.set_params(params['pi'], params['mu'], params['sd'])
        else:
            # Reconstruct full covariance from Cholesky factors
            L = params['cov_cholesky']
            cov_matrices = np.array([L[k] @ L[k].T for k in range(params['n_kernels'])])
            model.set_params(params['pi'], params['mu'], cov_matrices)

        model.fitted = params['fitted']

        return model
