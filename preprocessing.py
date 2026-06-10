import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from pathlib import Path


class FeaturePreprocessor:
    """Preprocessing pipeline for apple quality features."""
    
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.label_encoder = None
        self.feature_names = None
        
    def encode_target(self, y, fit=True):
        """Encode categorical target to numerical values."""
        if fit:
            self.label_encoder = LabelEncoder()
            return self.label_encoder.fit_transform(y)
        else:
            return self.label_encoder.transform(y)
    
    def standardize(self, X, fit=True):
        """Standardize features (zero mean, unit variance)."""
        if fit:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def minmax_scale(self, X, fit=True):
        """Min-Max normalization to [0, 1] range."""
        if fit:
            self.scaler = MinMaxScaler()
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def robust_scale(self, X, fit=True):
        """Scale using statistics robust to outliers."""
        if fit:
            self.scaler = RobustScaler()
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def reduce_dimensions_pca(self, X, n_components=0.95, fit=True):
        """Reduce dimensions using PCA.
        
        Args:
            X: Feature data
            n_components: Number of components or variance to retain
            fit: Whether to fit PCA (True for training, False for test)
        """
        if fit:
            self.pca = PCA(n_components=n_components)
            X_pca = self.pca.fit_transform(X)
            print(f"PCA: {X.shape[1]} features -> {X_pca.shape[1]} components")
            print(f"Explained variance: {sum(self.pca.explained_variance_ratio_):.2%}")
        else:
            X_pca = self.pca.transform(X)
        
        return X_pca
    
    def apply_full_pipeline(self, X, pipeline_config=None):
        """Apply a complete preprocessing pipeline.
        
        Args:
            X: Raw feature data
            pipeline_config: Dictionary of preprocessing steps
        """
        if pipeline_config is None:
            pipeline_config = {
                'standardize': True
            }
        
        X_processed = X.copy()
        
        # Standardize
        if pipeline_config.get('standardize', False):
            X_processed = self.standardize(X_processed, fit=True)
            print("Applied standardization")
        
        # MinMax scaling
        if pipeline_config.get('minmax', False):
            X_processed = self.minmax_scale(X_processed, fit=True)
            print("Applied Min-Max scaling")
        
        # Robust scaling
        if pipeline_config.get('robust', False):
            X_processed = self.robust_scale(X_processed, fit=True)
            print("Applied Robust scaling")
        
        return X_processed


def get_preprocessing_options():
    """Return available preprocessing pipeline configurations."""
    return {
        'standard': {
            'standardize': True
        },
        'minmax': {
            'minmax': True
        },
        'robust': {
            'robust': True
        },
        'none': {}
    }


if __name__ == "__main__":
    # Example usage
    import os
    import pandas as pd
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "apple_quality.csv")
    df = pd.read_csv(data_path)
    X = df.drop(['A_id', 'Quality'], axis=1).values
    
    print("Raw data shape:", X.shape)
    
    # Apply preprocessing
    preprocessor = FeaturePreprocessor()
    X_processed = preprocessor.apply_full_pipeline(X)
    
    print("Processed data shape:", X_processed.shape)
