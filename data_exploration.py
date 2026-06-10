import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class DataExplorer:
    """Explore and understand the apple quality dataset."""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        
    def load_data(self):
        """Load the CSV data."""
        self.df = pd.read_csv(self.data_path)
        
        # Remove the last row if it's metadata
        if self.df.iloc[-1].isnull().all() or 'Created_by' in str(self.df.iloc[-1].values):
            self.df = self.df.iloc[:-1]
        
        # Remove A_id column and separate features/target
        self.X = self.df.drop(['A_id', 'Quality'], axis=1)
        
        # Convert all feature columns to numeric, coercing errors
        for col in self.X.columns:
            self.X[col] = pd.to_numeric(self.X[col], errors='coerce')
        
        self.y = self.df['Quality']
        self.feature_names = self.X.columns.tolist()
        
        print(f"Data loaded: {self.df.shape[0]} samples, {self.X.shape[1]} features")
        print(f"Features: {self.feature_names}")
        return self
    
    def basic_stats(self):
        """Print basic statistics."""
        print("\n" + "="*60)
        print("BASIC DATASET STATISTICS")
        print("="*60)
        
        print(f"\nTotal samples: {len(self.y)}")
        print(f"Total features: {self.X.shape[1]}")
        print(f"\nTarget distribution:")
        print(self.y.value_counts())
        
        print("\n--- Feature Statistics ---")
        print(self.X.describe())
        
        print("\n--- Missing Values ---")
        missing = self.X.isnull().sum().sum()
        print(f"Total missing values: {missing}")
        
        print("\n--- Data Types ---")
        print(self.X.dtypes)
        
        return self
    
    def plot_target_distribution(self, save_path=None):
        """Plot distribution of Quality classes."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Count plot
        self.y.value_counts().plot(kind='bar', ax=axes[0], edgecolor='black')
        axes[0].set_xlabel('Quality')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Distribution of Apple Quality')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
        
        # Pie chart
        self.y.value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[1])
        axes[1].set_title('Quality Distribution')
        axes[1].set_ylabel('')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        return self
    
    def plot_feature_distributions(self, save_path=None):
        """Plot distribution of each feature."""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, col in enumerate(self.feature_names):
            ax = axes[i]
            for quality in self.y.unique():
                data = self.X[self.y == quality][col]
                ax.hist(data, bins=30, alpha=0.5, label=quality, edgecolor='black')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{col} Distribution')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        return self
    
    def plot_correlation_matrix(self, save_path=None):
        """Plot correlation matrix between features."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr_matrix = self.X.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        return self
    
    def plot_feature_vs_target(self, save_path=None):
        """Plot each feature vs target class."""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, col in enumerate(self.feature_names):
            ax = axes[i]
            sns.boxplot(x=self.y, y=self.X[col], ax=ax)
            ax.set_xlabel('Quality')
            ax.set_ylabel(col)
            ax.set_title(f'{col} by Quality')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        return self
    
    def detect_outliers(self, method='iqr', threshold=1.5):
        """Detect outliers in the dataset."""
        print("\n" + "="*60)
        print("OUTLIER DETECTION")
        print("="*60)
        
        outlier_counts = {}
        
        for col in self.feature_names:
            Q1 = self.X[col].quantile(0.25)
            Q3 = self.X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = self.X[(self.X[col] < lower_bound) | (self.X[col] > upper_bound)]
            outlier_counts[col] = len(outliers)
        
        print(f"\nOutliers per feature (IQR method, threshold={threshold}):")
        for col, count in outlier_counts.items():
            print(f"  {col}: {count} outliers ({count/len(self.X)*100:.1f}%)")
        
        return self
    
    def get_processed_data(self):
        """Return processed data for modeling."""
        return self.X.values, self.y.values, self.feature_names


def run_full_eda(data_path, output_dir='outputs'):
    """Run complete exploratory data analysis."""
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize and run exploration
    explorer = DataExplorer(data_path)
    explorer.load_data()
    explorer.basic_stats()
    
    # Generate plots
    explorer.plot_target_distribution(f'{output_dir}/target_distribution.png')
    explorer.plot_feature_distributions(f'{output_dir}/feature_distributions.png')
    explorer.plot_correlation_matrix(f'{output_dir}/correlation_matrix.png')
    explorer.plot_feature_vs_target(f'{output_dir}/feature_vs_target.png')
    explorer.detect_outliers()
    
    return explorer


if __name__ == "__main__":
    # Run EDA
    data_path = os.path.join(os.path.dirname(__file__), "..", "apple_quality.csv")
    explorer = run_full_eda(data_path)
