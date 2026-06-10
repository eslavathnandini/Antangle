"""
Fruit Quality Prediction - Optimized Pipeline
==============================================
ML pipeline with PLS, optimized models, and ensembles.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from optimized_models import OptimizedModelTrainer
from visualization import Visualizer


class FruitQualityPipeline:
    """Optimized pipeline for fruit quality prediction."""
    
    def __init__(self, data_path, output_dir='outputs'):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.trainer = OptimizedModelTrainer()
        self.visualizer = Visualizer(output_dir)
        self.results = {}
        
    def run(self):
        """Run the complete optimized pipeline."""
        print("\n" + "#"*60)
        print("# FRUIT QUALITY PREDICTION - OPTIMIZED")
        print("#"*60)
        
        # Load and preprocess
        X, y = self.trainer.load_and_preprocess(self.data_path)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        self.trainer.train_pls(X_train, y_train, X_test, y_test)
        self.trainer.train_optimized_models(X_train, y_train, X_test, y_test)
        
        # Get best model
        best_model, best_name = self.trainer.get_best_model()
        
        # Final report
        y_pred = self.trainer.final_report(X_test, y_test)
        
        # Visualizations
        self.visualizer.generate_all_plots(self.trainer, X_test, y_test, y_pred)
        
        # Save results
        self.save_results()
        
        print("\n" + "#"*60)
        print(f"# DONE! Results saved to {self.output_dir}")
        print("#"*60)
        
        return self
    
    def save_results(self):
        """Save model and results."""
        import pickle
        
        # Save model
        with open(f'{self.output_dir}/best_model.pkl', 'wb') as f:
            pickle.dump({
                'model': self.trainer.best_model,
                'scaler': self.trainer.scaler,
                'label_encoder': self.trainer.le,
                'model_name': self.trainer.best_model_name
            }, f)
        
        # Save results CSV
        results_df = self.trainer.get_results_df()
        results_df.to_csv(f'{self.output_dir}/results.csv')
        
        print(f"\nModel saved to {self.output_dir}/best_model.pkl")
        print(f"Results saved to {self.output_dir}/results.csv")


if __name__ == "__main__":
    pipeline = FruitQualityPipeline("../apple_quality.csv")
    pipeline.run()
