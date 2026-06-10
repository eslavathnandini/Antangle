import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc


class Visualizer:
    """Visualization tools for model evaluation and interpretation."""
    
    def __init__(self, output_dir='outputs'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_model_comparison(self, results_df, save_path=None):
        """Plot comparison of different models."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Reset index if Model is in index
        if 'Model' not in results_df.columns:
            results_df = results_df.reset_index()
            if 'index' in results_df.columns:
                results_df = results_df.rename(columns={'index': 'Model'})
            elif results_df.columns[0] != 'Model':
                results_df = results_df.rename(columns={results_df.columns[0]: 'Model'})
        
        # Accuracy comparison
        axes[0].barh(results_df['Model'], results_df['Accuracy'])
        axes[0].set_xlabel('Accuracy')
        axes[0].set_title('Model Comparison - Accuracy')
        axes[0].set_xlim(0, 1)
        
        # F1 comparison
        axes[1].barh(results_df['Model'], results_df['F1'])
        axes[1].set_xlabel('F1 Score')
        axes[1].set_title('Model Comparison - F1 Score')
        axes[1].set_xlim(0, 1)
        
        # Precision comparison
        axes[2].barh(results_df['Model'], results_df['Precision'])
        axes[2].set_xlabel('Precision')
        axes[2].set_title('Model Comparison - Precision')
        axes[2].set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        return self
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name='Model', save_path=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Bad', 'Good'], 
                   yticklabels=['Bad', 'Good'], ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{model_name} - Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        return self
    
    def plot_roc_curve(self, y_true, y_prob, model_name='Model', save_path=None):
        """Plot ROC curve for binary classification."""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} - ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        return self
    
    def plot_feature_importance(self, model, feature_names, top_n=10, save_path=None):
        """Plot feature importance for tree-based models."""
        if not hasattr(model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute")
            return self
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        return self
    
    def plot_cv_results(self, results_df, save_path=None):
        """Plot cross-validation results with error bars."""
        # Check if CV columns exist
        if 'CV_Accuracy_mean' not in results_df.columns:
            print("CV results not available, skipping plot")
            return self
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Reset index if needed
        if 'Model' not in results_df.columns:
            results_df = results_df.reset_index()
            if 'index' in results_df.columns:
                results_df = results_df.rename(columns={'index': 'Model'})
        
        models = results_df['Model']
        means = results_df['CV_Accuracy_mean']
        stds = results_df['CV_Accuracy_std']
        
        x = np.arange(len(models))
        
        ax.bar(x, means, yerr=stds, capsize=5, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('CV Accuracy')
        ax.set_title('Cross-Validation Results')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        return self
    
    def plot_prediction_distribution(self, y_true, y_pred, model_name='Model', save_path=None):
        """Plot distribution of predictions vs actual values."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot of actual vs predicted
        x = np.arange(2)
        width = 0.35
        
        actual_counts = [np.sum(y_true == 0), np.sum(y_true == 1)]
        pred_counts = [np.sum(y_pred == 0), np.sum(y_pred == 1)]
        
        axes[0].bar(x - width/2, actual_counts, width, label='Actual', edgecolor='black')
        axes[0].bar(x + width/2, pred_counts, width, label='Predicted', edgecolor='black')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'{model_name} - Actual vs Predicted Distribution')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(['Bad', 'Good'])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Percentage correct by class
        classes = ['Bad', 'Good']
        correct_percentages = []
        for cls in [0, 1]:
            mask = y_true == cls
            if np.sum(mask) > 0:
                correct = np.sum(y_pred[mask] == cls)
                correct_percentages.append(correct / np.sum(mask) * 100)
            else:
                correct_percentages.append(0)
        
        axes[1].bar(classes, correct_percentages, edgecolor='black')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title(f'{model_name} - Accuracy by Class')
        axes[1].set_ylim(0, 105)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        return self
    
    def generate_all_plots(self, trainer, X_test, y_test, y_pred, feature_names=None):
        """Generate all visualization plots."""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Get results - handle both trainer types
        if hasattr(trainer, 'get_results_df'):
            results_df = trainer.get_results_df()
        else:
            results_df = trainer.get_results_dataframe()
        
        # Model comparison
        self.plot_model_comparison(results_df, f'{self.output_dir}/model_comparison.png')
        
        # Confusion matrix for best model
        self.plot_confusion_matrix(y_test, y_pred, 
                                   trainer.best_model_name,
                                   f'{self.output_dir}/confusion_matrix.png')
        
        # ROC curve (if applicable)
        if hasattr(trainer.best_model, 'predict_proba'):
            y_prob = trainer.best_model.predict_proba(X_test)
            self.plot_roc_curve(y_test, y_prob,
                               trainer.best_model_name,
                               f'{self.output_dir}/roc_curve.png')
        
        # Prediction distribution
        self.plot_prediction_distribution(y_test, y_pred,
                                         trainer.best_model_name,
                                         f'{self.output_dir}/prediction_distribution.png')
        
        # Feature importance (if applicable)
        if hasattr(trainer.best_model, 'feature_importances_') and feature_names:
            self.plot_feature_importance(trainer.best_model, feature_names,
                                        save_path=f'{self.output_dir}/feature_importance.png')
        
        print("All visualizations generated!")
        
        return self


if __name__ == "__main__":
    # Example usage
    visualizer = Visualizer()
    
    # Create sample data for demonstration
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    
    visualizer.plot_confusion_matrix(y_true, y_pred, 'Sample Model')
