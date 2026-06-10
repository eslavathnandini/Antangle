import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


class ModelTrainer:
    """Train and evaluate multiple classification models with hyperparameter tuning."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def get_param_distributions(self):
        """Define hyperparameter distributions for RandomizedSearchCV."""
        param_dist = {
            'Logistic Regression': {
                'C': uniform(0.01, 10),
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            
            'KNN': {
                'n_neighbors': randint(3, 20),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
                'p': [1, 2]
            },
            
            'Random Forest': {
                'n_estimators': randint(50, 300),
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None]
            },
            
            'Gradient Boosting': {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(2, 10),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'subsample': uniform(0.6, 0.4)
            },
            
            'SVM': {
                'C': uniform(0.1, 100),
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7)),
                'degree': [2, 3, 4]
            },
            
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25), (128, 64, 32)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': uniform(0.0001, 0.01),
                'learning_rate': ['constant', 'adaptive'],
                'learning_rate_init': uniform(0.001, 0.01)
            }
        }
        
        if HAS_XGB:
            param_dist['XGBoost'] = {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(2, 10),
                'min_child_weight': randint(1, 10),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'gamma': uniform(0, 0.5),
                'reg_alpha': uniform(0, 1),
                'reg_lambda': uniform(0, 1)
            }
        
        if HAS_LGBM:
            param_dist['LightGBM'] = {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(2, 10),
                'num_leaves': randint(20, 100),
                'min_child_samples': randint(5, 50),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'reg_alpha': uniform(0, 1),
                'reg_lambda': uniform(0, 1)
            }
        
        return param_dist
    
    def get_base_models(self):
        """Define base classification models (without tuned params)."""
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            ),
            
            'KNN': KNeighborsClassifier(),
            
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state
            ),
            
            'SVM': SVC(
                probability=True,
                random_state=self.random_state
            ),
            
            'Neural Network': MLPClassifier(
                max_iter=1000,
                random_state=self.random_state,
                early_stopping=True
            )
        }
        
        if HAS_XGB:
            models['XGBoost'] = XGBClassifier(
                random_state=self.random_state,
                verbosity=0,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        if HAS_LGBM:
            models['LightGBM'] = LGBMClassifier(
                random_state=self.random_state,
                verbose=-1
            )
        
        return models
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets."""
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=self.random_state, stratify=y_train_val
        )
        
        print(f"Data split:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def evaluate_model(self, y_true, y_pred, y_prob=None):
        """Calculate classification metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        roc_auc = None
        if y_prob is not None and len(np.unique(y_true)) == 2:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC_AUC': roc_auc
        }
    
    def tune_and_train_all(self, X_train, y_train, X_val, y_val, cv=5, n_iter=100):
        """Tune hyperparameters and train all models."""
        base_models = self.get_base_models()
        param_dists = self.get_param_distributions()
        
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING & MODEL TRAINING")
        print("="*70)
        
        for name, model in base_models.items():
            print(f"\n{'='*50}")
            print(f"Tuning {name}...")
            print(f"{'='*50}")
            
            try:
                if name in param_dists:
                    # RandomizedSearchCV for hyperparameter tuning
                    search = RandomizedSearchCV(
                        model,
                        param_dists[name],
                        n_iter=min(n_iter, self._count_combinations(param_dists[name])),
                        cv=cv,
                        scoring='accuracy',
                        n_jobs=-1,
                        random_state=self.random_state,
                        verbose=0
                    )
                    
                    search.fit(X_train, y_train)
                    
                    # Best model from tuning
                    best_model = search.best_estimator_
                    print(f"  Best params: {search.best_params_}")
                    print(f"  Best CV Accuracy: {search.best_score_:.4f}")
                else:
                    # No tuning, just fit
                    best_model = model
                    best_model.fit(X_train, y_train)
                    print(f"  No tuning grid defined, using defaults")
                
                # Evaluate on validation set
                y_pred_val = best_model.predict(X_val)
                y_prob_val = best_model.predict_proba(X_val) if hasattr(best_model, 'predict_proba') else None
                
                val_metrics = self.evaluate_model(y_val, y_pred_val, y_prob_val)
                
                # Cross-validation on train+val
                X_cv = np.vstack([X_train, X_val])
                y_cv = np.concatenate([y_train, y_val])
                
                kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
                cv_scores = cross_val_score(best_model, X_cv, y_cv, cv=kf, scoring='accuracy')
                
                cv_results = {
                    'mean_accuracy': cv_scores.mean(),
                    'std_accuracy': cv_scores.std(),
                    'scores': cv_scores
                }
                
                # Store results
                self.models[name] = best_model
                self.results[name] = {
                    'val_metrics': val_metrics,
                    'cv_results': cv_results
                }
                
                print(f"  Validation Accuracy: {val_metrics['Accuracy']:.4f}")
                print(f"  Validation ROC AUC: {val_metrics['ROC_AUC']:.4f}" if val_metrics['ROC_AUC'] else "")
                print(f"  CV Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
                
            except Exception as e:
                print(f"  Error with {name}: {str(e)}")
        
        return self
    
    def _count_combinations(self, param_dist):
        """Estimate number of parameter combinations."""
        count = 1
        for v in param_dist.values():
            if isinstance(v, list):
                count *= len(v)
            elif hasattr(v, 'rvs'):
                count *= 50  # approximate for distributions
        return count
    
    def get_best_model(self, metric='Accuracy'):
        """Get the best model based on specified metric."""
        if not self.results:
            raise ValueError("No models have been trained yet.")
        
        best_score = -np.inf
        best_name = None
        
        for name, result in self.results.items():
            score = result['val_metrics'][metric]
            if score and score > best_score:
                best_score = score
                best_name = name
        
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        
        print(f"\nBest Model: {best_name} (Accuracy: {best_score:.4f})")
        
        return self.best_model, best_name
    
    def get_results_dataframe(self):
        """Convert results to pandas DataFrame."""
        results_list = []
        
        for name, result in self.results.items():
            row = {'Model': name}
            row.update(result['val_metrics'])
            row['CV_Accuracy_mean'] = result['cv_results']['mean_accuracy']
            row['CV_Accuracy_std'] = result['cv_results']['std_accuracy']
            results_list.append(row)
        
        df = pd.DataFrame(results_list)
        df = df.sort_values('Accuracy', ascending=False)
        
        return df
    
    def final_evaluation(self, X_test, y_test):
        """Evaluate the best model on the test set."""
        if self.best_model is None:
            self.get_best_model()
        
        y_pred_test = self.best_model.predict(X_test)
        y_prob_test = self.best_model.predict_proba(X_test) if hasattr(self.best_model, 'predict_proba') else None
        
        test_metrics = self.evaluate_model(y_test, y_pred_test, y_prob_test)
        
        print("\n" + "="*70)
        print(f"FINAL EVALUATION ON TEST SET - {self.best_model_name}")
        print("="*70)
        print(f"  Accuracy: {test_metrics['Accuracy']:.4f}")
        print(f"  Precision: {test_metrics['Precision']:.4f}")
        print(f"  Recall: {test_metrics['Recall']:.4f}")
        print(f"  F1: {test_metrics['F1']:.4f}")
        if test_metrics['ROC_AUC']:
            print(f"  ROC AUC: {test_metrics['ROC_AUC']:.4f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_test))
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test))
        
        return test_metrics, y_pred_test


def get_quality_category(good_prob):
    """Convert prediction probabilities to quality category."""
    if good_prob >= 0.7:
        return 'High Quality (Good)'
    elif good_prob >= 0.3:
        return 'Medium Quality'
    else:
        return 'Low Quality (Bad)'


if __name__ == "__main__":
    from preprocessing import FeaturePreprocessor
    
    # Load data
    df = pd.read_csv("../apple_quality.csv")
    X = df.drop(['A_id', 'Quality'], axis=1).values
    y = df['Quality'].values
    
    # Preprocess
    preprocessor = FeaturePreprocessor()
    X_processed = preprocessor.apply_full_pipeline(X)
    y_encoded = preprocessor.encode_target(y)
    
    # Train models with tuning
    trainer = ModelTrainer()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X_processed, y_encoded)
    trainer.tune_and_train_all(X_train, y_train, X_val, y_val, n_iter=50)
    
    # Get best model
    best_model, best_name = trainer.get_best_model()
    
    # Final evaluation
    test_metrics, y_pred = trainer.final_evaluation(X_test, y_test)
    
    # Show results
    print("\n" + "="*70)
    print("ALL MODEL RESULTS (TUNED)")
    print("="*70)
    print(trainer.get_results_dataframe().to_string(index=False))
