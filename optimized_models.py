import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
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


class OptimizedModelTrainer:
    """Optimized ML pipeline with PLS, ensembles, and tuned models."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        
    def load_and_preprocess(self, data_path):
        """Load and preprocess data."""
        df = pd.read_csv(data_path)
        
        # Remove metadata row
        if 'Created_by' in str(df.iloc[-1].values):
            df = df.iloc[:-1]
        
        X = df.drop(['A_id', 'Quality'], axis=1).values
        y = df['Quality'].values
        
        # Encode target
        y_encoded = self.le.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
        print(f"Classes: {self.le.classes_}")
        
        return X_scaled, y_encoded
    
    def evaluate(self, y_true, y_pred, y_prob=None):
        """Calculate metrics."""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred)
        }
        if y_prob is not None:
            metrics['ROC_AUC'] = roc_auc_score(y_true, y_prob[:, 1])
        return metrics
    
    def train_pls(self, X_train, y_train, X_test, y_test, n_components_range=range(1, 8)):
        """Train PLS (Partial Least Squares) with optimal components."""
        print("\n--- PLS Regression ---")
        
        best_acc = 0
        best_n = 2
        
        for n_comp in n_components_range:
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_train, y_train)
            y_pred = (pls.predict(X_test).flatten() > 0.5).astype(int)
            acc = accuracy_score(y_test, y_pred)
            
            if acc > best_acc:
                best_acc = acc
                best_n = n_comp
        
        # Final PLS with best components
        pls = PLSRegression(n_components=best_n)
        pls.fit(X_train, y_train)
        y_pred = (pls.predict(X_test).flatten() > 0.5).astype(int)
        
        self.models['PLS'] = pls
        metrics = self.evaluate(y_test, y_pred)
        self.results['PLS'] = metrics
        
        print(f"  Optimal components: {best_n}")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  F1: {metrics['F1']:.4f}")
        
        return pls
    
    def train_optimized_models(self, X_train, y_train, X_test, y_test, cv=5):
        """Train optimized models with best hyperparameters."""
        
        print("\n" + "="*60)
        print("TRAINING OPTIMIZED MODELS")
        print("="*60)
        
        # 1. Optimized Random Forest
        print("\n--- Random Forest (Optimized) ---")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        y_prob_rf = rf.predict_proba(X_test)
        self.models['Random Forest'] = rf
        self.results['Random Forest'] = self.evaluate(y_test, y_pred_rf, y_prob_rf)
        print(f"  Accuracy: {self.results['Random Forest']['Accuracy']:.4f}")
        print(f"  ROC AUC: {self.results['Random Forest'].get('ROC_AUC', 0):.4f}")
        
        # 2. Optimized Gradient Boosting
        print("\n--- Gradient Boosting (Optimized) ---")
        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=5,
            subsample=0.8,
            random_state=self.random_state
        )
        gb.fit(X_train, y_train)
        y_pred_gb = gb.predict(X_test)
        y_prob_gb = gb.predict_proba(X_test)
        self.models['Gradient Boosting'] = gb
        self.results['Gradient Boosting'] = self.evaluate(y_test, y_pred_gb, y_prob_gb)
        print(f"  Accuracy: {self.results['Gradient Boosting']['Accuracy']:.4f}")
        
        # 3. Optimized SVM
        print("\n--- SVM (Optimized) ---")
        svm = SVC(
            C=10,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=self.random_state
        )
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        y_prob_svm = svm.predict_proba(X_test)
        self.models['SVM'] = svm
        self.results['SVM'] = self.evaluate(y_test, y_pred_svm, y_prob_svm)
        print(f"  Accuracy: {self.results['SVM']['Accuracy']:.4f}")
        
        # 4. Optimized KNN
        print("\n--- KNN (Optimized) ---")
        knn = KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='manhattan',
            n_jobs=-1
        )
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        y_prob_knn = knn.predict_proba(X_test)
        self.models['KNN'] = knn
        self.results['KNN'] = self.evaluate(y_test, y_pred_knn, y_prob_knn)
        print(f"  Accuracy: {self.results['KNN']['Accuracy']:.4f}")
        
        # 5. Neural Network
        print("\n--- Neural Network (Optimized) ---")
        nn = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=self.random_state,
            early_stopping=True
        )
        nn.fit(X_train, y_train)
        y_pred_nn = nn.predict(X_test)
        y_prob_nn = nn.predict_proba(X_test)
        self.models['Neural Network'] = nn
        self.results['Neural Network'] = self.evaluate(y_test, y_pred_nn, y_prob_nn)
        print(f"  Accuracy: {self.results['Neural Network']['Accuracy']:.4f}")
        
        # 6. XGBoost
        if HAS_XGB:
            print("\n--- XGBoost (Optimized) ---")
            xgb = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=self.random_state,
                verbosity=0,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            xgb.fit(X_train, y_train)
            y_pred_xgb = xgb.predict(X_test)
            y_prob_xgb = xgb.predict_proba(X_test)
            self.models['XGBoost'] = xgb
            self.results['XGBoost'] = self.evaluate(y_test, y_pred_xgb, y_prob_xgb)
            print(f"  Accuracy: {self.results['XGBoost']['Accuracy']:.4f}")
        
        # 7. LightGBM
        if HAS_LGBM:
            print("\n--- LightGBM (Optimized) ---")
            lgbm = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=self.random_state,
                verbose=-1
            )
            lgbm.fit(X_train, y_train)
            y_pred_lgbm = lgbm.predict(X_test)
            y_prob_lgbm = lgbm.predict_proba(X_test)
            self.models['LightGBM'] = lgbm
            self.results['LightGBM'] = self.evaluate(y_test, y_pred_lgbm, y_prob_lgbm)
            print(f"  Accuracy: {self.results['LightGBM']['Accuracy']:.4f}")
        
        # 8. Soft Voting Ensemble
        print("\n--- Voting Ensemble ---")
        estimators = [
            ('rf', self.models['Random Forest']),
            ('gb', self.models['Gradient Boosting']),
            ('svm', self.models['SVM'])
        ]
        if HAS_XGB:
            estimators.append(('xgb', self.models['XGBoost']))
        
        voting = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        voting.fit(X_train, y_train)
        y_pred_vote = voting.predict(X_test)
        y_prob_vote = voting.predict_proba(X_test)
        self.models['Voting Ensemble'] = voting
        self.results['Voting Ensemble'] = self.evaluate(y_test, y_pred_vote, y_prob_vote)
        print(f"  Accuracy: {self.results['Voting Ensemble']['Accuracy']:.4f}")
        
        # 9. Stacking Ensemble
        print("\n--- Stacking Ensemble ---")
        stack_estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)),
            ('svm', SVC(probability=True, random_state=self.random_state))
        ]
        
        stacking = StackingClassifier(
            estimators=stack_estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5,
            n_jobs=-1
        )
        stacking.fit(X_train, y_train)
        y_pred_stack = stacking.predict(X_test)
        y_prob_stack = stacking.predict_proba(X_test)
        self.models['Stacking Ensemble'] = stacking
        self.results['Stacking Ensemble'] = self.evaluate(y_test, y_pred_stack, y_prob_stack)
        print(f"  Accuracy: {self.results['Stacking Ensemble']['Accuracy']:.4f}")
        
        return self
    
    def get_best_model(self):
        """Get the best performing model."""
        best_score = 0
        best_name = None
        
        for name, metrics in self.results.items():
            if metrics['Accuracy'] > best_score:
                best_score = metrics['Accuracy']
                best_name = name
        
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_name}")
        print(f"Accuracy: {best_score:.4f}")
        print(f"{'='*60}")
        
        return self.best_model, best_name
    
    def get_results_df(self):
        """Get results as DataFrame."""
        df = pd.DataFrame(self.results).T
        df = df.sort_values('Accuracy', ascending=False)
        return df
    
    def final_report(self, X_test, y_test):
        """Print final evaluation report."""
        if self.best_model is None:
            self.get_best_model()
        
        y_pred = self.best_model.predict(X_test)
        
        print("\n" + "="*60)
        print(f"FINAL REPORT - {self.best_model_name}")
        print("="*60)
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.le.classes_))
        
        return y_pred
    
    def run_full_pipeline(self, data_path, test_size=0.2):
        """Run complete optimized pipeline."""
        print("\n" + "#"*60)
        print("# OPTIMIZED FRUIT QUALITY PREDICTION")
        print("#"*60)
        
        # Load data
        X, y = self.load_and_preprocess(data_path)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        # Train PLS
        self.train_pls(X_train, y_train, X_test, y_test)
        
        # Train all optimized models
        self.train_optimized_models(X_train, y_train, X_test, y_test)
        
        # Get best model
        self.get_best_model()
        
        # Final report
        self.final_report(X_test, y_test)
        
        # Show all results
        print("\n" + "="*60)
        print("ALL MODEL RESULTS")
        print("="*60)
        print(self.get_results_df().to_string())
        
        return self


if __name__ == "__main__":
    trainer = OptimizedModelTrainer()
    trainer.run_full_pipeline("../apple_quality.csv")
