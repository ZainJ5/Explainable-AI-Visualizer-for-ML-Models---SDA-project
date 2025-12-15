"""
Sample Model Generator
Creates sample ML models for demonstration purposes
"""

import numpy as np
import pickle
import os


def create_sample_models():
    """Create and save sample ML models for testing"""
    
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        
        feature_names = [
            'Age', 'Income', 'Credit_Score', 'Debt_Ratio',
            'Employment_Length', 'Num_Accounts', 'Num_Credit_Lines',
            'Num_Late_Payments', 'Balance', 'Loan_Amount'
        ]
        
        print("Creating Decision Tree model...")
        dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt_model.fit(X, y)
        dt_model.feature_names = feature_names
        
        with open('decision_tree_model.pkl', 'wb') as f:
            pickle.dump(dt_model, f, protocol=4)  
        print("✓ Decision Tree model saved as 'decision_tree_model.pkl'")
        
        print("\nCreating Random Forest model...")
        rf_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        rf_model.fit(X, y)
        rf_model.feature_names = feature_names
        
        with open('random_forest_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f, protocol=4)  
        print("✓ Random Forest model saved as 'random_forest_model.pkl'")
        
        print("\nCreating Logistic Regression model...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X, y)
        lr_model.feature_names = feature_names
        
        with open('logistic_regression_model.pkl', 'wb') as f:
            pickle.dump(lr_model, f, protocol=4)  
        print("✓ Logistic Regression model saved as 'logistic_regression_model.pkl'")
        
        print("\n" + "="*60)
        print("All sample models created successfully!")
        print("="*60)
        print("\nModel Details:")
        print(f"  - Features: {len(feature_names)}")
        print(f"  - Feature names: {', '.join(feature_names)}")
        print(f"  - Training samples: {len(X)}")
        print(f"  - Classes: {len(np.unique(y))}")
        print("\nYou can now load these models in the application:")
        print("  1. decision_tree_model.pkl")
        print("  2. random_forest_model.pkl")
        print("  3. logistic_regression_model.pkl")
        
        return True
        
    except ImportError as e:
        print(f"Error: sklearn not installed. Please install it: pip install scikit-learn")
        print(f"Details: {str(e)}")
        return False
    except Exception as e:
        print(f"Error creating models: {str(e)}")
        return False


if __name__ == "__main__":
    print("Sample Model Generator")
    print("="*60)
    print("\nThis script will create 3 sample ML models for demonstration:\n")
    print("  1. Decision Tree Classifier")
    print("  2. Random Forest Classifier")
    print("  3. Logistic Regression Classifier")
    print("\n" + "="*60 + "\n")
    
    create_sample_models()
