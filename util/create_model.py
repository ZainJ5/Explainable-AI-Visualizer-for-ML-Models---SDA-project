# util/create_model.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

def generate_and_save_model():
    # 1. Create simple data (e.g., classifying apples/oranges based on size/weight)
    data = {
        'Weight': [150, 160, 130, 140, 190, 170],
        'Size': [6.5, 7.0, 5.5, 6.0, 7.5, 6.8],
        'Color_Index': [0.8, 0.9, 0.7, 0.85, 0.95, 0.88], # 0=Green, 1=Red
        'Fruit_Type': ['Apple', 'Apple', 'Orange', 'Orange', 'Apple', 'Apple']
    }
    df = pd.DataFrame(data)

    X = df[['Weight', 'Size', 'Color_Index']]
    y = df['Fruit_Type']

    # 2. Train the Decision Tree Model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    # 3. Save the model and feature names
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, 'fruit_classifier_tree.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to: {model_path}")
    print(f"Feature names: {list(X.columns)}")
    print(f"Class names: {list(model.classes_)}")

    # We will manually use the feature names in the code for now
    return model_path

if __name__ == '__main__':
    generate_and_save_model()