import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Select features
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']]

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())

# Convert gender to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Improved Random Forest Model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Test model
pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, pred))

# Sample prediction
sample = [[1, 1, 25, 100, 0, 0]]
result = model.predict(sample)

if result[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger Did Not Survive")