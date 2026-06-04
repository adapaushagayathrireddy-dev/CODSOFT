import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("IMDb Movies India.csv",encoding="latin1")

# Remove missing ratings
df = df.dropna(subset=["Rating"])

# Clean Year column
df["Year"] = df["Year"].astype(str).str.extract(r'(\d{4})')
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Clean Duration column
df["Duration"] = df["Duration"].astype(str).str.extract(r'(\d+)')
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")

# Clean Votes column
df["Votes"] = df["Votes"].astype(str).str.replace(",", "", regex=False)
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

features = [
    "Genre",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3",
    "Year",
    "Duration",
    "Votes"
]

X = df[features]
y = df["Rating"]

categorical = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
numerical = ["Year", "Duration", "Votes"]

preprocessor = ColumnTransformer([
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical
    ),
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]),
        numerical
    )
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("R2 Score:", round(r2_score(y_test, predictions), 4))
print("MAE:", round(mean_absolute_error(y_test, predictions), 4))

# Example Prediction
sample = pd.DataFrame({
    "Genre": ["Drama"],
    "Director": ["Shoojit Sircar"],
    "Actor 1": ["Jimmy Sheirgill"],
    "Actor 2": ["Minissha Lamba"],
    "Actor 3": ["Yashpal Sharma"],
    "Year": [2005],
    "Duration": [142],
    "Votes": [1086]
})

rating = model.predict(sample)

print("Predicted Rating:", round(rating[0], 2))