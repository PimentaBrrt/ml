import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

encoder = OneHotEncoder()
scaler = StandardScaler()
l_encoder = LabelEncoder()

df = pd.read_csv("docs/knn/booking.csv")

df = df.drop(columns=["Booking_ID", "date of reservation"])

numeric_cols = ["number of adults", "number of children", "number of weekend nights", 
                "number of week nights", "lead time", "P-C", "P-not-C", 
                "average price", "special requests"]

categorical_cols = ["type of meal", "room type", "market segment type"]

X = df.drop("booking status", axis=1)

X_encoded = encoder.fit_transform(X[categorical_cols])
encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols), index=X.index)

X_scaled = scaler.fit_transform(X[numeric_cols])
scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols, index=X.index)

X = pd.concat([scaled_df, encoded_df], axis=1)

y = l_encoder.fit_transform(df["booking status"])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

svm_rbf = SVC(kernel="rbf", C=1, gamma="scale")
svm_linear = LinearSVC()

scores_rbf = cross_val_score(svm_rbf, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
scores_lin = cross_val_score(svm_linear, X, y, cv=skf, scoring='accuracy', n_jobs=-1)

print("RBF CV scores:", scores_rbf, "\nmean:", scores_rbf.mean())
print("\n\nLinearSVC CV scores:\n", scores_lin, "mean:", scores_lin.mean())