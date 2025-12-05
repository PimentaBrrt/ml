import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel="rbf", C=1, gamma="scale")

svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Acurácia do SVM:", acc)

print("\nRelatório de Classificação:\n")

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

print(report_df.round(2).to_markdown())