import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#load Csv
df  = pd.read_csv("churningdata.csv")

#drop the extra 
df = df.drop(columns=["customer_id"])

#targeting the churn 

target_col = "churn"

if df[target_col].dtype!= "int":
  df[target_col] = df[target_col].astype("int")

#spliting features and training 

X = df.drop(columns=[target_col])
Y = df[target_col]

#handling the missing values
num_cols = X.select_dtypes(include=["number","bool"]).columns
cat_cols = X.select_dtypes(include=["object","category"]).columns

X[num_cols]= X[num_cols].fillna(X[num_cols].median())
for c in cat_cols:
  X[c] = X[c].fillna(X[c].mode().iloc[0])


#One Hot encode
X = pd.get_dummies(X,columns=cat_cols,drop_first=True)

#train/test

X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size= 0.2, random_state=42, stratify=Y)




#scaler Function

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Randomforest Model training

model = RandomForestClassifier(
    n_estimators= 200,
    max_depth= 10,
    class_weight ="balanced",
    random_state = 42
)
model.fit(X_train,Y_train)


# Evaluate
Y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

#Feature importance
import numpy as np
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

print("\nTop features influencing churn:")
print(top_features)


