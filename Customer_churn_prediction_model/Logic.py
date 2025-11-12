
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


#csv def
df = pd.read_csv("churningdata.csv")

# specify the churn column

target_col = "churn"  

# 0 or 1 contained churn 
if df[target_col].dtype != "int":
    df[target_col] = df[target_col].astype("int")
    

#sperating the columns of churn from others
X = df.drop(columns=[target_col])
y = df[target_col]

# missed val handling 

num_cols = X.select_dtypes(include=["number", "bool"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

X[num_cols] = X[num_cols].fillna(X[num_cols].median())
for c in cat_cols:
    X[c] = X[c].fillna(X[c].mode().iloc[0])

# one hot encoding
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# train and test parameters for the model to understand the X and Y
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

   # Adding the scaler standard (Enhancement)
scaler = StandardScaler()

   # scale only the numeric columns (Enhancement)
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# training model with the data from X and Y
model = LogisticRegression(max_iter=1000,class_weight= "balanced")#added a class weight for balancing the churn detection so that the model can not skip the rep and also balancet the churn of all entity.  
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# See top features influencing churn (by absolute weight)
import numpy as np
feature_importance = pd.Series(model.coef_.ravel(), index=X.columns)
top_pos = feature_importance.sort_values(ascending=False).head(10)
top_neg = feature_importance.sort_values().head(10)

print("\nTop features increasing churn probability:")
print(top_pos)

print("\nTop features decreasing churn probability:")
print(top_neg)
