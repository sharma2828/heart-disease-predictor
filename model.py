import pandas as pd
import numpy as np
df=pd.read_csv('heart_disease_uci.csv')
df.replace('?', np.nan, inplace=True)      #Handle '?' values
df = df.drop(columns=['id'])      #dropping useless columns 
df = df.dropna(subset=['num'])      #dropping rows where target is missing

#Convert target to binary
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df = df.drop(columns=['num'])

df = pd.get_dummies(df, columns=['dataset'], drop_first=True)      #one hot encoding

#defining numerical and categorical columns 
num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

#CONVERT NUMERICAL COLUMNS TO FLOAT
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

#FILL MISSING VALUES
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

#ENCODE CATEGORICAL VARIABLES
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
df['cp'] = df['cp'].map({
    'typical angina': 0,
    'atypical angina': 1,
    'non-anginal': 2,
    'asymptomatic': 3
})
df['fbs'] = df['fbs'].map({True: 1, False: 0})
df['restecg'] = df['restecg'].map({
    'normal': 0,
    'st-t abnormality': 1,
    'lv hypertrophy': 2
})
df['exang'] = df['exang'].map({True: 1, False: 0})
df['slope'] = df['slope'].map({
    'upsloping': 0,
    'flat': 1,
    'downsloping': 2
})
df['thal'] = df['thal'].map({
    'normal': 0,
    'fixed defect': 1,
    'reversable defect': 2
})
df = df.astype({col: 'int' for col in df.select_dtypes(include='bool').columns})


# Model Code:
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,     # 20% test data
    random_state=42    # for reproducibility
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nLogistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("\nRandom Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

print("\nFeature Importance:\n", feature_importance)

import pickle
# save model
pickle.dump(rf, open("model.pkl", "wb"))
# save scaler (if you used it)
pickle.dump(scaler, open("scaler.pkl", "wb"))