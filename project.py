import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix



data = pd.read_csv("Activity_Dataset_V1.csv")

print(data.shape)
print(data.dtypes)
print(data.isna().sum())
print("_" * 100)
print(data.isna().sum().sum())

data["total_steps"].fillna(data["total_steps"].mean(), inplace=True)
print(data.isna().sum())


print(data[["avg_pace","max_pace","min_pace"]].head(10))

change = ["avg_pace","max_pace","min_pace","activity_day","workout_type"]
for x in change:
    La = LabelEncoder()
    data[x] = La.fit_transform(data[x])
    
print(data.isna().sum())
print(data.dtypes)
print(data[["avg_pace","max_pace","min_pace"]].head())

plt.figure(figsize=(14,8))
for (x,y) in zip(data.columns.values,range(data.shape[1])):
    plt.subplot(3,7,y+1)
    sns.distplot(data[x])
plt.show()

plt.figure(figsize=(14,8))
sns.heatmap(data.corr(), annot=True, cmap="hot")
plt.show()

for x in data.columns.values:
    print(x)
    print(data[x].value_counts())
    print("_" * 100)

col = ["anaerobic(%)", "aerobic(%)","max_heart_rate","min_pace","max_pace","avg_pace","total_steps","workout_type"]

plt.figure(figsize=(16,8))
for x,y in zip(col,range(len(col))):
    plt.subplot(2,4,y+1)
    sns.countplot(data[x])
plt.show()

print(data[["calories"]].head(10))
print(data[["calories"]].describe())

data["calories_classes"] = 1
data.loc[data["calories"] > 275, "calories_classes"] = 2 
print(data["calories_classes"].value_counts())

plt.figure(figsize=(14,8))
sns.heatmap(data.corr(), annot=True, cmap="hot")
plt.show()

x = data.drop("calories_classes", axis=1)
y = data["calories_classes"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle =True)
print(X_train.shape)



DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)

print("_"*100)
print(DT.score(X_train, y_train))
print(DT.score(X_test, y_test))
print(DT.feature_importances_)
print("_"*100)
y_pred = DT.predict(X_test)

# confusion_matrix
Cm = confusion_matrix(y_test,y_pred)
print(Cm)
sns.heatmap(Cm,annot=True, fmt="d", cmap="magma")
plt.show()

# The autput result
result = pd.DataFrame({"y_test":y_test, "y_pred":y_pred})
# result.to_csv("The autput.csv",index=False)