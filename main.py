import pandas as pd

file_path = 'C:\\Users\\manis\\Downloads\\archive (1)\\fetal_health.csv'
fetal_data = pd.read_csv(file_path)

fetal_data.head(), fetal_data.describe(), fetal_data.info()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Plotting the distribution of the target variable 'fetal_health'
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(x='fetal_health', data=fetal_data, ax=ax[0])
ax[0].set_title('Distribution of Fetal Health Classes')
ax[0].set_xlabel('Fetal Health Class')
ax[0].set_ylabel('Count')

sns.histplot(fetal_data['baseline value'], kde=True, bins=30, ax=ax[1])
ax[1].set_title('Distribution of Baseline Fetal Heart Rate')
ax[1].set_xlabel('Baseline Fetal Heart Rate')
ax[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


#-------------------------------------
#-------------------------------------


# Correlation matrix
corr_matrix = fetal_data.corr()

plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix of Features')
plt.show()

#------------------------------------------------
#------------------------------------------------

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Standardizing the features (mean = 0 and variance = 1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(fetal_data.drop('fetal_health', axis=1))

# Applying PCA
pca = PCA(n_components=0.95)  # Keep 95% of the variance
principal_components = pca.fit_transform(scaled_features)
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio))
plt.title('Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.grid(True)
plt.show()

n_components_chosen = pca.n_components_
n_components_chosen

#-----------------------------------------------------
#-----------------------------------------------------

print("LOGISTIC REGRESSION")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    principal_components,
    fetal_data['fetal_health'],
    test_size=0.3,
    random_state=42,
    stratify=fetal_data['fetal_health']  # Ensuring proportional representation of classes
)

# Training logistic regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Predicting on the test set
y_pred = log_reg.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(accuracy, report)

#-------------------------------------------------------------
#-------------------------------------------------------------

print("K NEAREST NEIGHBOURS")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Evaluating the KNN model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
report_knn = classification_report(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

print("Accuracy of KNN:", accuracy_knn)
print("Classification Report for KNN:\n", report_knn)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn, annot=True, fmt='g', cmap='Blues', xticklabels=['Normal', 'Suspect', 'Pathological'], yticklabels=['Normal', 'Suspect', 'Pathological'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for KNN')
plt.show()

#--------------------------------------------------------
#----------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Training Random Forest model
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Predicting on the test set
y_pred_rf = random_forest.predict(X_test)

# Evaluating the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

print("RANDOM FOREST")

print(accuracy_rf, report_rf)

feature_importances = random_forest.feature_importances_
feature_names = ['PC{}'.format(i+1) for i in range(X_train.shape[1])]

# Sorting the features by importance
indices = np.argsort(feature_importances)[::-1]

# Creating the plot
plt.figure(figsize=(12, 8))
plt.title('Feature Importances by Random Forest')
plt.bar(range(len(indices)), feature_importances[indices], color='b', align='center')
plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
plt.xlim([-1, len(indices)])
plt.xlabel('Feature Names')
plt.ylabel('Importance')
plt.show()


# Training Gradient Boosting model
gradient_boosting = GradientBoostingClassifier(n_estimators=100, random_state=42)
gradient_boosting.fit(X_train, y_train)

# Predict on the test set
y_pred_gb = gradient_boosting.predict(X_test)

# Evaluating the Gradient Boosting model
accuracy_gb = accuracy_score(y_test, y_pred_gb)
report_gb = classification_report(y_test, y_pred_gb)

print("GRADIENT BOOSTING")

print(accuracy_gb, report_gb)

#--------------------------------------------------------
#------------------------------------------------------

print("SUPPORT VECTOR MACHINES")

from sklearn.svm import SVC

# Training the SVM model with RBF kernel
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred_svm = svm_model.predict(X_test)

# Evaluating the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)

print(accuracy_svm, report_svm)

#--------------------------------------------
#-------------------------------------------

print("NEURAL NETWORK")

from sklearn.neural_network import MLPClassifier

# Initialize the neural network model
nn_model = MLPClassifier(hidden_layer_sizes=(50, 100, 50), activation='relu', solver='adam',
                         max_iter=300, random_state=42)

# Train the neural network model
nn_model.fit(X_train, y_train)

# Predict on the test set
y_pred_nn = nn_model.predict(X_test)

# Evaluating the neural network model
accuracy_nn = accuracy_score(y_test, y_pred_nn)
report_nn = classification_report(y_test, y_pred_nn)

print(accuracy_nn, report_nn)

# print("FINE TUNED NEURAL NETWORK")
#
# # from sklearn.model_selection import GridSearchCV
#
# # Setting up parameters grid for hyperparameter tuning
# parameter_space = {
#     'hidden_layer_sizes': [(100, 50), (150, 100, 50), (200, 100)],
#     'max_iter': [500, 1000],  # Increased iterations
#     'learning_rate_init': [0.001, 0.01],  # Exploring different learning rates
# }
#
# # Create a MLPClassifier model for GridSearch
# nn_model_tuned = MLPClassifier(solver='adam', random_state=42)
#
# # Setting up GridSearchCV to find the best hyperparameters
# clf = GridSearchCV(nn_model_tuned, parameter_space, n_jobs=-1, cv=3, scoring='accuracy')
# clf.fit(X_train, y_train)
#
# # Best parameters found
# best_params = clf.best_params_
# best_score = clf.best_score_
#
# print(best_params, best_score)







