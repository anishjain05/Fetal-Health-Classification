import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
file_path = 'C:\\Users\\manis\\Downloads\\archive (1)\\fetal_health.csv'
fetal_data = pd.read_csv(file_path)

# Split the dataset into two non-overlapping parts, maintaining class proportionality
data_part1, data_part2 = train_test_split(fetal_data, test_size=0.5, random_state=42, stratify=fetal_data['fetal_health'])

def prepare_and_evaluate(data, title):
    print(f"--- {title} ---")

    # Standardizing the features
    scaler = StandardScaler()
    features = data.drop('fetal_health', axis=1)
    scaled_features = scaler.fit_transform(features)

    # Applying PCA
    pca = PCA(n_components=0.95)  # Keep 95% of the variance
    principal_components = pca.fit_transform(scaled_features)
    explained_variance_ratio = pca.explained_variance_ratio_

    # Plotting cumulative explained variance for PCA
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio))
    plt.title(f'Cumulative Explained Variance by PCA Components ({title})')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.grid(True)
    plt.show()

    # Correlation matrix of the original features
    corr_matrix = data.drop('fetal_health', axis=1).corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap='coolwarm', linewidths=.5)
    plt.title(f'Correlation Matrix of Features ({title})')
    plt.show()

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        principal_components,
        data['fetal_health'],
        test_size=0.3,
        random_state=42,
        stratify=data['fetal_health']
    )

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # K Nearest Neighbours
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
    print(classification_report(y_test, y_pred_knn))

    # Plotting the confusion matrix for KNN
    plt.figure(figsize=(8, 6))
    conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
    sns.heatmap(conf_matrix_knn, annot=True, fmt='g', cmap='Blues', xticklabels=['Normal', 'Suspect', 'Pathological'], yticklabels=['Normal', 'Suspect', 'Pathological'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix for KNN ({title})')
    plt.show()

    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_train, y_train)
    y_pred_rf = random_forest.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))

    # Plotting feature importances for Random Forest
    plt.figure(figsize=(12, 8))
    feature_importances = random_forest.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    plt.title(f'Feature Importances by Random Forest ({title})')
    plt.bar(range(len(indices)), feature_importances[indices], color='b', align='center')
    plt.xticks(range(len(indices)), [f'PC{i+1}' for i in indices], rotation=90)
    plt.xlabel('Principal Components')
    plt.ylabel('Importance')
    plt.show()

    # Gradient Boosting
    gradient_boosting = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gradient_boosting.fit(X_train, y_train)
    y_pred_gb = gradient_boosting.predict(X_test)
    print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
    print(classification_report(y_test, y_pred_gb))

    # Support Vector Machine
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
    print(classification_report(y_test, y_pred_svm))

    # Neural Network
    nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=300, random_state=42)
    nn_model.fit(X_train, y_train)
    y_pred_nn = nn_model.predict(X_test)
    print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))
    print(classification_report(y_test, y_pred_nn))

# Evaluate models on both parts of the dataset
prepare_and_evaluate(data_part1, "Dataset Part 1")
prepare_and_evaluate(data_part2, "Dataset Part 2")
