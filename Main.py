# ---------------------------------------------- Section 1 -------------------------------------------------------------
# -------------------------------------- Data Collection and Preprocessing ---------------------------------------------
# ------------------------------------------- Import Libraries ---------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------- Load Dataset ----------------------------------------------------------------
student_data = pd.read_csv("Student_performance_data _.csv")
print(student_data.head())
print(student_data.info())

# ------------------------------------------ Handle Missing Values -----------------------------------------------------
numeric_features = student_data.select_dtypes(include=['number']).columns
student_data[numeric_features] = student_data[numeric_features].fillna(student_data[numeric_features].mean())
student_data.fillna(0, inplace=True)

print("\nMissing Values After Imputation:")
print(student_data.isnull().sum())


# ----------------------------------------------------- Outliers ------------------------------------------------------
def detect_outliers_iqr(stdata, column):
    # Calculate IQR
    Q1 = stdata[column].quantile(0.25)
    Q3 = stdata[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return stdata[(stdata[column] < lower_bound) | (stdata[column] > upper_bound)]

    numeric_columns = ['Music', 'Extracurricular']
    for col in numeric_columns:
        detect_outliers_iqr(student_data, col)
# ---------------------------- Encode Categorical Variables ------------------------------------------------------------
categorical_cols = student_data.select_dtypes(include=['object', 'category']).columns
label_encoders = {}
for col in categorical_cols:
    label_encoder = LabelEncoder()
    student_data[col] = label_encoder.fit_transform(student_data[col])
    label_encoders[col] = label_encoder

# ------------------------------------------- Drop irrelevant columns --------------------------------------------------
student_data = student_data.drop(columns=['StudentID', 'GPA'])

#-------------------------------------------------------Scale Features--------------------------------------------------
A = student_data.drop("GradeClass", axis=1)
B = student_data["GradeClass"]
scaler = StandardScaler()
A_scaled = scaler.fit_transform(A)

# ------------------------------------------------- Split Data ---------------------------------------------------------
A_train, A_test, B_train, B_test = train_test_split(A_scaled, B, test_size=0.3, random_state=42)

# ----------------------------------------------- Balancing class with SMOTE -------------------------------------------
smote = SMOTE(random_state = 42)
A_train_balanced, B_train_balanced = smote.fit_resample(A_train, B_train)

# ----------------------------- Plot class distribution after SMOTE ----------------------------------------------------
plt.figure(figsize=(8, 6))
sns.countplot(x=B_train_balanced, palette="viridis")
plt.title("Class Distribution After SMOTE")
plt.xlabel("GradeClass")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
# --------------------------------------------- Section 2 --------------------------------------------------------------
# ---------------------------------------- Model Selection and Training ------------------------------------------------
# ------------------------------------------------ Initialize models ---------------------------------------------------
# These models are the best suited for this Classification tasks
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=8, class_weight='balanced')
lr_model = LogisticRegression(random_state=42, max_iter=500, solver='liblinear', multi_class='ovr')

# ------------------------------------------------ Train models --------------------------------------------------------
rf_model.fit(A_train_balanced, B_train_balanced)
lr_model.fit(A_train_balanced, B_train_balanced)


# ------------------------------------- Hyperparameter Tuning for Logistic Regression ----------------------------------
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularization strength
    'solver': ['lbfgs', 'liblinear']  # Solvers for optimization
}

# Perform Grid Search with cross-validation
lr_grid_search = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid=lr_param_grid, cv=5, n_jobs=-1)

# Fit the model to the balanced training data
lr_grid_search.fit(A_train_balanced, B_train_balanced)

# Get the best parameters from GridSearchCV
print("Best parameters for Logistic Regression:", lr_grid_search.best_params_)

# Use the best model
best_logistic_model = lr_grid_search.best_estimator_

# Save the best Logistic Regression model
joblib.dump(best_logistic_model, 'best_logistic_regression_model.pkl')

# Predict and Evaluate Logistic Regression
logistic_predictions = best_logistic_model.predict(A_test)

# Evaluate Tuned Logistic Regression Model
print("\nTuned Logistic Regression Performance")
logistic_accuracy = accuracy_score(B_test, logistic_predictions)
logistic_precision = precision_score(B_test, logistic_predictions, average="weighted")
logistic_recall = recall_score(B_test, logistic_predictions, average="weighted")
logistic_f1 = f1_score(B_test, logistic_predictions, average="weighted")

print(f"Accuracy: {logistic_accuracy:.2f}")
print(f"Precision: {logistic_precision:.2f}")
print(f"Recall: {logistic_recall:.2f}")
print(f"F1 Score: {logistic_f1:.2f}")
print("\nClassification Report:\n", classification_report(B_test, logistic_predictions))

# ---------------------------------- Hyperparameter Tuning for Random Forest -------------------------------------------
# Define hyperparameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [10, 20, None],  # Maximum depth of the trees
    'max_features': ['sqrt', 'log2', None]  # Number of features to consider for splitting nodes
}

# Perform Grid Search with cross-validation
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=rf_param_grid, cv=5, n_jobs=-1)

# Fit the model to the training data
rf_grid_search.fit(A_train_balanced, B_train_balanced)

# Get the best parameters from GridSearchCV
print("Best parameters for Random Forest:", rf_grid_search.best_params_)

# Train the best model from GridSearch
best_rf_model = rf_grid_search.best_estimator_

# Save the best Random Forest model
joblib.dump(best_rf_model, 'best_rf_model.pkl')

# Make predictions with the best model
best_rf_predictions = best_rf_model.predict(A_test)

# Evaluate the model's performance
best_rf_accuracy = accuracy_score(B_test, best_rf_predictions)
print(f"Best Random Forest Accuracy: {best_rf_accuracy:.2f}")

# --------------------------------------------- Training Random Forest with SMOTE --------------------------------------
# Model Training with SMOTE
random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
random_forest_model.fit(A_train_balanced, B_train_balanced)

# Save Model
joblib.dump(random_forest_model, "random_forest_model.pkl")

# Predict and Evaluate Random Forest
rf_predictions = random_forest_model.predict(A_test)

# Evaluate Random Forest Classifier
print("Random Forest Classifier Performance Base Model")
rf_accuracy = accuracy_score(B_test, rf_predictions)
rf_precision = precision_score(B_test, rf_predictions, average="weighted")
rf_recall = recall_score(B_test, rf_predictions, average="weighted")
rf_f1 = f1_score(B_test, rf_predictions, average="weighted")

print(f"Accuracy: {rf_accuracy:.2f}")
print(f"Precision: {rf_precision:.2f}")
print(f"Recall: {rf_recall:.2f}")
print(f"F1 Score: {rf_f1:.2f}")
print("\nClassification Report:\n", classification_report(B_test, rf_predictions))


# ------------------------------------------ Training Logistic Regression with SMOTE -----------------------------------
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(A_train_balanced, B_train_balanced)

# Save the base Logistic Regression model
joblib.dump(logistic_model, 'logistic_regression_model.pkl')

# Predict and Evaluate Logistic Regression
logistic_predictions = logistic_model.predict(A_test)

# Evaluate Logistic Regression Model
print("\nLogistic Regression Performance Base Model")
logistic_accuracy = accuracy_score(B_test, logistic_predictions)
logistic_precision = precision_score(B_test, logistic_predictions, average="weighted")
logistic_recall = recall_score(B_test, logistic_predictions, average="weighted")
logistic_f1 = f1_score(B_test, logistic_predictions, average="weighted")

print(f"Accuracy: {logistic_accuracy:.2f}")
print(f"Precision: {logistic_precision:.2f}")
print(f"Recall: {logistic_recall:.2f}")
print(f"F1 Score: {logistic_f1:.2f}")
print("\nClassification Report:\n", classification_report(B_test, logistic_predictions))


# ------------------------------------------------ Section 3 -----------------------------------------------------------
# ----------------------------------------- Prediction and Evaluation --------------------------------------------------
# -------------------------------------- Generate prediction -----------------------------------------------------------
rf_prediction = rf_model.predict(A_test)
lr_prediction = lr_model.predict(A_test)

# ------------------------------------------------ Evaluate Models -----------------------------------------------------
def evaluate_model(B_test, predictions, model_name):
    metrics = {
        "Accuracy": accuracy_score(B_test, predictions),
        "Precision": precision_score(B_test, predictions, average="weighted"),
        "Recall": recall_score(B_test, predictions, average="weighted"),
        "F1 Score": f1_score(B_test, predictions, average="weighted"),
    }
    print(f"\nPerformance of {model_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    print(f"\nClassification Report:\n{classification_report(B_test, predictions)}")
    print("-" * 50)
    return metrics


rf_metrics = evaluate_model(B_test, rf_prediction, "Random Forest")
lr_metrics = evaluate_model(B_test, lr_prediction, "Logistic Regression")
# ----------------------------------------- Compare Models -------------------------------------------------------------
print("\nModel Comparison Summary:")
comparison_df = pd.DataFrame([rf_metrics, lr_metrics], index=["Random Forest Classifier", "Logistic Regression"])
print(comparison_df)
# -------------------------------------------- Discussion of Results ---------------------------------------------------
if rf_metrics["F1 Score"] > lr_metrics["F1 Score"]:
    better_model = "Random Forest"
    reason = "Random Forest achieved higher F1 score, suggesting better balance between precision and recall. It also showed stronger performance in handling the dataset's complexity, likely due to its ability to model non-linear relationships and feature interactions effectively."
else:
    better_model = "Logistic Regression"
    reason = "Logistic Regression achieved a higher F1 score, indicating that it handled the classification task more effectively. Its simplicity and efficiency in finding decision boundaries for this dataset made it the better-performing model."

print(f"\nConclusion: The better-performing model is **{better_model}**.")
print(f"Reason: {reason}")
# -------------------------------------------- Section 4 ---------------------------------------------------------------
# ----------------------------------- Visualization and Insights -------------------------------------------------------
# -------------------------------------------Confusion Matrix for Logistic Regression-----------------------------------
cmatrix_lr = confusion_matrix(B_test, logistic_predictions)

# Use the unique values in 'GradeClass' for tick labels
class_labels = np.unique(B_test)

plt.figure(figsize=(10, 7))
sns.heatmap(cmatrix_lr, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels,  # Use the unique class labels
            yticklabels=class_labels)  # Use the unique class labels
plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("Actual Labels", fontsize=12)
plt.title("Confusion Matrix for Logistic Regression", fontsize=14, fontweight="bold")
plt.show()

# ---------------------------------- Confusion Matrix for Random Forest Classifier -------------------------------------
cmatrix_rf = confusion_matrix(B_test, rf_predictions)

plt.figure(figsize=(10, 7))
sns.heatmap(cmatrix_rf, annot=True, fmt='g', cmap='Blues',
            xticklabels=class_labels,  # Use the unique class labels
            yticklabels=class_labels)  # Use the unique class labels
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('Actual Labels', fontsize=12)
plt.title('Confusion Matrix for Random Forest Classifier', fontsize=14, fontweight="bold")
plt.show()
# ------------------------------- Feature Importance for Random Forest -------------------------------------------------
importances = random_forest_model.feature_importances_
feature_names = A.columns  # Original feature names before scaling

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# -------------------------------Plot Feature Importance For Random Forest ---------------------------------------------
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance for Random Forest', fontsize=16, fontweight='bold')
plt.xlabel('Importance', fontsize=14, fontweight='bold')
plt.ylabel('Feature', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
# ---------------------------------- Feature Importance for Logistic Regression ----------------------------------------
feature_names = student_data.columns[:-1]  # Exclude the target column (assuming it's the last column)
coefficients = lr_model.coef_[0]
# Calculate absolute values of coefficients for importance
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.abs(coefficients)  # Use absolute values for interpretability
}).sort_values(by='Importance', ascending=False)

# -------------------------------------- Plot Feature Importance for Logistic Regression -------------------------------
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance for Logistic Regression (Student Data)', fontsize=16, fontweight='bold')
plt.xlabel('Importance (Absolute Coefficient)', fontsize=14, fontweight='bold')
plt.ylabel('Feature', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# -------------------------------------- Performance Graph -------------------------------------------------------------

# Metrics for Random Forest
models = ['Random Forest']
accuracy = [rf_accuracy]
precision = [rf_precision]
recall = [rf_recall]
f1 = [rf_f1]

x = np.arange(len(models))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 7))

# ----------------------------- Plotting bars for each metric for Random Forest ----------------------------------------
rects1 = ax.bar(x - width*1.5, accuracy, width, label='Accuracy', color='skyblue', edgecolor='black', linewidth=1.2)
rects2 = ax.bar(x - width/2, precision, width, label='Precision', color='lightgreen', edgecolor='black', linewidth=1.2)
rects3 = ax.bar(x + width/2, recall, width, label='Recall', color='salmon', edgecolor='black', linewidth=1.2)
rects4 = ax.bar(x + width*1.5, f1, width, label='F1 Score', color='gold', edgecolor='black', linewidth=1.2)

# -------------------------------------------Add labels, title, and legend ---------------------------------------------
ax.set_ylabel('Scores', fontsize=14, fontweight='bold')
ax.set_title('Random Forest Classifier Performance', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, fontweight='bold')
ax.legend(fontsize=12)

# --------------------------------- Autolabel function to display the height of bars in bold ---------------------------
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',  # Annotate with bold text
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # Offset for visibility
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

# Apply autolabel to all bars
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

# Adjust layout for a cleaner look
fig.tight_layout()
plt.show()



