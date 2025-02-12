import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize

def main():
    st.title("Machine Learning Prototype for RF and SVM ")

    # --- Form for User Input ---
    with st.form("data_form"):
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        model_choice = st.selectbox("Select Model", ["Random Forest (RF)", "Support Vector Machine (SVM)"])
        feature_selection = st.radio("Feature Selection Method", ["None", "Wrapper RFE"])
        submitted = st.form_submit_button("Submit")

    # --- After Submission ---
    if submitted and uploaded_file is not None:
        # --- Load Dataset ---
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset Preview", df.head())

        # --- Define Target Column ---
        target_column = df.columns[-1]
        if df[target_column].dtype == 'object':
            df[target_column] = df[target_column].astype('category').cat.codes

        # --- Drop Non-Numeric Columns (except target) ---
        for col in df.columns[:-1]:
            if df[col].dtype == 'object':
                df.drop(columns=[col], inplace=True)

        # --- Dimensionality Check ---
        column_count = len(df.columns)
        is_high_dimensional = column_count > 500
        dimensionality_label = 'High-Dimensional' if is_high_dimensional else 'Low-Dimensional'
        st.write(f"### Dataset Type: {dimensionality_label}")
        st.write(f"### Total Columns: {column_count}")

        # --- Data Splitting based on Model and Dataset Type ---
        if is_high_dimensional:
            test_size = 0.35  # High-dimensional datasets
        else:
            # Small-dimensional dataset
            if model_choice == "Random Forest (RF)":
                test_size = 0.25  # RF on small-dimensional
            else:
                test_size = 0.45  # SVM on small-dimensional

        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # --- Feature Selection for High-Dimensional Dataset ---
        if is_high_dimensional and feature_selection == "Wrapper RFE":
            st.write("Performing feature selection on high-dimensional dataset...")
            correlation = pd.DataFrame(X_train).corrwith(y_train)
            top_features = correlation.abs().sort_values(ascending=False).head(500).index
            X_train = X_train[top_features]
            X_test = X_test[top_features]

        # --- Model Training ---
        if model_choice == "Random Forest (RF)":
            model = RandomForestClassifier(random_state=42)
        else:
            model = SVC(probability=True, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- Evaluation Metrics ---
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)

        # Display metrics
        st.write("### Evaluation Metrics")
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1 Score: {f1:.4f}")
        st.write(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

        # --- Confusion Matrix Visualization ---
        st.write("### Confusion Matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        st.pyplot(plt)

        # --- ROC AUC Curve Visualization ---
        try:
            classes = sorted(y_test.unique())
            if len(classes) > 2:
                # --- Multi-class ROC ---
                y_test_bin = label_binarize(y_test, classes=classes)
                if model_choice == "Support Vector Machine (SVM)":
                    y_score = model.decision_function(X_test)
                else:
                    y_score = model.predict_proba(X_test)

                plt.figure(figsize=(10, 6))
                for i, class_label in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc = roc_auc_score(y_test_bin[:, i], y_score[:, i])
                    plt.plot(fpr, tpr, label=f'Class {class_label} (AUC = {roc_auc:.2f})')

                plt.plot([0, 1], [0, 1], 'r--', label='Random Baseline')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='best')
                st.pyplot(plt)

            else:
                # --- Binary Classification ROC ---
                if model_choice == "Support Vector Machine (SVM)":
                    y_prob = model.decision_function(X_test)
                else:
                    y_prob = model.predict_proba(X_test)[:, 1]

                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = roc_auc_score(y_test, y_prob)

                plt.figure(figsize=(10, 6))
                plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Random Baseline')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='best')
                st.pyplot(plt)

        except Exception as e:
            st.error(f"Error generating ROC curve: {e}")

if __name__ == "__main__":
    main()