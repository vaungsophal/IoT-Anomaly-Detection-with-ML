import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Function to load and preprocess dataset
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    non_numeric_cols = df.select_dtypes(exclude=np.number).columns
    df = df.drop(columns=non_numeric_cols, errors='ignore')
    correlation_matrix = df.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
    df = df.drop(columns=to_drop, errors='ignore')
    numeric_df = df.select_dtypes(include=np.number)
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(numeric_df.median())
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(numeric_df)
    return df_scaled, numeric_df.columns

# Function to balance data with SMOTE
def balance_data(X, y):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)

# Streamlit app
st.title("Machine Learning Model Runner")
st.sidebar.header("Choose Model")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Load and preprocess data
    try:
        df_scaled, columns = load_and_preprocess_data(uploaded_file)
        st.write("### Data Preprocessed Successfully")
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        st.stop()

    # Detect anomalies and split data
    X = pd.DataFrame(df_scaled, columns=columns)
    y = np.random.choice([0, 1], size=len(X))  # Replace with actual labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, y_train = balance_data(X_train, y_train)

    # Model selection
    model_choice = st.sidebar.selectbox("Select a Model", ["SVM", "Random Forest", "Decision Tree", "KNN", "Naive Bayes"])

# Inside the 'Run Model' button block
if st.sidebar.button("Run Model"):
    try:
        model_filename = f"{model_choice}_model.pkl"  # Generate a filename for the model based on the model choice

        # Try loading the saved model first
        try:
            model = joblib.load(model_filename)  # Attempt to load the model
            st.write(f"Loaded {model_choice} model from file.")
        except FileNotFoundError:
            # If the model doesn't exist, create and train a new one
            if model_choice == "SVM":
                st.write("### Support Vector Machine (SVM)")
                model = SVC(kernel='linear', probability=True, random_state=0)
            elif model_choice == "Random Forest":
                st.write("### Random Forest")
                model = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
            elif model_choice == "Decision Tree":
                st.write("### Decision Tree")
                model = DecisionTreeClassifier(random_state=0)
            elif model_choice == "KNN":
                st.write("### K-Nearest Neighbors (KNN)")
                model = KNeighborsClassifier(n_neighbors=5)
            elif model_choice == "Naive Bayes":
                st.write("### Naive Bayes")
                model = GaussianNB()
            else:
                st.error("Invalid model choice")
                st.stop()

            # Train the model
            model.fit(X_train, y_train)
            
            # Save the trained model to disk
            joblib.dump(model, model_filename)  # Save the model using joblib
            st.write(f"Saved {model_choice} model to file.")

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write(f"**AUC:** {auc:.4f}")
        
        # Generate and display the enhanced classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.drop(columns=['support'])

        # Plotting the heatmap
        plt.figure(figsize=(10, 6))
        sns.set(font_scale=1.2)
        sns.heatmap(report_df.iloc[:-2, :].astype(float), annot=True, cmap='Blues', fmt='.2f', cbar=False,
                    xticklabels=report_df.columns, yticklabels=report_df.index)
        plt.title('Classification Report Heatmap')
        st.pyplot(plt.gcf())
        plt.clf()

        # Display the classification report as text
        st.write("### Classification Report (Text Format)")
        st.text(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Anomalous'], yticklabels=['Benign', 'Anomalous'])
        st.pyplot(plt.gcf())
        plt.clf()

        # ROC Curve
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{model_choice} (AUC: {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        st.pyplot(plt.gcf())
        plt.clf()

    except Exception as e:
        st.error(f"Error while running model: {e}")
else:
    st.write("Upload a dataset to start.")
