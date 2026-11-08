import os
import urllib.request
import zipfile
import warnings
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
DATA_FILE = os.path.join(DATA_DIR, "bank-full.csv")
MODEL_FILE = os.path.join(
    OUTPUT_DIR, "bank_marketing_classifier.joblib"
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAMPLE_SIZE = 5000
RANDOM_STATE = 42


def load_dataset():

    if not os.path.exists(DATA_FILE):

        print("Dataset not found. Downloading dataset...")
        dataset_url = (
            "https://archive.ics.uci.edu/static/public/222/"
            "bank+marketing.zip"
        )
        zip_file = os.path.join(
            DATA_DIR, "bank-marketing.zip"
        )
        urllib.request.urlretrieve(dataset_url, zip_file)
        with zipfile.ZipFile(zip_file, "r") as archive:
            archive.extractall(DATA_DIR)
        extracted_file = os.path.join(
            DATA_DIR, "bank", "bank-full.csv"
        )

        if not os.path.exists(extracted_file):
            raise FileNotFoundError(
                "Could not find bank-full.csv in the downloaded archive."
            )

        os.replace(extracted_file, DATA_FILE)

    df = pd.read_csv(DATA_FILE, sep=";")

    print("\n" + "=" * 60)
    print("BANK MARKETING CLASSIFICATION PROJECT")
    print("=" * 60)
    print("\nDataset loaded successfully.")
    return df

def select_sample(df):

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"\nOriginal dataset records: {len(df):,}")
    print(f"Original dataset columns: {len(df.columns)}")

    if len(df) > SAMPLE_SIZE:

        df, _ = train_test_split(
            df,
            train_size=SAMPLE_SIZE,
            random_state=RANDOM_STATE,
            stratify=df["y"]
        )

    df = df.reset_index(drop=True)

    print(f"Records selected: {len(df):,}")

    print("\nFirst five records:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nTarget distribution:")
    print(df["y"].value_counts())

    print("\nTarget percentages:")
    print(
        (df["y"].value_counts(normalize=True) * 100)
        .round(2)
        .astype(str) + "%"
    )
    return df


def prepare_data(df):
    print("\n" + "=" * 60)
    print("PREPARING DATA")
    print("=" * 60)
    X = df.drop(columns=["y", "duration"])
    y = df["y"].map({
        "no": 0,
        "yes": 1
    })

    print(f"\nRecords: {len(X):,}")
    print(f"Input features: {X.shape[1]}")
    print("\nTarget labels:")
    print("0 = Not subscribed")
    print("1 = Subscribed")
    return X, y


def split_data(X, y):

    print("\n" + "=" * 60)
    print("SPLITTING DATA")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\nTraining records: {len(X_train):,}")
    print(f"Testing records: {len(X_test):,}")
    return X_train, X_test, y_train, y_test

def build_model(X):

    print("\n" + "=" * 60)
    print("BUILDING CLASSIFICATION MODEL")
    print("=" * 60)

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    classifier = DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    print("\nDecision Tree classifier created.")

    return model


def train_model(model, X_train, y_train):
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    model.fit(X_train, y_train)
    print("\nModel training completed successfully.")
    return model

def evaluate_model(model, X_test, y_test):

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Not Subscribed",
                "Subscribed"
            ],
            zero_division=0
        )
    )
    print("\nConfusion Matrix:")
    print(cm)
    return y_pred, y_probability, accuracy, roc_auc, cm


def plot_target_distribution(df):

    print("\nCreating target distribution graph...")

    target_counts = (
        df["y"]
        .value_counts()
        .reindex(["no", "yes"], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        ["Not Subscribed", "Subscribed"],
        target_counts.values,
        width=0.6
    )

    ax.set_title(
        "Bank Marketing - Target Distribution",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Subscription Status")
    ax.set_ylabel("Number of Customers")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.bar_label(bars, fmt="%d", padding=3)

    fig.tight_layout()

    file_path = os.path.join(
        OUTPUT_DIR,
        "target_distribution.png"
    )

    fig.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )
    print(f"Saved: {file_path}")
    return fig

def plot_confusion_matrix(y_test, y_pred):
    print("Creating confusion matrix graph...")
    fig, ax = plt.subplots(figsize=(7, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0, 1],
        display_labels=[
            "Not Subscribed",
            "Subscribed"
        ],
        cmap="Blues",
        colorbar=False,
        values_format="d",
        ax=ax
    )

    ax.set_title(
        "Confusion Matrix",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    fig.tight_layout()

    file_path = os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    )

    fig.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )
    print(f"Saved: {file_path}")
    return fig

def plot_roc_curve(y_test, y_probability, roc_auc):

    print("Creating ROC curve graph...")

    fig, ax = plt.subplots(figsize=(7, 5))

    RocCurveDisplay.from_predictions(
        y_test,
        y_probability,
        name=f"Decision Tree (AUC = {roc_auc:.3f})",
        ax=ax
    )

    ax.set_title(
        "ROC Curve",
        fontsize=14,
        fontweight="bold"
    )

    ax.grid(linestyle="--", alpha=0.3)

    fig.tight_layout()

    file_path = os.path.join(
        OUTPUT_DIR,
        "roc_curve.png"
    )

    fig.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {file_path}")
    return fig

def display_graphs(df, y_test, y_pred, y_probability, roc_auc):

    print("\n" + "=" * 60)
    print("GENERATING THREE SEPARATE GRAPHS")
    print("=" * 60)

    fig1 = plot_target_distribution(df)
    fig2 = plot_confusion_matrix(
        y_test,
        y_pred
    )
    fig3 = plot_roc_curve(
        y_test,
        y_probability,
        roc_auc
    )

    print("\nOpening all three graph windows...")
    plt.show(block=True)
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)


def save_model(model):
    joblib.dump(model, MODEL_FILE)
    print("\nTrained model saved successfully:")
    print(MODEL_FILE)

def make_sample_prediction(model, X_test):
    print("\n" + "=" * 60)
    print("SAMPLE CUSTOMER PREDICTION")
    print("=" * 60)

    sample_customer = X_test.iloc[[0]]

    prediction = model.predict(sample_customer)[0]

    probability = model.predict_proba(
        sample_customer
    )[0, 1]

    print("\nSample customer features:")
    print(sample_customer.T)

    if prediction == 1:
        print("\nPrediction: Customer may subscribe.")
    else:
        print("\nPrediction: Customer may not subscribe.")

    print(
        f"Estimated subscription probability: "
        f"{probability * 100:.2f}%"
    )

def main():
    df = load_dataset()
    df = select_sample(df)
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = build_model(X)
    model = train_model(
        model,
        X_train,
        y_train
    )

    y_pred, y_probability, accuracy, roc_auc, cm = evaluate_model(
        model,
        X_test,
        y_test
    )

    display_graphs(
        df,
        y_test,
        y_pred,
        y_probability,
        roc_auc
    )

    save_model(model)
    make_sample_prediction(
        model,
        X_test
    )

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nRecords used: {len(df):,}")
    print(f"Training records: {len(X_train):,}")
    print(f"Testing records: {len(X_test):,}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"\nAll output files are in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()