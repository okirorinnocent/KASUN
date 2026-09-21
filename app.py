import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="CASMI26 Molecule ID Pipeline",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Molecule ID & Mass Spectra Predictor")
st.write("Upload your training and testing Parquet files to train a Random Forest model and generate predictions.")

# --- 2. Constants ---
ID_COL = 'spectrum_id'
TARGET_COL = 'normalized_smiles'
NESTED_COLS = {'ms2_mzs', 'ms2_normalized_intensities', 'collision_energy_ev'}

# --- 3. Sidebar Configuration ---
st.sidebar.header("📁 Data Upload")
train_file = st.sidebar.file_uploader(
    "Upload Train Parquet File", type=["parquet"])
test_file = st.sidebar.file_uploader(
    "Upload Test Parquet File", type=["parquet"])
sample_sub_file = st.sidebar.file_uploader(
    "Upload Sample Submission (Optional)", type=["csv"])

st.sidebar.header("⚙️ Model Settings")
downsample_size = st.sidebar.number_input(
    "Max Train Rows (Downsample)", value=50000, step=5000, min_value=100)
n_estimators = st.sidebar.slider(
    "Number of Trees", min_value=10, max_value=100, value=20)
max_depth = st.sidebar.slider("Max Depth", min_value=3, max_value=20, value=10)

# --- 4. Main Execution Pipeline ---
if train_file and test_file:
    if st.button("🚀 Run Pipeline", type="primary"):
        try:
            with st.status("Processing Data & Training Model...", expanded=True) as status:

                # Step 1: Schema & Feature Discovery
                st.write("🔍 Discovering common parquet features...")
                train_schema = pq.read_schema(train_file).names
                test_schema = pq.read_schema(test_file).names

                common_features = [
                    col for col in train_schema
                    if col in test_schema and col not in NESTED_COLS and not col.startswith('__')
                ]

                if not common_features:
                    st.error(
                        "No matching non-nested features found between train and test datasets.")
                    st.stop()

                # Step 2: Data Loading
                st.write("📥 Loading data into memory...")
                # Reset file pointers to beginning before reading
                train_file.seek(0)
                test_file.seek(0)

                train_df = pd.read_parquet(
                    train_file, columns=common_features + [TARGET_COL])
                test_df = pd.read_parquet(test_file, columns=common_features)

                # Reset indices explicitly to prevent index mismatch issues
                train_df = train_df.reset_index(drop=True)
                test_df = test_df.reset_index(drop=True)

                # Downsample training data for memory safety
                if len(train_df) > downsample_size:
                    train_df = train_df.sample(
                        n=int(downsample_size), random_state=42).reset_index(drop=True)

                test_ids = test_df[ID_COL].copy(
                ) if ID_COL in test_df.columns else pd.Series(range(len(test_df)))
                model_features = [
                    col for col in common_features if col != ID_COL]

                # Step 3: Feature Preprocessing
                st.write("⚙️ Processing numerical and categorical variables...")

                # Handle Numerical Columns
                num_cols = train_df[model_features].select_dtypes(
                    include=['int64', 'float64', 'float32', 'int32']).columns.tolist()
                for col in num_cols:
                    med = train_df[col].median()
                    fill_val = med if pd.notna(med) else 0.0
                    train_df[col] = train_df[col].fillna(
                        fill_val).astype(np.float32)
                    test_df[col] = test_df[col].fillna(
                        fill_val).astype(np.float32)

                # Handle Categorical Columns (Fixed pd.concat InvalidIndexError)
                cat_cols = train_df[model_features].select_dtypes(
                    include=['object', 'category', 'string']).columns.tolist()
                for col in cat_cols:
                    le = LabelEncoder()
                    # ignore_index=True prevents InvalidIndexError caused by index collisions
                    combined_series = pd.concat(
                        [train_df[col].astype(str), test_df[col].astype(str)],
                        ignore_index=True
                    )
                    le.fit(combined_series)
                    train_df[col] = le.transform(
                        train_df[col].astype(str)).astype(np.int32)
                    test_df[col] = le.transform(
                        test_df[col].astype(str)).astype(np.int32)

                X = train_df[model_features].values.astype(np.float32)
                X_test = test_df[model_features].values.astype(np.float32)

                # Target Encoding
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(
                    train_df[TARGET_COL].astype(str))

                # Free up training memory before fitting
                del train_df
                gc.collect()

                # Step 4: Model Training
                st.write("🌲 Training Random Forest Classifier...")
                model = RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X, y)

                # Step 5: Predictions & Output Preparation
                st.write("🎯 Generating test set predictions...")
                test_pred_indices = model.predict(X_test)
                final_predictions = target_encoder.inverse_transform(
                    test_pred_indices)

                test_preds_df = pd.DataFrame({
                    ID_COL: test_ids,
                    TARGET_COL: final_predictions
                }).drop_duplicates(subset=[ID_COL], keep='first')

                # Format with sample submission template if supplied
                if sample_sub_file is not None:
                    sample_sub_file.seek(0)
                    sub = pd.read_csv(sample_sub_file)
                    if ID_COL in sub.columns:
                        sub = sub[[ID_COL]].merge(
                            test_preds_df, on=ID_COL, how='left')
                        fallback_value = final_predictions[0] if len(
                            final_predictions) > 0 else ""
                        sub[TARGET_COL] = sub[TARGET_COL].fillna(
                            fallback_value)
                    else:
                        sub[TARGET_COL] = final_predictions[:len(sub)]
                else:
                    sub = test_preds_df

                status.update(
                    label="Pipeline execution finished successfully!", state="complete", expanded=False)

            # --- 5. Display Results ---
            st.success("✅ `submission.csv` successfully generated!")
            st.subheader("Preview of Predictions")
            st.dataframe(sub.head(10), use_container_width=True)

            # Download CSV Button
            csv_data = sub.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download submission.csv",
                data=csv_data,
                file_name="submission.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f" An error occurred during execution: `{str(e)}`")

else:
    st.info("👈 Please upload the training and testing files in the sidebar to run the pipeline.")
