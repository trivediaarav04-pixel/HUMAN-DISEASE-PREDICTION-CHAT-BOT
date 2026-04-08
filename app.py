import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import streamlit_option_menu as som
import os
import time


st.set_page_config(page_title="Disease Predictor", page_icon="🩺", layout="wide")

script_dir   = os.path.dirname(os.path.abspath(__file__))
model_path   = os.path.join(script_dir, 'model_RFC.joblib')
dataset_path = os.path.join(script_dir, 'Dataset', 'dataset.csv')
severity_path= os.path.join(script_dir, 'Dataset', 'Symptom-severity.csv')
contrib_log  = os.path.join(script_dir, 'contributions_log.csv')


@st.cache_data
def read_csv_safely(path):
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception as e:
        return None, f"Could not read {path}: {e}"

@st.cache_data
def load_severity():
    df, err = read_csv_safely(severity_path)
    if err or df is None:
        return None, err
    df['Symptom'] = df['Symptom'].astype(str).str.replace("_", " ").str.strip()
    # ensure numeric weights
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(1).astype(int)
    return df, None

@st.cache_resource
def load_or_init_model():
    if os.path.exists(model_path):
        try:
            return jb.load(model_path), True
        except Exception:
            pass
    return RandomForestClassifier(n_estimators=200, random_state=42), False

def encode_symptoms(symptoms_input, severity_df):
    sym = severity_df['Symptom'].values
    weight = severity_df['weight'].values
    encoded = []
    for s in symptoms_input:
        if s and s in sym:
            encoded.append(int(weight[np.where(sym == s)[0][0]]))
        else:
            encoded.append(0)
    return encoded

def predict_once(symptoms_input, model, severity_df):
    encoded = encode_symptoms(symptoms_input, severity_df)
    pred = model.predict([encoded])[0]
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba([encoded])[0]
        classes = model.classes_
        prob_dict = {cls: float(prob[i]) for i, cls in enumerate(classes)}
    else:
        prob_dict = {}
    return pred, prob_dict, encoded

def persist_df(df, path):
    df.to_csv(path, index=False)

def log_contribution(row_dict):
    df = pd.DataFrame([row_dict])
    if os.path.exists(contrib_log):
        old = pd.read_csv(contrib_log)
        df = pd.concat([old, df], ignore_index=True)
    persist_df(df, contrib_log)


severity, sev_err = load_severity()
if sev_err:
    st.sidebar.error(sev_err)
symptoms_list = severity['Symptom'].tolist() if severity is not None else []

model_RFC, model_loaded = load_or_init_model()

with st.sidebar:
    selected_option = som.option_menu(
        'Disease Prediction System Based on Symptoms',
        options=['Prediction', 'Add Data', 'Train Model'],
        icons=['hospital', 'database-fill-add', 'train-front'],
        menu_icon='bandaid'
    )


def show_top3_cards(prob_dict):
    if not prob_dict:
        return
    top3 = pd.Series(prob_dict).sort_values(ascending=False).head(3)
    c1, c2, c3 = st.columns(3)
    for (lab, val), c in zip(top3.items(), [c1, c2, c3]):
        with c:
            st.metric(label=f"Top • {lab}", value=f"{val*100:.1f}%")

def simple_why(encoded_vector, model, input_symptoms):
    # Quick-and-friendly "why": multiply model feature importance by your nonzero inputs
    # (not exact SHAP, but intuitive)
    if not hasattr(model, "feature_importances_"):
        return None
    fi = model.feature_importances_
    scores = []
    for i, v in enumerate(encoded_vector):
        if v > 0:
            scores.append((i, v * fi[i]))
    scores.sort(key=lambda x: x[1], reverse=True)
    items = []
    for idx, score in scores[:5]:
        sym = input_symptoms[idx]
        if sym:
            items.append((sym, score))
    return items

if selected_option == 'Prediction':
    if not model_loaded:
        st.warning("⚠️ Model not found! Go to **Train Model** tab to train & save it first.")
        st.stop()

    st.header('Disease Prediction System based on Symptoms')

    with st.expander("⌨️ Type symptoms (comma-separated)", expanded=False):
        typed = st.text_input(
            "Example: fever, headache, nausea",
            key="typed_symptoms",
            help="We'll auto-normalize spaces/underscores. Unknown symptoms can be added to severity below."
        )
        default_new_weight = st.slider("Default weight for NEW typed symptoms", 1, 10, 2)

        add_unknowns = st.checkbox("Add unknown typed symptoms to severity database automatically", value=True,
                                   help="If a typed symptom is not in severity, we will add it with the default weight above.")

    typed_syms = []
    if typed.strip():
        typed_syms = [s.strip().lower().replace("_", " ") for s in typed.split(",") if s.strip()]

    # ensure severity has any brand-new typed symptoms if user opted in
    if typed_syms and add_unknowns and severity is not None:
        sev_lower = severity['Symptom'].str.lower().tolist()
        new_rows = []
        for s in typed_syms:
            if s not in sev_lower:
                new_rows.append({"Symptom": s, "weight": default_new_weight})
        if new_rows:
            severity = pd.concat([severity, pd.DataFrame(new_rows)], ignore_index=True)
            severity = severity.drop_duplicates(subset=['Symptom'], keep='last').reset_index(drop=True)
            persist_df(severity.assign(Symptom=severity['Symptom'].str.replace(" ", "_")), severity_path)  # save with underscores like original
            st.success(f"✅ Added {len(new_rows)} new symptom(s) to severity. (Weight={default_new_weight})")
            # refresh cached severity & list
            load_severity.clear()
            severity, _ = load_severity()
            symptoms_list = severity['Symptom'].tolist()

    # --- B) Dropdown symptoms (same layout: 17 slots in 3 columns) ---
    col1, col2, col3 = st.columns(3)
    # Provide '' option as empty + searchable selectboxes
    opts = [''] + symptoms_list

    with col1:
        Sym_1  = st.selectbox('Symptom 1',  opts, index=0, key="s1")
        Sym_4  = st.selectbox('Symptom 4',  opts, index=0, key="s4")
        Sym_7  = st.selectbox('Symptom 7',  opts, index=0, key="s7")
        Sym_10 = st.selectbox('Symptom 10', opts, index=0, key="s10")
        Sym_13 = st.selectbox('Symptom 13', opts, index=0, key="s13")
        Sym_16 = st.selectbox('Symptom 16', opts, index=0, key="s16")
    with col2:
        Sym_2  = st.selectbox('Symptom 2',  opts, index=0, key="s2")
        Sym_5  = st.selectbox('Symptom 5',  opts, index=0, key="s5")
        Sym_8  = st.selectbox('Symptom 8',  opts, index=0, key="s8")
        Sym_11 = st.selectbox('Symptom 11', opts, index=0, key="s11")
        Sym_14 = st.selectbox('Symptom 14', opts, index=0, key="s14")
        Sym_17 = st.selectbox('Symptom 17', opts, index=0, key="s17")
    with col3:
        Sym_3  = st.selectbox('Symptom 3',  opts, index=0, key="s3")
        Sym_6  = st.selectbox('Symptom 6',  opts, index=0, key="s6")
        Sym_9  = st.selectbox('Symptom 9',  opts, index=0, key="s9")
        Sym_12 = st.selectbox('Symptom 12', opts, index=0, key="s12")
        Sym_15 = st.selectbox('Symptom 15', opts, index=0, key="s15")

    dropdown_syms = [Sym_1, Sym_2, Sym_3, Sym_4, Sym_5, Sym_6, Sym_7, Sym_8, Sym_9, Sym_10, Sym_11, Sym_12, Sym_13, Sym_14, Sym_15, Sym_16, Sym_17]
    dropdown_syms = [s for s in dropdown_syms if s]  # drop empty

    # Merge typed + dropdown, de-dup, cap at 17
    merged_syms = []
    for s in typed_syms + dropdown_syms:
        if s and s not in merged_syms:
            merged_syms.append(s)
    if len(merged_syms) > 17:
        st.warning(f"You selected/typed {len(merged_syms)} symptoms. Trimming to 17.")
        merged_syms = merged_syms[:17]

    # Build fixed length 17 vector
    data_input = merged_syms + [""] * (17 - len(merged_syms))

    # Preview
    with st.expander("📋 Current Input Summary", expanded=False):
        if merged_syms:
            for i, s in enumerate(merged_syms, 1):
                st.write(f"- Symptom {i}: {s}")
        else:
            st.write("No symptoms provided yet.")

    # Predict
    if st.button('Make Prediction'):
        if severity is None:
            st.error("Severity file missing or unreadable. Please fix Dataset/Symptom-severity.csv")
            st.stop()

        pred, prob_dict, encoded = predict_once(data_input, model_RFC, severity)
        st.success(f"Predicted Disease: **{pred}**")

        # top-3 cards + bar chart (top 5)
        show_top3_cards(prob_dict)
        if prob_dict:
            prob_series = pd.Series(prob_dict).sort_values(ascending=False).head(5)
            st.bar_chart(prob_series)

        # quick “why” panel
        why = simple_why(encoded, model_RFC, data_input)
        if why:
            with st.expander("🧠 Why this prediction? (top contributing symptoms)", expanded=False):
                for sym, score in why:
                    st.write(f"- **{sym}** ↦ contribution score: `{score:.4f}`")

        # Allow download of current inputs
        out_df = pd.DataFrame([data_input], columns=[f"Symptom_{i}" for i in range(1, 18)])
        st.download_button("⬇️ Download this symptom row", out_df.to_csv(index=False), "prediction_input.csv")


elif selected_option == 'Add Data':
    st.title('Your Contribution is Valuable!')
    st.write('##### Provide data here')
    st.info("Pick symptoms or type them. Unknown typed symptoms can be auto-added to severity.")

    label = st.text_input('Disease Label')

    # allow typing too
    typed2 = st.text_input("Type symptoms (comma-separated)", key="typed_add",
                           placeholder="fever, headache, nausea")
    default_weight_add = st.slider("Default weight for NEW symptoms (Add Data)", 1, 10, 2)
    auto_add = st.checkbox("Auto-add new typed symptoms to severity", value=True)

    # selectboxes (same UX)
    col1, col2, col3 = st.columns(3)
    opts = [''] + symptoms_list
    with col1:
        Sym_1  = st.selectbox('Symptom 1',  opts, index=0, key="a1")
        Sym_4  = st.selectbox('Symptom 4',  opts, index=0, key="a4")
        Sym_7  = st.selectbox('Symptom 7',  opts, index=0, key="a7")
        Sym_10 = st.selectbox('Symptom 10', opts, index=0, key="a10")
        Sym_13 = st.selectbox('Symptom 13', opts, index=0, key="a13")
        Sym_16 = st.selectbox('Symptom 16', opts, index=0, key="a16")
    with col2:
        Sym_2  = st.selectbox('Symptom 2',  opts, index=0, key="a2")
        Sym_5  = st.selectbox('Symptom 5',  opts, index=0, key="a5")
        Sym_8  = st.selectbox('Symptom 8',  opts, index=0, key="a8")
        Sym_11 = st.selectbox('Symptom 11', opts, index=0, key="a11")
        Sym_14 = st.selectbox('Symptom 14', opts, index=0, key="a14")
        Sym_17 = st.selectbox('Symptom 17', opts, index=0, key="a17")
    with col3:
        Sym_3  = st.selectbox('Symptom 3',  opts, index=0, key="a3")
        Sym_6  = st.selectbox('Symptom 6',  opts, index=0, key="a6")
        Sym_9  = st.selectbox('Symptom 9',  opts, index=0, key="a9")
        Sym_12 = st.selectbox('Symptom 12', opts, index=0, key="a12")
        Sym_15 = st.selectbox('Symptom 15', opts, index=0, key="a15")

    dropdown_syms = [Sym_1, Sym_2, Sym_3, Sym_4, Sym_5, Sym_6, Sym_7, Sym_8, Sym_9,
                     Sym_10, Sym_11, Sym_12, Sym_13, Sym_14, Sym_15, Sym_16, Sym_17]
    dropdown_syms = [s for s in dropdown_syms if s]

    typed_list = []
    if typed2.strip():
        typed_list = [s.strip().lower().replace("_", " ") for s in typed2.split(",") if s.strip()]

    # auto-add brand-new typed symptoms to severity
    if typed_list and auto_add and severity is not None:
        sev_lower = severity['Symptom'].str.lower().tolist()
        new_rows = []
        for s in typed_list:
            if s not in sev_lower:
                new_rows.append({"Symptom": s, "weight": default_weight_add})
        if new_rows:
            severity = pd.concat([severity, pd.DataFrame(new_rows)], ignore_index=True)
            severity = severity.drop_duplicates(subset=['Symptom'], keep='last').reset_index(drop=True)
            persist_df(severity.assign(Symptom=severity['Symptom'].str.replace(" ", "_")), severity_path)
            st.success(f"✅ Added {len(new_rows)} new symptom(s) to severity.")
            load_severity.clear()
            severity, _ = load_severity()
            symptoms_list = severity['Symptom'].tolist()

    merged_syms = []
    for s in typed_list + dropdown_syms:
        if s and s not in merged_syms:
            merged_syms.append(s)
    if len(merged_syms) > 17:
        st.warning(f"Selected/typed {len(merged_syms)} symptoms. Trimming to 17.")
        merged_syms = merged_syms[:17]
    data_row = merged_syms + [""] * (17 - len(merged_syms))

    # summary
    if label or merged_syms:
        with st.expander("📋 Current Input Summary"):
            st.write(f"**Disease:** {label}")
            if merged_syms:
                for i, s in enumerate(merged_syms, 1):
                    st.write(f"- Symptom {i}: {s}")
            else:
                st.write("No symptoms yet.")

    def add_data_row(label, row17):
        if not os.path.exists(dataset_path):
            st.error(f"Dataset file not found at: {dataset_path}")
            return False
        try:
            dataset = pd.read_csv(dataset_path)
            df = pd.DataFrame([[label] + row17], columns=dataset.columns)
            dataset = pd.concat([dataset, df], ignore_index=True)
            persist_df(dataset, dataset_path)
            st.success("✅ Data insertion completed successfully. Thank you!")
            st.dataframe(df)
            log_contribution({"type":"add_data","label":label,"symptoms":"; ".join([s for s in row17 if s])})
            return True
        except Exception as e:
            st.error(f"Error adding data to dataset: {e}")
            return False

    if st.button("Submit"):
        if not label.strip():
            st.error("Please enter a disease label")
        elif not any(s for s in data_row):
            st.error("Please enter at least one symptom")
        else:
            if add_data_row(label, data_row):
                st.balloons()

    # download/upload helpers
    st.divider()
    if os.path.exists(dataset_path):
        ds, _ = read_csv_safely(dataset_path)
        if ds is not None:
            st.download_button("⬇️ Download current dataset.csv", ds.to_csv(index=False), "dataset.csv")

    uploaded = st.file_uploader("Or upload a NEW dataset.csv to overwrite", type=["csv"])
    if uploaded is not None:
        try:
            newds = pd.read_csv(uploaded)
            persist_df(newds, dataset_path)
            st.success("✅ Dataset replaced.")
            read_csv_safely.clear()
        except Exception as e:
            st.error(f"Upload failed: {e}")


elif selected_option == 'Train Model':
    st.title('Model Training Page')
    st.header("Train the Model")
    st.write("Click the button to start training the model")

    def train_then_save():
        if not os.path.exists(dataset_path):
            st.error(f"Dataset file not found at: {dataset_path}")
            return None, None
        if severity is None:
            st.error("Severity file missing or unreadable.")
            return None, None
        try:
            dataset = pd.read_csv(dataset_path)
            # normalize object columns
            for col in dataset.columns:
                if dataset[col].dtype == 'object':
                    dataset[col] = dataset[col].astype(str).str.replace('_', ' ').str.strip()
            dataset.fillna(0, inplace=True)

            vals = dataset.values
            symp = severity['Symptom'].unique()
            cols = dataset.columns

            # map names -> weights
            for i in range(len(symp)):
                vals[vals == symp[i]] = severity[severity['Symptom'] == symp[i]]['weight'].values[0]

            df = pd.DataFrame(vals, columns=cols)
            # clean some known junk (keep from your original)
            df = df.replace(['spotting  urination', 'dischromic  patches', 'foul smell of urine'], 0)

            for col in df.columns[1:]:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            X = df.iloc[:, 1:].values
            y = df['Disease'].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            model_RFC.fit(X_train, y_train)
            jb.dump(model_RFC, model_path)

            pred = model_RFC.predict(X_test)
            Acc = accuracy_score(y_test, pred)

            # feature importance table
            fi = None
            if hasattr(model_RFC, "feature_importances_"):
                fi = pd.DataFrame({
                    "feature": [f"Symptom_{i}" for i in range(1, 18)],
                    "importance": model_RFC.feature_importances_
                }).sort_values("importance", ascending=False)

            return Acc, fi
        except Exception as e:
            st.error(f"Error during training: {e}")
            return None, None

    if st.button("Start Training"):
        with st.spinner("Training model..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)
            Acc, fi = train_then_save()

        if Acc is not None:
            st.success(f"✅ Model trained successfully with accuracy: {Acc*100:.2f}%")
            st.metric("Accuracy", f"{Acc*100:.2f}%")
            if fi is not None and not fi.empty:
                with st.expander("🔍 Feature importance (global)", expanded=False):
                    st.dataframe(fi)
            st.balloons()

            # downloads
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    st.download_button("⬇️ Download model_RFC.joblib", f, file_name="model_RFC.joblib")
            if os.path.exists(dataset_path):
                ds, _ = read_csv_safely(dataset_path)
                if ds is not None:
                    st.download_button("⬇️ Download dataset.csv", ds.to_csv(index=False), "dataset.csv")

            st.rerun()
        else:
            st.error("❌ Model training failed. Please check your data files.")
 
 