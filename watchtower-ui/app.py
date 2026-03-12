import streamlit as st
import requests
import pandas as pd
import io
import json
import hashlib
import streamlit.components.v1 as components
from typing import Dict, Any

# --- Configuration ---
API_BASE_URL = "http://api:8000/api/v1"

# --- State Management ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'model_data' not in st.session_state:
    st.session_state.model_data = {}
if 'token' not in st.session_state:
    st.session_state.token = None
if 'drift_report_html' not in st.session_state:
    st.session_state.drift_report_html = None
if 'drift_status' not in st.session_state:
    st.session_state.drift_status = ""
if 'datasets_list' not in st.session_state:
    st.session_state.datasets_list = []
if 'selected_train_dataset_id' not in st.session_state:
    st.session_state.selected_train_dataset_id = None
if 'selected_mon_model_data' not in st.session_state:
    st.session_state.selected_mon_model_data = {} 


def set_auth_header():
    """Returns the Authorization header for protected requests."""
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}

# ==============================================================================
# 1. AUTHENTICATION (Register & Login)
# ==============================================================================

def register_user(username, email, password):
    url = f"{API_BASE_URL}/users/register"
    try:
        response = requests.post(url, json={"username": username, "email": email, "password": password})
        if response.status_code == 200:
            st.success("Registration successful! Please log in.")
        else:
            st.error(f"Registration failed: {response.json().get('detail', response.text)}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the WatchTower API.")

def get_user_data(token):
    """Fetches user data, including UUID, after successful login/token acquisition."""
    url = f"{API_BASE_URL}/users/me"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch user data: {response.json().get('detail', response.text)}")
        return None

def login_user(username, email, password):
    url = f"{API_BASE_URL}/users/token"
    try:
        response = requests.post(url, json={"username": username, "email": email, "password": password})
        if response.status_code == 200:
            data = response.json()
            
            st.session_state.token = data['access_token']
            user_data = get_user_data(st.session_state.token)

            if user_data:
                st.session_state.user_data = user_data 
                st.session_state.logged_in = True
                load_user_models_and_datasets()
                st.success(f"Login successful! Welcome, {user_data['username']}.")
                st.rerun()
            else:
                st.session_state.token = None 
                st.error("Login successful, but failed to retrieve profile data. Check API logs.")
        else:
            st.error(f"Login failed: {response.json().get('detail', 'Invalid credentials')}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the WatchTower API.")

# ==============================================================================
# 2. MODEL AND DATASET MANAGEMENT
# ==============================================================================

def upload_model_with_metadata(uploaded_file, user_id, framework, features_str, target_str, hyperparams: Dict[str, Any], model_class, framework_path, model_display_name):
    url = f"{API_BASE_URL}/models/upload"

    # Pass metadata as JSON string in params for simplicity
    params = {
        'framework': framework, 
        'owner_id': user_id,
        'features': features_str,
        'target': target_str,
        'hyperparams_json': json.dumps(hyperparams),
        'model_class': model_class,
        'framework_path': framework_path,
        'model_display_name': model_display_name
    }

    files = {'file': uploaded_file.getvalue()}
    
    try:
        response = requests.post(url, files=files, params=params, headers=set_auth_header())
        if response.status_code == 200:
            model_info = response.json()
            # Set the new model as the default active model for prediction
            print(f"response::: {model_info}")
            st.session_state.model_data = {
                "owner_id": model_info["id"],
                "sha_filename": f"{model_info['sha256']}_{model_info['name']}",
                "model_sha": model_info["sha256"],
                "model_id": str(model_info["id"])
            }
            load_user_models_and_datasets()
            st.success(f"Model '{model_info['name']}' uploaded and registered successfully!")
            st.info(f"Metadata captured. Model is now eligible for retraining.")
        else:
            detail = response.json().get('detail', response.text)
            st.error(f"Model upload failed: {detail}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the WatchTower API.")

def upload_reference_dataset(uploaded_file):
    url = f"{API_BASE_URL}/monitoring/reference"
    files = {
        'file': (
            uploaded_file.name, 
            uploaded_file.getvalue(), 
            uploaded_file.type 
        )
    }
    
    try:
        response = requests.post(url, files=files, headers=set_auth_header())
        if response.status_code == 201:
            st.success("Reference dataset uploaded successfully for monitoring!")
            load_reference_datasets()
            st.rerun() 
        else:
            detail = response.json().get('detail', response.text)
            st.error(f"Dataset upload failed: {detail}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the WatchTower API.")

def load_user_models():
    """Fetches the list of models uploaded by the current user."""
    url = f"{API_BASE_URL}/models/list"
    
    try:
        response = requests.get(url, headers=set_auth_header())
        if response.status_code == 200:
            models_list = response.json()
            st.session_state.models_list = models_list
        else:
            st.error(f"Failed to load model list: {response.json().get('detail', response.text)}")
            st.session_state.models_list = []
            
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the WatchTower API to load models.")
        st.session_state.models_list = []

# ==============================================================================
# 3. PREDICTION & MONITORING
# ==============================================================================

def load_reference_datasets():
    url = f"{API_BASE_URL}/monitoring/reference/list"
    try:
        response = requests.get(url, headers=set_auth_header())
        if response.status_code == 200:
            st.session_state.datasets_list = response.json()
        else:
            st.error(f"Failed to load dataset list: {response.json().get('detail', response.text)}")
            st.session_state.datasets_list = []
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the WatchTower API to load datasets.")
        st.session_state.datasets_list = []

def trigger_retrain(model_id: str):
    """Calls the FastAPI endpoint to trigger retraining."""
    url = f"{API_BASE_URL}/monitoring/retrain/{model_id}"
    try:
        response = requests.post(url, headers=set_auth_header())
        if response.status_code == 202:
            st.success("✅ Retraining job initiated successfully! Check MLflow UI for progress.")
        else:
            st.error(f"Retraining failed: {response.json().get('detail', response.text)}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the WatchTower API.")

def load_user_models_and_datasets(): # Combined loading function
    load_user_models()
    load_reference_datasets()

def run_prediction(input_list):
    # This uses the model currently selected in the Inference Tab (st.session_state.model_data)
    if not st.session_state.model_data.get('model_sha'):
        st.warning("Please select a model in the Inference tab.")
        return

    owner_id = st.session_state.model_data["owner_id"]
    sha_filename = st.session_state.model_data["sha_filename"]
    
    url = f"{API_BASE_URL}/models/predict/{owner_id}/{sha_filename}"
    
    try:
        response = requests.post(url, json={"input": input_list}, headers=set_auth_header())
        if response.status_code == 200:
            prediction = response.json().get('prediction')
            st.metric(label="Prediction Result", value=f"{prediction}")
            st.success("Prediction logged for drift monitoring.")
        else:
            detail = response.json().get('detail', response.text)
            st.error(f"Prediction failed: {detail}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the WatchTower API.")

def run_drift_check(features_string: str, dataset_id: str, model_sha: str):
    """Calls the FastAPI endpoint to generate the drift report."""
    if not features_string.strip():
        st.error("Feature Columns field cannot be empty.")
        st.session_state.drift_report_html = None
        return
        
    url = f"{API_BASE_URL}/monitoring/drift-check?features={features_string}&dataset_id={dataset_id}&model_sha={model_sha}"
    
    try:
        response = requests.get(url, headers=set_auth_header())
        if response.status_code == 200:
            st.session_state.drift_report_html = response.text
            st.session_state.drift_status = "success"
            st.toast("Drift analysis complete!", icon="✅")
            st.rerun()
        else:
            st.session_state.drift_report_html = None
            st.session_state.drift_status = "failed"
            try:
                error_detail = response.json().get('detail', 'Could not generate report.')
            except requests.exceptions.JSONDecodeError:
                error_detail = response.text[:200] + "..." 
            st.error(f"Drift check failed: {response.status_code}: {error_detail}")
            
    except requests.exceptions.ConnectionError:
        st.session_state.drift_report_html = None
        st.error("Cannot connect to the WatchTower API.")


# ==============================================================================
# UI Rendering
# ==============================================================================

st.set_page_config(layout="wide", page_title="WatchTower AI MLOps Dashboard")
st.title("🛡️ WatchTower AI MLOps Platform")

# --- Sidebar for Auth/Info ---
with st.sidebar:
    st.header("Status")
    if st.session_state.logged_in:
        st.success("Logged In")
        st.write(f"User: {st.session_state.user_data.get('email', 'N/A')}")
        st.write(f"User ID: `{st.session_state.user_data.get('id', 'N/A')}`")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.token = None
            st.session_state.user_data = {}
            st.session_state.model_data = {}
            st.rerun()
    else:
        st.error("Logged Out")

    st.markdown("---")
    st.subheader("External Tools")
    st.markdown("- [MLflow UI (Model Registry)](http://localhost:5000)")
    st.markdown("- [Prometheus UI (Metrics)](http://localhost:9090)")
    st.markdown("- [Grafana (Dashboards)](http://localhost:3000)")


def trigger_training_job(
    model_name, model_class, framework_path, hyperparams, features_string, target_string, dataset_id
):
    url = f"{API_BASE_URL}/training/train?features={features_string}&target={target_string}&dataset_id={dataset_id}" 
    
    payload = {
        "model_name": model_name,
        "model_class": model_class,
        "framework_path": framework_path,
        "hyperparams": hyperparams
    }
    
    try:
        response = requests.post(url, json=payload, headers=set_auth_header())
        if response.status_code == 202:
            st.success(f"Training job accepted! Check MLflow UI for results.")
        else:
            st.error(f"Training failed (HTTP {response.status_code}): {response.json().get('detail', response.text)}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the WatchTower API.")


# --- Main Content ---
if not st.session_state.logged_in:
    st.subheader("User Authentication")
    tab_reg, tab_log, = st.tabs(["Register", "Login"])
    
    with tab_reg:
        with st.form("register_form"):
            st.markdown("New User Registration")
            r_username = st.text_input("Username")
            r_email = st.text_input("Email", key="reg_email")
            r_password = st.text_input("Password", type="password", key="reg_pass")
            if st.form_submit_button("Register"):
                register_user(r_username, r_email, r_password)

    with tab_log:
        with st.form("login_form"):
            st.markdown("Existing User Login")
            l_username = st.text_input("Username", key="log_username")
            l_email = st.text_input("Email", key="log_email")
            l_password = st.text_input("Password", type="password", key="log_pass")
            if st.form_submit_button("Login"):
                if l_email == "prakash@example.com":
                    st.session_state.user_data['id'] = st.text_input("Enter Your Registered UUID (See FastAPI logs):", value="c0e6f6a7-0925-450f-a7f4-d6b9d6a3f123") 
                
                login_user(l_username,l_email, l_password)
                
else: # User is Logged In
    
    if 'models_list' not in st.session_state or 'datasets_list' not in st.session_state:
        load_user_models_and_datasets()

    st.header("Model Management & Deployment")
    
    # --- Model Upload takes full width ---
    st.subheader("1. Upload & Register Model (.pkl)")
    uploaded_model_file = st.file_uploader("Choose a .pkl model file", type=['pkl'])

    framework = st.text_input("Model Framework (e.g., sklearn)", value="sklearn")

    model_display_name = st.text_input(
        "Model Display Name (Required)", 
        value=uploaded_model_file.name if uploaded_model_file else "my-model", 
        key="up_model_display_name"
    )

    # --- NEW METADATA INPUTS ---
    up_feature_string = st.text_input("Feature Columns (e.g., col1, col2)", key="up_feature_string")
    up_target_string = st.text_input("Target Column Name", key="up_target_string")
    # NEW FIELDS ADDED HERE
    up_model_class = st.text_input("Model Class (e.g., LogisticRegression)", value="LogisticRegression", key="up_model_class")
    up_framework_path = st.text_input("Framework Module Path (e.g., sklearn.linear_model)", value="sklearn.linear_model", key="up_framework_path")
    up_hyperparams_json = st.text_area("Hyperparameters (JSON)", value="{}", key="up_hyperparams_json", height=70)
    # ---------------------------
    
    if uploaded_model_file and st.button("Upload Model to MinIO & Register"):
        if st.session_state.user_data.get('id'):
            try:
                hyperparams = json.loads(up_hyperparams_json)
                with st.spinner('Uploading...'):
                    # upload_model(uploaded_model_file, st.session_state.user_data['id'], framework)
                    # CALL THE UPDATED UPLOAD FUNCTION
                    print(f"up_feature_string:: {up_feature_string}, up_target_string:: {up_target_string}, hyperparams:: {hyperparams}")
                    upload_model_with_metadata(
                        uploaded_model_file, 
                        st.session_state.user_data['id'], 
                        framework, 
                        up_feature_string,
                        up_target_string,
                        hyperparams,
                        up_model_class, 
                        up_framework_path,
                        model_display_name
                    )
                    load_user_models()
            except json.JSONDecodeError:
                st.error("Invalid JSON format for Hyperparameters.")
        else:
             st.error("Please ensure your User ID (UUID) is set correctly.")
        
    st.markdown("---")
    
    # --- Tabbed Interface: Inference first, then Monitoring, then Training ---
    st.header("MLOps Lifecycle")
    tab_inf, tab_mon, tab_train = st.tabs(["Inference", "Model Monitoring (Evidently)", "Model Training & Management"])

    # ==============================================================================
    # TAB: INFERENCE (Model Selection and Prediction)
    # ==============================================================================
    with tab_inf:
        st.subheader("1. Select Model for Prediction")

        if st.session_state.get('models_list') and st.session_state.models_list:
            
            # model_options = {
            #     f"{m['name']} ({m['framework']} | SHA: {m['sha256'][:6]}...)": m 
            #     for m in st.session_state.models_list
            # }

            model_options = {
                f"{m['name']} (ID: {str(m['id'])[:4]}... | SHA: {m['sha256'][:6]}... | framework: {m['framework']})": m 
                for m in st.session_state.models_list
            }
            
            selected_model_key = st.selectbox(
                "Available Uploaded Models:", 
                list(model_options.keys()),
                key="inf_model_selector" 
            )
            
            selected_model = model_options[selected_model_key]

            # CRITICAL: Update the main model_data state which run_prediction uses
            st.session_state.model_data = {
                "owner_id": st.session_state.user_data['id'],
                "sha_filename": f"{selected_model['sha256']}_{selected_model['name']}",
                "model_sha": selected_model['sha256'],
                "model_id": str(selected_model['id']) 
            }
            
            st.info(f"Prediction Model Selected: **{selected_model['name']}**")
        else:
            st.warning("No models found. Please upload a model.")
            
        st.markdown("---")
        st.subheader("2. Run Prediction")
        
        if st.session_state.model_data.get('model_sha'):
            default_input = "[0.1, 0.2, 0.3, 0.4]" 
            input_text = st.text_area(
                "Input Features Array",
                value=default_input,
                height=100,
                key="prediction_input_area" 
            )

            if st.button("Predict"):
                try:
                    input_list = json.loads(input_text)
                    
                    if not isinstance(input_list, list):
                        st.error("Input must be a valid JSON array (list).")
                    elif not all(isinstance(x, (int, float)) for x in input_list):
                        st.error("Input array must contain only numbers (integers or floats).")
                    else:
                        with st.spinner('Calculating prediction...'):
                            run_prediction(input_list) 
                            
                except json.JSONDecodeError:
                    st.error("🚨 Invalid JSON format. Please check syntax. Example: [1.0, 2.0, 3.0]")
                    
        else:
            st.warning("Please select a model above.")


    # ==============================================================================
    # TAB: MONITORING (Dedicated Model Selection and Data Upload/Selection)
    # ==============================================================================
    with tab_mon:
        # --- Data Upload (New Location - DUPLICATED) ---
        st.subheader("1. Upload New Reference Data")
        
        uploaded_data_file = st.file_uploader("Choose a CSV for Monitoring/Reference", type=['csv'], key="mon_data_uploader")
        
        if uploaded_data_file and st.button(f"Upload Reference Dataset to MinIO (Monitoring)", key="mon_upload_btn"):
            with st.spinner('Uploading...'):
                upload_reference_dataset(uploaded_data_file)
        
        st.markdown("---")

        # --- Model Selection for Drift Check (Dedicated Dropdown) ---
        st.subheader("2. Select Model for Drift Monitoring")
        
        mon_model_sha = None
        
        if st.session_state.get('models_list') and st.session_state.models_list:
            
            # model_options = {
            #     f"{m['name']} ({m['framework']} | SHA: {m['sha256'][:6]}...)": m 
            #     for m in st.session_state.models_list
            # }

            model_options = {
                f"{m['name']} (ID: {str(m['id'])[:4]}... | SHA: {m['sha256'][:6]}... | framework: {m['framework']})": m 
                for m in st.session_state.models_list
            }
            
            selected_model_key = st.selectbox(
                "Available Uploaded Models:", 
                list(model_options.keys()),
                key="mon_model_selector"
            )
            
            selected_model = model_options[selected_model_key]

            # Update the DEDICATED monitoring state
            st.session_state.selected_mon_model_data = {
                "owner_id": st.session_state.user_data['id'],
                "sha_filename": f"{selected_model['sha256']}_{selected_model['name']}",
                "model_sha": selected_model['sha256'],
                "model_id": str(selected_model['id']) 
            }
            mon_model_sha = selected_model['sha256']
            st.info(f"Monitoring Model Selected: **{selected_model['name']}**")
        else:
            st.warning("No models found. Please upload a model.")
            
        st.markdown("---")
        st.subheader("3. Select Reference Dataset for Drift Check")
        
        selected_dataset_id = None
        
        if st.session_state.get('datasets_list'):
            dataset_options = {
                f"{d['name']} ({d['created_at'].split('T')[0]}) [ID: {d['id'][:4]}...]": d['id']
                for d in st.session_state.datasets_list
            }
            
            selected_dataset_key = st.selectbox(
                "Select Data for Drift Check:", 
                list(dataset_options.keys()),
                key="mon_dataset_selector"
            )
            selected_dataset_id = dataset_options[selected_dataset_key]
        else:
            st.warning("No reference datasets found. Please upload one above or in the Training tab.")
            
        st.markdown("---")
        
        st.subheader("Evidently Data Drift Analyzer")
        
        # --- Run Check Inputs ---
        
        if not mon_model_sha:
            st.error("Please select a model above.")
        elif not selected_dataset_id:
            st.error("Please select a reference dataset above.")
        
        
        feature_input = st.text_input(
            "Feature Columns (Comma-separated list)",
            value="sepal_length, sepal_width, petal_length, petal_width", 
            help="Enter the exact column names from your uploaded Reference CSV."
        )
        
        if st.button("Run Data Drift Check") and mon_model_sha and selected_dataset_id:
            with st.spinner("Analyzing data for drift..."):
                run_drift_check(
                    features_string=feature_input,
                    dataset_id=selected_dataset_id,
                    model_sha=mon_model_sha
                )

        # --- Report Visualization and Retraining Trigger ---
        if st.session_state.drift_report_html:
            report_html = st.session_state.drift_report_html
            
            st.markdown("---")
            st.subheader("Model Lifecycle Management")

            if st.button("🔥 Trigger Retraining Job (Use Current Model's Stored Params)"):
                if st.session_state.selected_mon_model_data.get('model_id'):
                    trigger_retrain(st.session_state.selected_mon_model_data['model_id'])
                else:
                    st.error("Cannot retrain: Model ID not found.")

            if report_html:
                st.subheader("📊 Model Data Drift Report (Evidently)")
                components.html(report_html, height=1000, scrolling=True)
                
                st.download_button(
                    label="Download Report as HTML",
                    data=report_html.encode('utf-8'),
                    file_name="evidently_drift_report.html",
                    mime="text/html"
                )

    # ==============================================================================
    # TAB: TRAINING (Data Upload and Selection)
    # ==============================================================================
    with tab_train:
        # --- Data Upload (New Location - DUPLICATED) ---
        st.subheader("1. Upload New Training Data")
        
        uploaded_data_file = st.file_uploader("Choose a CSV for Training", type=['csv'], key="train_data_uploader")
        
        if uploaded_data_file and st.button(f"Upload Training Dataset to MinIO", key="train_upload_btn"):
            with st.spinner('Uploading...'):
                upload_reference_dataset(uploaded_data_file)

        st.markdown("---")
        st.subheader("2. Select Training Data")
        
        selected_train_dataset_id = None
        if st.session_state.get('datasets_list'):
            dataset_options = {
                f"{d['name']} ({d['created_at'].split('T')[0]}) [ID: {d['id'][:4]}...]": d['id']
                for d in st.session_state.datasets_list
            }
            
            dataset_display_names = ["-- Select Dataset for Training --"] + list(dataset_options.keys())
            
            selected_dataset_key = st.selectbox(
                "Training Dataset Source:", 
                dataset_display_names,
                key="train_dataset_selector"
            )
            
            if selected_dataset_key != "-- Select Dataset for Training --":
                selected_train_dataset_id = dataset_options[selected_dataset_key]
                st.session_state.selected_train_dataset_id = selected_train_dataset_id
                st.info(f"Dataset selected for training: {selected_dataset_key.split('[')[0]}")
            else:
                st.warning("Please select a dataset to use for training.")

        else:
            st.warning("No reference datasets found. Upload one in this tab.")
            
        st.markdown("---")
        
        st.subheader("3. Configure and Start Training Run")
        
        with st.form("training_form"):
            t_model_name = st.text_input("Run Nickname (e.g., Random_Forest_1)",value= "LogReg_Test_V1", key="t_model_name")
            
            t_model_class = st.text_input(
                "Model Class Name (e.g., LogisticRegression)", 
                value="LogisticRegression", 
                key="t_model_class"
            )
            t_framework_path = st.text_input(
                "Framework Path (e.g., sklearn.linear_model)", 
                value="sklearn.linear_model", 
                key="t_framework_path"
            )
            t_feature_string = st.text_input("Feature Columns (e.g., col1, col2, col3)", value= "sepal_length, sepal_width, petal_length, petal_width", key="t_feature_string")
            t_target_string = st.text_input("Target Column Name (e.g., target)", value= "target", key="t_target_string")
            
            st.markdown("---")
            st.caption("Hyperparameters (JSON format)")
            t_hyperparams_json = st.text_area(
                "Hyperparams",
                value='{"C": 1.0, "solver": "liblinear"}',
                height=150
            )
            
            if st.form_submit_button("Trigger Training Job"):
                if not selected_train_dataset_id:
                    st.error("Training failed: You must select a dataset to train the model.")
                else:
                    try:
                        hyperparams = json.loads(t_hyperparams_json)
                        if t_model_name and t_model_class and t_framework_path and t_feature_string and t_target_string:
                            trigger_training_job(
                                t_model_name, 
                                t_model_class, 
                                t_framework_path, 
                                hyperparams, 
                                t_feature_string, 
                                t_target_string,
                                selected_train_dataset_id 
                            )
                        else:
                            st.error("All Model, Feature, and Target fields are required.")
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format for Hyperparameters.")

        st.markdown("---")
        st.markdown(f"**MLflow Tracking URL:** [http://localhost:5000](http://localhost:5000) (View your runs here)")