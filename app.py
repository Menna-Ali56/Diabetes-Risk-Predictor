# Diabetes Predictor App - English Version
# Save this file as: diabetes_predictor.py

import streamlit as st
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px
import os  

# ========== Page Configuration ==========
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Custom CSS for Better Styling ==========
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .result-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    }
    .danger-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .info-card {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 30px;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ========== Header ==========
st.markdown("""
<div class="main-header">
    <h1 style="color: white; font-size: 3rem;">🩺 Diabetes Risk Predictor</h1>
    <p style="color: white; font-size: 1.2rem;">AI-Powered Health Assessment Tool</p>
</div>
""", unsafe_allow_html=True)

# ========== Load Best Model from MLflow ==========
@st.cache_resource
@st.cache_resource
def load_best_model():
    try:
        # تحديد المسار
        current_path = os.path.dirname(os.path.abspath(__file__))
        mlruns_path = os.path.join(current_path, "mlruns")
        mlflow.set_tracking_uri(f"file:///{mlruns_path}")
        
        client = MlflowClient()
        all_runs = []
        
        # جلب كل التجارب
        experiments = client.search_experiments()
        
        for exp in experiments:
        
            if not exp.experiment_id.isdigit() or exp.name.startswith('.') or 'checkpoint' in exp.name.lower():
                continue
            
            try:
                runs = client.search_runs(exp.experiment_id)
                for run in runs:
                    acc = run.data.metrics.get("accuracy", None)
                    if acc is not None:
                        all_runs.append({
                            "run_id": run.info.run_id,
                            "accuracy": acc,
                            "experiment": exp.name
                        })
            except Exception:
                continue
        
        if not all_runs:
            return None, None
        
        best_run = max(all_runs, key=lambda x: x["accuracy"])
        
        with st.sidebar:
            st.success(f"✅ Best Model Loaded")
            st.metric("Model Source", best_run['experiment'])
            st.metric("Accuracy", f"{best_run['accuracy']:.2%}")
        
        model_uri = f"runs:/{best_run['run_id']}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        return model, best_run['accuracy']
    
    except Exception as e:
        # دي أهم حتة: لو فيه تجربة بايظة، اطبعها في التيرمينال بس ما تبوظش الـ UI
        print(f"Skipping invalid experiment/run: {e}")
        return None, None
    try:
        # ✅ التعديل 2: تغيير المسار للمجلد الحالي
        mlflow.set_tracking_uri("file:///" + os.path.abspath("."))
        client = MlflowClient()
        
        all_runs = []
        experiments = client.search_experiments()
        
        for exp in experiments:
            # ✅ التعديل 3: تجاهل مجلد .ipynb_checkpoints
            if '.ipynb_checkpoints' in exp.name or '.ipynb_checkpoints' in exp.experiment_id:
                continue
            
            try:
                runs = client.search_runs(exp.experiment_id)
                for run in runs:
                    acc = run.data.metrics.get("accuracy", None)
                    if acc is not None:
                        all_runs.append({
                            "run_id": run.info.run_id,
                            "accuracy": acc,
                            "experiment": exp.name
                        })
            except:
                continue
        
        if not all_runs:
            st.warning("⚠️ No trained models found. Using default model.")
            return None, None
        
        best_run = max(all_runs, key=lambda x: x["accuracy"])
        
        with st.sidebar:
            st.success(f"✅ Best Model Loaded")
            st.metric("Model", best_run['experiment'][:30])
            st.metric("Accuracy", f"{best_run['accuracy']:.2%}")
        
        model_uri = f"runs:/{best_run['run_id']}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        return model, best_run['accuracy']
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def get_default_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_dummy = np.random.rand(100, 8)
    y_dummy = np.random.randint(0, 2, 100)
    model.fit(X_dummy, y_dummy)
    return model

# ========== Load Model ==========
model, best_accuracy = load_best_model()

if model is None:
    model = get_default_model()
    st.sidebar.warning("⚠️ Using temporary default model")

# ========== Sidebar Information ==========
with st.sidebar:
    st.markdown("## 📊 About")
    st.markdown("""
    This tool uses Machine Learning to predict the likelihood of diabetes based on medical parameters.
    
    ### 🏥 Input Parameters:
    - **Pregnancies**: Number of times pregnant
    - **Glucose**: Plasma glucose concentration
    - **BloodPressure**: Diastolic blood pressure
    - **SkinThickness**: Triceps skin fold thickness
    - **Insulin**: 2-Hour serum insulin
    - **BMI**: Body Mass Index
    - **DPF**: Diabetes Pedigree Function
    - **Age**: Age in years
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Model Performance")
    
    if best_accuracy:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=best_accuracy * 100,
            title={'text': "Model Accuracy (%)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "#667eea"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 80], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                 'thickness': 0.75,
                                 'value': 90}}))
        st.plotly_chart(fig, use_container_width=True)

# ========== Main Input Form ==========
st.markdown("## 📝 Enter Patient Data")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    with st.container():
        st.markdown("### 👤 Personal Information")
        age = st.slider("Age (years)", 0, 120, 30, help="Patient's age in years")
        pregnancies = st.number_input("Number of Pregnancies", 0, 20, 1, help="Number of times pregnant")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🩸 Blood Measurements")
    glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 120, 
                               help="2-hour plasma glucose concentration")
    insulin = st.number_input("Insulin Level (mu U/ml)", 0, 900, 80,
                              help="2-hour serum insulin")

with col2:
    st.markdown("### 💪 Body Measurements")
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1,
                          help="Body Mass Index (weight in kg / height in m²)")
    skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20,
                                     help="Triceps skin fold thickness")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ❤️ Vital Signs")
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 150, 70,
                                     help="Diastolic blood pressure")

with col2:
    st.markdown("### 🧬 Genetic Factors")
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01,
                          help="Family history of diabetes")

# ========== Predict Button ==========
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button("🔍 Analyze Health Risk", use_container_width=True, type="primary")

# ========== Prediction and Results ==========
if predict_button:
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else [0, 0]
        
        st.markdown("---")
        st.markdown("## 📊 Diagnosis Results")
        
        # Risk Gauge Chart
        risk_percentage = (probability[1] if prediction == 1 else (1 - probability[0])) * 100
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_percentage,
            title={'text': "Diabetes Risk Level (%)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "white" if prediction == 1 else "#84fab0"},
                'steps': [
                    {'range': [0, 30], 'color': "#84fab0", 'name': "Low Risk"},
                    {'range': [30, 70], 'color': "#ffd93d", 'name': "Moderate Risk"},
                    {'range': [70, 100], 'color': "#f5576c", 'name': "High Risk"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_percentage
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Result Card
        if prediction == 1:
            st.markdown("""
            <div class="result-card danger-card">
                <h2 style="color: white;">⚠️ HIGH RISK</h2>
                <p style="color: white; font-size: 1.2rem;">Based on the analysis, the patient shows signs suggesting diabetes risk.</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("💡 Medical Recommendations"):
                st.markdown("""
                ### 🏥 Immediate Actions:
                1. **Consult a healthcare provider immediately**
                2. **Schedule a comprehensive diabetes screening**
                3. **Monitor blood glucose levels regularly**
                
                ### 🥗 Lifestyle Modifications:
                - Adopt a **balanced diet** low in refined sugars
                - Engage in **regular physical activity** (30 min/day)
                - Maintain a **healthy weight**
                - **Quit smoking** and limit alcohol consumption
                
                ### 📋 Follow-up:
                - Regular HbA1c tests every 3-6 months
                - Eye and foot examinations annually
                - Blood pressure and cholesterol monitoring
                """)
        else:
            st.markdown("""
            <div class="result-card success-card">
                <h2 style="color: #2d3748;">✅ LOW RISK</h2>
                <p style="color: #2d3748; font-size: 1.2rem;">Based on the analysis, the patient shows no immediate signs of diabetes.</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("💡 Preventive Recommendations"):
                st.markdown("""
                ### 🛡️ Preventive Care:
                1. **Annual health check-ups**
                2. **Maintain healthy lifestyle habits**
                3. **Regular physical activity** (at least 150 minutes/week)
                
                ### 🥗 Healthy Habits:
                - Eat a **balanced diet** rich in fruits and vegetables
                - **Limit processed foods** and sugary beverages
                - **Stay hydrated** with water
                - Get **adequate sleep** (7-9 hours)
                
                ### 📊 Monitoring:
                - Annual blood glucose screening
                - Track weight and BMI regularly
                - Monitor blood pressure
                """)
        
        # Confidence Score
        col1, col2, col3 = st.columns(3)
        with col2:
            confidence = probability[1] if prediction == 1 else probability[0]
            st.markdown(f"""
            <div class="info-card" style="text-align: center;">
                <h3>🎯 Prediction Confidence</h3>
                <h2 style="color: #667eea;">{confidence:.1%}</h2>
                <p>Model certainty in this prediction</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Data Summary Table
        with st.expander("📋 Patient Data Summary"):
            summary_df = pd.DataFrame({
                "Parameter": ["Age", "Pregnancies", "Glucose", "Blood Pressure", 
                            "Skin Thickness", "Insulin", "BMI", "Diabetes Pedigree"],
                "Value": [age, pregnancies, glucose, blood_pressure, 
                         skin_thickness, insulin, bmi, dpf],
                "Normal Range": ["20-60", "0-10", "70-140", "60-90", 
                               "10-40", "2-25", "18.5-25", "< 0.5"],
                "Status": [
                    "✅ Normal" if 20 <= age <= 60 else "⚠️ Check",
                    "✅ Normal" if pregnancies <= 10 else "⚠️ High",
                    "✅ Normal" if 70 <= glucose <= 140 else "⚠️ High",
                    "✅ Normal" if 60 <= blood_pressure <= 90 else "⚠️ Check",
                    "✅ Normal" if 10 <= skin_thickness <= 40 else "⚠️ Check",
                    "✅ Normal" if insulin <= 25 else "⚠️ High",
                    "✅ Normal" if 18.5 <= bmi <= 25 else "⚠️ High",
                    "✅ Normal" if dpf < 0.5 else "⚠️ Elevated"
                ]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Feature Importance Plot
        if hasattr(model, 'feature_importances_'):
            with st.expander("📊 Feature Importance Analysis"):
                feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                               'Insulin', 'BMI', 'DPF', 'Age']
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                            title="Which factors most influence the prediction?",
                            color='Importance', color_continuous_scale='Viridis')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# ========== Footer ==========
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment.</p>
    <p>Made with ❤️ using Machine Learning & Streamlit</p>
</div>
""", unsafe_allow_html=True)
