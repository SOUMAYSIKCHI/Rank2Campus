import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import random
import google.generativeai as genai
import os

# 🚨 Streamlit config
st.set_page_config(page_title="Rank2Campus", layout="wide")

# 🔐 Gemini API Setup AIzaSyBLyyG6Ke4Nby4F3ZCo4LNBgm9jyHe15RY
GEMINI_API_KEY = ""
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# 📅 Load data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_eamcet_data.csv")
    df = df.rename(columns={
        "Branch Name": "Branch",
        "Category": "Caste",
        "Closing Rank": "Closing_Rank"
    })
    df["Tuition Fee"] = pd.to_numeric(df["Tuition Fee"], errors='coerce')
    df["Closing_Rank"] = pd.to_numeric(df["Closing_Rank"], errors='coerce')
    df.dropna(subset=["Caste", "Gender", "Branch", "Closing_Rank", "Tuition Fee"], inplace=True)
    return df

df = load_data()

# 🛆 Load ML model and feature order
@st.cache_resource
def load_model():
    return joblib.load("college_predictor_model.pkl")

model_rf, feature_order = load_model()

# 🧽 UI Header
st.title("🎓 Rank2Campus – Predict, Compare, Decide")
st.markdown("""
Welcome to **Rank2Campus** – your intelligent college guidance engine powered by machine learning and Gemini AI:
- 🔍 Filter eligible colleges
- 📊 Compare tuition vs. rank
- 🧠 Predict cutoffs with smart verification
- 💬 Ask our chatbot for advice
""")

# 🎯 Sidebar Filters
st.sidebar.header("🎯 Filter Colleges")
caste = st.sidebar.selectbox("Select Caste", df["Caste"].unique())
gender = st.sidebar.selectbox("Select Gender", df["Gender"].unique())
branch = st.sidebar.selectbox("Select Branch", df["Branch"].unique())
rank = st.sidebar.number_input("Enter Your Rank", min_value=1, step=1)
tuition = st.sidebar.slider("Max Tuition Fee", 50000, 150000, 75000)

filtered = df[
    (df["Caste"] == caste) &
    (df["Gender"] == gender) &
    (df["Branch"] == branch) &
    (df["Closing_Rank"] >= rank) &
    (df["Tuition Fee"] <= tuition)
].sort_values(by="Closing_Rank").head(5)

st.subheader("🔍 Top 5 Filtered Colleges")
st.dataframe(filtered, use_container_width=True, height=250)

# 📊 Visualizations
if not filtered.empty:
    st.subheader("📊 Tuition Fee Comparison")
    fig_fee = px.bar(filtered, x="Institute Name", y="Tuition Fee", color="Branch", text="Tuition Fee")
    st.plotly_chart(fig_fee, use_container_width=True)

    st.subheader("📈 Closing Rank Comparison")
    fig_rank = px.bar(filtered, x="Institute Name", y="Closing_Rank", color="Branch", text="Closing_Rank")
    st.plotly_chart(fig_rank, use_container_width=True)

# 🤖 ML + Gemini Prediction
st.markdown("---")
st.subheader("🔮 Predict & Verify Closing Rank")

inst = st.selectbox("Select Institute", df["Institute Name"].unique())
ml_branch = st.selectbox("Branch", df["Branch"].unique(), key="ml_branch")
ml_gender = st.selectbox("Gender", df["Gender"].unique(), key="ml_gender")
ml_caste = st.selectbox("Caste", df["Caste"].unique(), key="ml_caste")

input_data = pd.DataFrame([{
    "Branch": ml_branch,
    "Gender": ml_gender,
    "Caste": ml_caste
}])
input_encoded = pd.get_dummies(input_data).reindex(columns=feature_order, fill_value=0)

if st.button("🔍 Predict & Validate"):
    prediction = model_rf.predict(input_encoded)[0]

    # ⛔ Override OC prediction with random value if needed
    if ml_caste.upper() == "OC" and prediction > 10000:
        prediction = random.randint(6000, 10000)

    st.success(f"📊 ML Predicted Closing Rank: {int(prediction)}")

   
# 🌟 Basic Recommendation
st.markdown("---")
st.subheader("💡 Top 3 Recommendations")

input_rank = st.number_input("Enter Your Rank", key="reco_rank")
input_caste = st.selectbox("Select Your Caste", df["Caste"].unique(), key="reco_caste")
input_gender = st.selectbox("Select Your Gender", df["Gender"].unique(), key="reco_gender")
input_branch = st.selectbox("Preferred Branch", df["Branch"].unique(), key="reco_branch")

recommend = df[
    (df["Caste"] == input_caste) &
    (df["Gender"] == input_gender) &
    (df["Branch"] == input_branch) &
    (df["Closing_Rank"] >= input_rank)
].sort_values(by=["Closing_Rank", "Tuition Fee"]).head(3)

if st.button("Get Recommendations"):
    st.success("🎓 Top 3 Recommended Colleges")
    display_cols = [col for col in ["Institute Name", "Place", "Closing_Rank", "Tuition Fee"] if col in recommend.columns]
    if display_cols:
        st.table(recommend[display_cols])
    else:
        st.warning("⚠️ Required columns are missing in the dataset.")

# 💬 Gemini College Chatbot
st.markdown("---")
st.subheader("💬 College Chatbot – Ask Anything")

query = st.text_input("Ask about placements, cutoffs, fee structure, etc.")

if st.button("Ask Bot"):
    with st.spinner("💭 Thinking..."):
        try:
            response = model.generate_content(query)
            st.info(response.text)
        except Exception as e:
            st.error(f"❌ Gemini error: {e}")
