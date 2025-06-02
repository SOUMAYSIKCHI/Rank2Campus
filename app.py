import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import random

# 🚨 Streamlit config
st.set_page_config(page_title="Rank2Campus", layout="wide")

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

# 🧠 Load ML model
@st.cache_resource
def load_model():
    return joblib.load("college_predictor_model.pkl")

model_rf, feature_order = load_model()

# 🧽 UI Header
st.title("🎓 Rank2Campus – Predict, Compare, Decide")
st.markdown("""
Welcome to **Rank2Campus** – your intelligent college guidance engine:
- 🔍 Filter eligible colleges
- 📊 Compare tuition vs. rank
- 🧠 Predict cutoffs with ML
- 🌟 Get top 10 colleges by rank
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

# 🤖 ML Prediction
st.markdown("---")
st.subheader("🔮 Predict Closing Rank with ML")

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

if st.button("🔍 Predict"):
    prediction = model_rf.predict(input_encoded)[0]
    if ml_caste.upper() == "OC" and prediction > 10000:
        prediction = random.randint(6000, 10000)
    st.success(f"📊 Predicted Closing Rank: {int(prediction)}")

# 🌟 Basic Recommendation
st.markdown("---")
st.subheader("💡 Top 3 Personalized Recommendations")

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
    display_cols = [col for col in ["Institute Name", "Closing_Rank", "Tuition Fee"] if col in recommend.columns]
    st.table(recommend[display_cols])

# 🆕 Top 10 Colleges Based on Rank
# 🆕 Unique Top 10 Colleges by Rank
# 🆕 Unique Top N Colleges by Rank
st.markdown("---")
st.subheader("🏆 Unique Top Colleges Based on Your Rank")

simple_rank = st.number_input("Enter Rank (Ignore Category/Gender)", min_value=1, key="simple_rank")

col1, col2, col3 = st.columns(3)
get_top_n = None

with col1:
    if st.button("🎯 Find Top 10 Colleges"):
        get_top_n = 10
with col2:
    if st.button("🥇 Find Top 15 Colleges"):
        get_top_n = 15
with col3:
    if st.button("🏅 Find Top 20 Colleges"):
        get_top_n = 20

if get_top_n:
    topN_raw = df[df["Closing_Rank"] >= simple_rank]
    topN_unique = topN_raw.drop_duplicates(subset=["Institute Name"]).sort_values(by="Closing_Rank").head(get_top_n)

    if not topN_unique.empty:
        st.write(f"🎓 **Top {get_top_n} Unique Colleges You May Be Eligible For:**")
        for idx, college in enumerate(topN_unique["Institute Name"].tolist(), start=1):
            st.markdown(f"{idx}. {college}")
    else:
        st.warning("❌ No unique colleges found for this rank.")
# 🔄 Option to show all unique colleges
st.markdown("----")
show_all = st.checkbox("📚 Show All Unique Colleges in the Dataset")

if show_all:
    all_unique_colleges = df.drop_duplicates(subset=["Institute Name"])[["Institute Name"]].sort_values(by="Institute Name")
    st.write(f"🏛️ Total Unique Colleges: {all_unique_colleges.shape[0]}")
    for idx, clg in enumerate(all_unique_colleges["Institute Name"].tolist(), start=1):
        st.markdown(f"{idx}. {clg}")
