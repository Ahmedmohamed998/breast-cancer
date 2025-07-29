import streamlit as st
import numpy as np
import pickle
from collections import Counter

st.set_page_config(layout='wide')
st.title('🔬 Breast Cancer Prediction')
st.markdown("")
rf_model = pickle.load(open('pkl/rf_model.pkl', 'rb'))
lg_model = pickle.load(open('pkl/lg_model.pkl', 'rb'))
svm_model = pickle.load(open('pkl/svm_model.pkl', 'rb'))
dt_model = pickle.load(open('pkl/dt_model.pkl', 'rb'))

input_features = [
    'radius_mean', 'perimeter_mean', 'area_mean', 'compactness_mean',
    'concavity_mean', 'concave points_mean', 'radius_se', 'perimeter_se',
    'area_se', 'radius_worst', 'perimeter_worst', 'area_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst'
]

def user_input():
    features = {}
    cols = st.columns(3)

    for i, feature in enumerate(input_features):
        with cols[i % 3]:
            features[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.01)

    return features

user_features = user_input()
features_array = np.array([list(user_features.values())]).reshape(1, -1)

if 'y_pred' not in st.session_state:
    st.session_state.y_pred = []

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button('Random Forest'):
        pred = rf_model.predict(features_array)[0]
        st.session_state.y_pred.append(pred)
        st.success(f'Random Forest: {pred} ({"Malignant" if pred == "M" else "Benign"})')

with col2:
    if st.button('Logistic Regression'):
        pred = lg_model.predict(features_array)[0]
        st.session_state.y_pred.append(pred)
        st.success(f'Logistic Regression: {pred} ({"Malignant" if pred == "M" else "Benign"})')

with col3:
    if st.button('SVM'):
        pred = svm_model.predict(features_array)[0]
        st.session_state.y_pred.append(pred)
        st.success(f'SVM: {pred} ({"Malignant" if pred == "M" else "Benign"})')

with col4:
    if st.button('Decision Tree'):
        pred = dt_model.predict(features_array)[0]
        st.session_state.y_pred.append(pred)
        st.success(f'Decision Tree: {pred} ({"Malignant" if pred == "M" else "Benign"})')

with col5:
    if st.button('Final Voting From Models'):
        if len(st.session_state.y_pred) == 4:
            final_vote = Counter(st.session_state.y_pred).most_common(1)[0][0]
            st.success(f'Final Prediction: {final_vote} ({"Malignant" if final_vote == "M" else "Benign"})')
            del st.session_state.y_pred
        else:
            st.error('Please run all 4 model predictions before final voting.')
