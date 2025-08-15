import streamlit as st
from annotated_text import annotated_text
import json
import requests

# Variables
showExampleButton = True

st.title('🕵️‍♀️ Oskar')
annotated_text(
    "Let's test an AI model capable of detecting fraudulent banking transactions ",
    ("(all with 91% accuracy 😉)", "", "#5d00a8")
)

st.write('<span style="color:#818589"> by Corentin Deldalle </span><br/><br/><br/>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a JSON file", type=["JSON"])

if uploaded_file is not None:
    data = json.load(uploaded_file)

    st.write("Contenu du fichier JSON :")
    st.json(data, expanded=False)
    showExampleButton = False

    if st.button('Lancer la prédiction'):
        try:
            response = requests.post(
                'http://localhost:8000/predict',
                json=data
            )
            if response.status_code == 200:
                preds = response.json()
                st.write('Résultat de la prédiction :')
                if preds["message"] == "Fraude détectée":
                    annotated_text('', ("Fraude détectée", "", "#FA3939"))
                else:
                    annotated_text('', ("Transaction normale", "", "#189C02"))
            else:
                st.error(f"Erreur API : {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Erreur lors de la requête : {e}")

st.write("<br/><br/>", unsafe_allow_html=True)

exemple = [
    0.09652539431568392, 0.0073790751145725, 2.36518252886777, -2.60028738169342,
    1.11160176466249, 3.2764413263667, -1.7761407852846, 2.11453088108345,
    -0.830084106599812, 0.900489974281822, -3.37617706712688, 2.05681205985716,
    -3.9842567303491, 1.02196791340796, -5.96790496861639, -1.15160831490592,
    1.67973960141233, 5.58611468683053, 2.78913115591619, -2.24107485298106,
    -0.0063884972816308, -0.563944162060269, -0.902099515410175,
    -0.404382448597275, -0.0129439005065412, 0.589836154947909,
    -0.734449052212668, -0.447529183217452, -0.362374775122343, -0.16121850015180653
]

col1, col2 = st.columns([3,1])

if showExampleButton == True:
    with col1:
        st.markdown('<div style="margin-top:10px;">Ou tester avec un exemple de transaction</div>',
                    unsafe_allow_html=True)

    with col2:
        if st.button("Testez l'exemple"):
            try:
                payload = {"features": exemple}
                response = requests.post(
                    'http://localhost:8000/predict',
                    json=payload
                )
                if response.status_code == 200:
                    st.write("Résultat :")
                    result = response.json()
                    if result["message"] == "Fraude détectée":
                        annotated_text('', ("Fraude détectée", "", "#FA3939"))
                    else:
                        annotated_text('', ("Transaction normal", "", "#20CC02"))
                else:
                    st.error(f"Erreur API : {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Erreur lors de la requête : {e}")