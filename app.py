
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Charger le modèle et les encodeurs
@st.cache_resource
def load_model_resources():
    with open('financial_inclusion_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, encoders, scaler

# Configuration de l'application
st.set_page_config(
    page_title="Prédiction d'Inclusion Financière",
    page_icon="Money",
    layout="wide"
)

# Titre de l'application
st.title("Prédiction d'Inclusion Financière")
st.write('''
Cette application prédit si une personne est susceptible d'avoir un compte bancaire 
en fonction de ses informations démographiques et économiques.
''')

# Chargement des ressources
model, encoders, scaler = load_model_resources()

# Interface utilisateur pour la saisie des données
st.sidebar.header("Entrez les informations")

# Création du formulaire
with st.sidebar.form("prediction_form"):
    # Pays
    country = st.selectbox("Pays", ["Kenya", "Rwanda", "Tanzania", "Uganda"])

    # Type de localisation
    location_type = st.selectbox("Type de localisation", ["Rural", "Urban"])

    # Accès au téléphone portable
    cellphone_access = st.selectbox("Accès à un téléphone portable", ["Yes", "No"])

    # Taille du ménage
    household_size = st.number_input("Taille du ménage", min_value=1, max_value=20, value=4)

    # Âge du répondant
    age = st.number_input("Âge", min_value=16, max_value=100, value=30)

    # Genre du répondant
    gender = st.selectbox("Genre", ["Male", "Female"])

    # Relation avec le chef de ménage
    relationship = st.selectbox("Relation avec le chef de ménage", [
        "Head of Household", "Spouse", "Child", "Parent", 
        "Other relative", "Other non-relatives", "Dont know"
    ])

    # État civil
    marital_status = st.selectbox("État civil", [
        "Married/Living together", "Divorced/Seperated", 
        "Widowed", "Single/Never Married", "Don't know"
    ])

    # Niveau d'éducation
    education = st.selectbox("Niveau d'éducation", [
        "No formal education", "Primary education", 
        "Secondary education", "Vocational/Specialised training", 
        "Tertiary education", "Other/Dont know/RTA"
    ])

    # Type d'emploi
    job_type = st.selectbox("Type d'emploi", [
        "Farming and Fishing", "Self employed", 
        "Formally employed Government", "Formally employed Private", 
        "Informally employed", "Remittance Dependent", 
        "Government Dependent", "Other Income", 
        "No Income", "Dont Know/Refuse to answer"
    ])

    # Bouton de prédiction
    predict_button = st.form_submit_button("Prédire")

# Affichage de la prédiction
if predict_button:
    # Préparation des données pour la prédiction
    input_data = {
        'country': country,
        'location_type': location_type,
        'cellphone_access': cellphone_access,
        'household_size': household_size,
        'age_of_respondent': age,
        'gender_of_respondent': gender,
        'relationship_with_head': relationship,
        'marital_status': marital_status,
        'education_level': education,
        'job_type': job_type
    }

    # Créer un DataFrame à partir des données d'entrée
    input_df = pd.DataFrame([input_data])

    # Encoder les variables catégorielles
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Appliquer le scaling
    input_scaled = scaler.transform(input_df)

    # Effectuer la prédiction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # Afficher les résultats
    st.header("Résultat de la prédiction")

    # Afficher la prédiction avec une mise en forme conditionnelle
    if prediction[0] == 1:
        st.success("Cette personne est susceptible d'avoir un compte bancaire.")
    else:
        st.error("Cette personne est susceptible de ne pas avoir de compte bancaire.")

    # Afficher la probabilité
    st.write(f"Probabilité d'avoir un compte bancaire: {prediction_proba[0][1]:.2%}")

    # Afficher un graphique de la probabilité
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.barh(["Pas de compte", "Compte bancaire"], prediction_proba[0], color=['red', 'green'])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probabilité')
    ax.set_title('Probabilité de prédiction')
    st.pyplot(fig)

    # Afficher les caractéristiques importantes
    st.subheader("Analyse des facteurs importants")
    st.write('''
    D'après notre modèle, les facteurs les plus importants pour déterminer si une personne
    possède un compte bancaire sont généralement:
    1. L'âge
    2. Le niveau d'éducation
    3. Le type d'emploi
    4. L'accès à un téléphone portable
    5. Le pays de résidence

    Ces facteurs peuvent varier selon les cas individuels.
    ''')

# Section d'information sur l'inclusion financière
st.header("Qu'est-ce que l'inclusion financière?")
st.write('''
L'inclusion financière signifie que les particuliers et les entreprises ont accès à des produits 
et services financiers utiles et abordables qui répondent à leurs besoins – transactions, paiements, 
épargne, crédit et assurance – délivrés de manière responsable et durable.

L'accès à un compte de transaction est une première étape vers une inclusion financière plus large, 
car il permet aux gens de stocker de l'argent et d'envoyer et recevoir des paiements.
''')

# Pied de page
st.sidebar.markdown("---")
st.sidebar.info(
    "Cette application a été développée pour prédire l'inclusion financière en Afrique de l'Est "
    "en utilisant des techniques de machine learning."
)
