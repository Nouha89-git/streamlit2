
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Charger le mod�le et les encodeurs
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
    page_title="Pr�diction d'Inclusion Financi�re",
    page_icon="Money",
    layout="wide"
)

# Titre de l'application
st.title("Pr�diction d'Inclusion Financi�re")
st.write('''
Cette application pr�dit si une personne est susceptible d'avoir un compte bancaire 
en fonction de ses informations d�mographiques et �conomiques.
''')

# Chargement des ressources
model, encoders, scaler = load_model_resources()

# Interface utilisateur pour la saisie des donn�es
st.sidebar.header("Entrez les informations")

# Cr�ation du formulaire
with st.sidebar.form("prediction_form"):
    # Pays
    country = st.selectbox("Pays", ["Kenya", "Rwanda", "Tanzania", "Uganda"])

    # Type de localisation
    location_type = st.selectbox("Type de localisation", ["Rural", "Urban"])

    # Acc�s au t�l�phone portable
    cellphone_access = st.selectbox("Acc�s � un t�l�phone portable", ["Yes", "No"])

    # Taille du m�nage
    household_size = st.number_input("Taille du m�nage", min_value=1, max_value=20, value=4)

    # �ge du r�pondant
    age = st.number_input("�ge", min_value=16, max_value=100, value=30)

    # Genre du r�pondant
    gender = st.selectbox("Genre", ["Male", "Female"])

    # Relation avec le chef de m�nage
    relationship = st.selectbox("Relation avec le chef de m�nage", [
        "Head of Household", "Spouse", "Child", "Parent", 
        "Other relative", "Other non-relatives", "Dont know"
    ])

    # �tat civil
    marital_status = st.selectbox("�tat civil", [
        "Married/Living together", "Divorced/Seperated", 
        "Widowed", "Single/Never Married", "Don't know"
    ])

    # Niveau d'�ducation
    education = st.selectbox("Niveau d'�ducation", [
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

    # Bouton de pr�diction
    predict_button = st.form_submit_button("Pr�dire")

# Affichage de la pr�diction
if predict_button:
    # Pr�paration des donn�es pour la pr�diction
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

    # Cr�er un DataFrame � partir des donn�es d'entr�e
    input_df = pd.DataFrame([input_data])

    # Encoder les variables cat�gorielles
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Appliquer le scaling
    input_scaled = scaler.transform(input_df)

    # Effectuer la pr�diction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # Afficher les r�sultats
    st.header("R�sultat de la pr�diction")

    # Afficher la pr�diction avec une mise en forme conditionnelle
    if prediction[0] == 1:
        st.success("Cette personne est susceptible d'avoir un compte bancaire.")
    else:
        st.error("Cette personne est susceptible de ne pas avoir de compte bancaire.")

    # Afficher la probabilit�
    st.write(f"Probabilit� d'avoir un compte bancaire: {prediction_proba[0][1]:.2%}")

    # Afficher un graphique de la probabilit�
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.barh(["Pas de compte", "Compte bancaire"], prediction_proba[0], color=['red', 'green'])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probabilit�')
    ax.set_title('Probabilit� de pr�diction')
    st.pyplot(fig)

    # Afficher les caract�ristiques importantes
    st.subheader("Analyse des facteurs importants")
    st.write('''
    D'apr�s notre mod�le, les facteurs les plus importants pour d�terminer si une personne
    poss�de un compte bancaire sont g�n�ralement:
    1. L'�ge
    2. Le niveau d'�ducation
    3. Le type d'emploi
    4. L'acc�s � un t�l�phone portable
    5. Le pays de r�sidence

    Ces facteurs peuvent varier selon les cas individuels.
    ''')

# Section d'information sur l'inclusion financi�re
st.header("Qu'est-ce que l'inclusion financi�re?")
st.write('''
L'inclusion financi�re signifie que les particuliers et les entreprises ont acc�s � des produits 
et services financiers utiles et abordables qui r�pondent � leurs besoins � transactions, paiements, 
�pargne, cr�dit et assurance � d�livr�s de mani�re responsable et durable.

L'acc�s � un compte de transaction est une premi�re �tape vers une inclusion financi�re plus large, 
car il permet aux gens de stocker de l'argent et d'envoyer et recevoir des paiements.
''')

# Pied de page
st.sidebar.markdown("---")
st.sidebar.info(
    "Cette application a �t� d�velopp�e pour pr�dire l'inclusion financi�re en Afrique de l'Est "
    "en utilisant des techniques de machine learning."
)
