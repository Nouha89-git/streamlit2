import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import streamlit as st
import pickle
import os
import warnings

warnings.filterwarnings('ignore')


# 1. Préparation des données
def load_data():
    """Charger les données depuis un fichier local"""
    # Remplacer par le chemin de votre fichier
    file_path = "Financial_inclusion_dataset.csv"

    # Si le fichier est un .csv
    df = pd.read_csv(file_path)

    # Si le fichier est un .xlsx ou .xls, décommentez cette ligne:
    # df = pd.read_excel(file_path)

    return df


def explore_data(df):
    """Explorer et afficher les informations de base sur les données"""
    print("Aperçu des données:")
    print(df.head())

    print("\nInformations sur les données:")
    print(df.info())

    print("\nStatistiques descriptives:")
    print(df.describe(include='all'))

    print("\nValeurs manquantes par colonne:")
    print(df.isnull().sum())

    # Visualisations basiques avec matplotlib/seaborn
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(df.select_dtypes(include=['int64', 'float64']).columns[:5]):
        if col != 'uniqueid':  # Ignorer l'ID unique
            plt.subplot(1, 5, i + 1)
            sns.histplot(df[col], kde=True)
            plt.title(col)
    plt.tight_layout()
    plt.savefig('numeric_distributions.png')
    print("\nGraphiques de distribution des variables numériques sauvegardés dans 'numeric_distributions.png'")

    return df


def preprocess_data(df):
    """Prétraiter les données pour le modèle ML"""
    # Créer une copie des données
    processed_df = df.copy()

    # Vérifier et supprimer les doublons
    duplicates = processed_df.duplicated().sum()
    if duplicates > 0:
        print(f"Suppression de {duplicates} doublons")
        processed_df = processed_df.drop_duplicates()

    # Gérer les valeurs manquantes
    numeric_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = processed_df.select_dtypes(include=['object']).columns

    # Imputer les valeurs numériques manquantes avec la médiane
    if processed_df[numeric_cols].isnull().sum().sum() > 0:
        num_imputer = SimpleImputer(strategy='median')
        processed_df[numeric_cols] = num_imputer.fit_transform(processed_df[numeric_cols])

    # Imputer les valeurs catégorielles manquantes avec le mode
    if processed_df[categorical_cols].isnull().sum().sum() > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        processed_df[categorical_cols] = pd.DataFrame(
            cat_imputer.fit_transform(processed_df[categorical_cols]),
            columns=categorical_cols,
            index=processed_df.index
        )

    # Gérer les valeurs aberrantes (méthode IQR pour les colonnes numériques)
    for col in numeric_cols:
        if col != 'uniqueid':  # Exclure l'ID unique
            Q1 = processed_df[col].quantile(0.25)
            Q3 = processed_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Remplacer les valeurs aberrantes par les limites
            processed_df[col] = np.where(
                processed_df[col] < lower_bound,
                lower_bound,
                np.where(processed_df[col] > upper_bound, upper_bound, processed_df[col])
            )

    # Encoder les caractéristiques catégorielles
    label_encoders = {}
    for col in categorical_cols:
        if col != 'uniqueid':  # Exclure l'ID unique
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col])
            label_encoders[col] = le

    # Sauvegarder les encodeurs pour une utilisation ultérieure dans Streamlit
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    return processed_df, label_encoders


def train_model(df, target_column='bank_account'):
    """Entraîner et évaluer un modèle de classification"""
    # Définir les caractéristiques et la cible
    if target_column in df.columns:
        X = df.drop(columns=[target_column, 'uniqueid'])
        y = df[target_column]
    else:
        # Si la colonne cible n'est pas présente, nous supposons qu'elle a déjà été séparée
        X = df.drop(columns=['uniqueid']) if 'uniqueid' in df.columns else df
        y = None  # Vous devrez fournir y séparément dans ce cas

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardiser les caractéristiques numériques
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Sauvegarder le scaler pour une utilisation ultérieure
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Entraîner un modèle Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Évaluer le modèle
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nPrécision du modèle: {accuracy:.4f}")

    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred))

    # Afficher la matrice de confusion
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de confusion')
    plt.ylabel('Valeurs réelles')
    plt.xlabel('Valeurs prédites')
    plt.savefig('confusion_matrix.png')

    # Sauvegarder le modèle
    with open('financial_inclusion_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("\nModèle sauvegardé dans 'financial_inclusion_model.pkl'")

    # Importance des caractéristiques
    feature_importance = pd.DataFrame(
        {'feature': X.columns, 'importance': clf.feature_importances_}
    ).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 des caractéristiques les plus importantes')
    plt.tight_layout()
    plt.savefig('feature_importance.png')

    return clf, scaler


# 2. Application Streamlit
def create_streamlit_app():
    """Créer le fichier app.py pour l'application Streamlit"""
    app_code =
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
"""

    # Écrire le code dans un fichier app.py
    with open("app.py", "w") as f:
        f.write(app_code)

    print("\nFichier app.py créé pour l'application Streamlit")


def create_requirements_file():
    """Créer le fichier requirements.txt pour le déploiement"""
    requirements = """
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
streamlit==1.28.0
"""
    with open("requirements.txt", "w") as f:
        f.write(requirements)

    print("\nFichier requirements.txt créé")


def create_readme_file():
    """Créer un fichier README.md pour le dépôt GitHub"""
    readme_content = """
# Application de Prédiction d'Inclusion Financière

Cette application prédit si une personne est susceptible d'avoir un compte bancaire en fonction de ses informations démographiques et économiques.

## À propos de l'ensemble de données

L'ensemble de données contient des informations démographiques et les services financiers utilisés par environ 33 600 personnes en Afrique de l'Est. Le modèle ML prédit quelles personnes sont les plus susceptibles d'avoir ou d'utiliser un compte bancaire.

## Installation et exécution locale

1. Clonez ce dépôt
2. Installez les dépendances : `pip install -r requirements.txt`
3. Exécutez l'application : `streamlit run app.py`

## Déploiement

Cette application est déployée sur Streamlit Share. Vous pouvez y accéder [ici](lien-vers-votre-application).

"""
    with open("README.md", "w") as f:
        f.write(readme_content)

    print("\nFichier README.md créé")


def deploy_instructions():
    """Afficher les instructions pour le déploiement sur GitHub et Streamlit Share"""
    instructions = """
=== INSTRUCTIONS POUR LE DÉPLOIEMENT ===

1. Créez un compte GitHub si vous n'en avez pas déjà un
   - Visitez https://github.com/ et inscrivez-vous

2. Créez un nouveau dépôt sur GitHub
   - Allez sur https://github.com/new
   - Nommez votre dépôt (par exemple, "financial-inclusion-predictor")
   - Choisissez "Public" comme visibilité
   - Cliquez sur "Create repository"

3. Initialisez et poussez votre code vers GitHub
   - Ouvrez un terminal dans le répertoire de votre projet
   - Exécutez les commandes suivantes :

     git init
     git add .
     git commit -m "Premier commit"
     git branch -M main
     git remote add origin https://github.com/VOTRE-NOM-UTILISATEUR/VOTRE-NOM-REPO.git
     git push -u origin main

4. Créez un compte Streamlit Share
   - Visitez https://share.streamlit.io/ et inscrivez-vous avec votre compte GitHub

5. Déployez votre application
   - Dans Streamlit Share, cliquez sur "New app"
   - Sélectionnez votre dépôt, la branche (main) et le fichier app.py
   - Cliquez sur "Deploy"

Votre application sera alors accessible publiquement via une URL fournie par Streamlit Share.
"""
    print(instructions)


def main():
    """Fonction principale exécutant toutes les étapes du projet"""
    print("=== PROJET DE PRÉDICTION D'INCLUSION FINANCIÈRE ===\n")

    # Étape 1: Charger et explorer les données
    print("1. Chargement et exploration des données...")
    try:
        df = load_data()
        df = explore_data(df)

        # Étape 2: Prétraiter les données
        print("\n2. Prétraitement des données...")
        processed_df, label_encoders = preprocess_data(df)

        # Étape 3: Entraîner le modèle
        print("\n3. Entraînement du modèle...")
        model, scaler = train_model(processed_df)

        # Étape 4: Créer l'application Streamlit
        print("\n4. Création de l'application Streamlit...")
        create_streamlit_app()

        # Étape 5: Préparer les fichiers pour le déploiement
        print("\n5. Préparation des fichiers pour le déploiement...")
        create_requirements_file()
        create_readme_file()

        # Étape 6: Instructions pour le déploiement
        print("\n6. Instructions pour le déploiement...")
        deploy_instructions()

        print("\n=== PROJET COMPLÉTÉ AVEC SUCCÈS ===")
        print("\nPour exécuter l'application localement, utilisez la commande:")
        print("streamlit run app.py")

    except FileNotFoundError:
        print("ERREUR: Le fichier de données n'a pas été trouvé.")
        print("Veuillez placer votre fichier de données (CSV ou Excel) dans le même répertoire que ce script.")
        print(
            "Ensuite, modifiez la variable 'file_path' dans la fonction load_data() pour spécifier le nom correct du fichier.")

    except Exception as e:
        print(f"ERREUR: Une erreur s'est produite: {str(e)}")


if __name__ == "__main__":
    main()