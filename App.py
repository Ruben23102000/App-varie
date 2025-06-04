import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.title("📉 Analisi del Rischio Churn Studenti")

# ---------------------------------------------
# 1. CREAZIONE DATASET SINTETICO
# ---------------------------------------------
np.random.seed(42)  # Fissiamo il seed per riproducibilità
n = 500  # Numero studenti

# Generiamo un dataset fittizio con le principali caratteristiche
studenti_df = pd.DataFrame({
    'student_id': np.arange(1, n + 1),
    'eta': np.random.randint(18, 41, size=n),
    'ore_studio_settimanali': np.random.randint(0, 60, size=n),
    'numero_esami_superati': np.random.randint(0, 35, size=n),
    'media_voti': np.round(np.random.uniform(18, 30, size=n), 1),
    'partecipazione_eventi': np.random.choice([0, 1], size=n),
    'lavoro_part_time': np.random.choice([0, 1], size=n),
    'carriera_in_corso': np.random.choice([0, 1], size=n),
    'sesso': np.random.choice(['M', 'F'], size=n),
    'regione_residenza': np.random.choice(['Lombardia', 'Lazio', 'Campania', 'Sicilia', 'Veneto', 'Altro'], size=n),
    'tipo_corso': np.random.choice(['Triennale', 'Magistrale'], size=n),
    'gruppo_corso': np.random.choice(['Economia', 'Ingegneria', 'Lettere', 'Scienze'], size=n),
    'churned': np.random.choice([0, 1], size=n, p=[0.7, 0.3])  # 30% di abbandono
})

# Visualizziamo il dataset
st.subheader("🎓 Dataset Studenti")
st.dataframe(studenti_df.head())

# ---------------------------------------------
# 2. PREPROCESSING DEI DATI
# ---------------------------------------------
# Separiamo variabili indipendenti (X) e target (y)
X = studenti_df.drop(columns=["churned"])
y = studenti_df["churned"]

# Codifica variabili categoriche con one-hot encoding
X = pd.get_dummies(X, columns=["sesso", "regione_residenza", "tipo_corso", "gruppo_corso"], drop_first=True)

# ---------------------------------------------
# 3. SPLIT TRAIN/TEST
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# ---------------------------------------------
# 4. ADDESTRAMENTO MODELLO RANDOM FOREST
# ---------------------------------------------
# Creiamo e addestriamo il classificatore
model = RandomForestClassifier(n_estimators=50, max_depth=6, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------
# 5. PREDIZIONE SU TEST SET
# ---------------------------------------------
# Probabilità di churn (classe 1)
probs = model.predict_proba(X_test)[:, 1]
# Predizione binaria
preds = model.predict(X_test)

# ---------------------------------------------
# 6. COSTRUZIONE RISULTATI CON PERCENTUALI E CATEGORIE
# ---------------------------------------------
results = X_test.copy()
results["student_id"] = studenti_df.loc[results.index, "student_id"].values
results["churn_pred"] = preds
results["churn_prob"] = np.round(probs * 100, 1)

# Classificazione in categorie di rischio
results["categoria_rischio"] = pd.cut(
    probs,
    bins=[-np.inf, 0.4, 0.7, np.inf],
    labels=["Basso Rischio", "Medio Rischio", "Alto Rischio"]
)

# ---------------------------------------------
# 7. VISUALIZZAZIONE RISULTATI
# ---------------------------------------------
st.subheader("📊 Risultati Previsione")
scelta = st.selectbox("Filtra per categoria di rischio", ["Tutti"] + results["categoria_rischio"].unique().tolist())

# Filtro dinamico per rischio
if scelta != "Tutti":
    st.dataframe(results[results["categoria_rischio"] == scelta][["student_id", "churn_prob", "categoria_rischio"]])
else:
    st.dataframe(results[["student_id", "churn_prob", "categoria_rischio"]])

# ---------------------------------------------
# 8. STATISTICHE DI SINTESI
# ---------------------------------------------
st.subheader("📈 Statistiche per Categoria di Rischio")
st.dataframe(
    results.groupby("categoria_rischio")["churn_prob"]
           .agg(["count", "mean", "min", "max"])
           .rename(columns={"count": "Numero Studenti", "mean": "Media %", "min": "Min %", "max": "Max %"})
)

# ---------------------------------------------
# 9. VALUTAZIONE DEL MODELLO
# ---------------------------------------------
st.subheader("📑 Performance del Modello")
st.text(classification_report(y_test, preds))
