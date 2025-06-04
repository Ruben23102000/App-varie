 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/App.py b/App.py
index a0e927bd42313ad6e0dc2c9fa03a69ade830f184..235fb9dee0a8d4318803dcc5586628fa636365e4 100644
--- a/App.py
+++ b/App.py
@@ -1,108 +1,129 @@
 import streamlit as st
 import pandas as pd
 import numpy as np
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import classification_report
 
 st.title("📉 Analisi del Rischio Churn Studenti")
+# Breve descrizione dei passaggi
+st.markdown("Questo esempio mostra passo passo come costruire un modello di previsione del churn degli studenti.")
+
 
 # ---------------------------------------------
 # 1. CREAZIONE DATASET SINTETICO
+st.markdown("### Passo 1: Creazione del dataset sintetico")
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
 
+st.markdown("Qui sotto mostriamo le prime righe del dataset generato.")
 # Visualizziamo il dataset
 st.subheader("🎓 Dataset Studenti")
 st.dataframe(studenti_df.head())
 
 # ---------------------------------------------
 # 2. PREPROCESSING DEI DATI
+st.markdown("### Passo 2: Preprocessing dei dati")
+st.markdown("Separiamo le variabili e applichiamo il one-hot encoding alle categoriali.")
 # ---------------------------------------------
 # Separiamo variabili indipendenti (X) e target (y)
 X = studenti_df.drop(columns=["churned"])
 y = studenti_df["churned"]
 
 # Codifica variabili categoriche con one-hot encoding
 X = pd.get_dummies(X, columns=["sesso", "regione_residenza", "tipo_corso", "gruppo_corso"], drop_first=True)
 
 # ---------------------------------------------
 # 3. SPLIT TRAIN/TEST
+st.markdown("### Passo 3: Suddivisione dei dati in training e test")
+st.markdown("Distribuiamo i dati in modo stratificato per mantenere la proporzione di churn.")
 # ---------------------------------------------
 X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
 
 # ---------------------------------------------
 # 4. ADDESTRAMENTO MODELLO RANDOM FOREST
+st.markdown("### Passo 4: Addestramento del modello Random Forest")
+st.markdown("Il modello apprende la relazione tra caratteristiche degli studenti e rischio di abbandono.")
 # ---------------------------------------------
 # Creiamo e addestriamo il classificatore
 model = RandomForestClassifier(n_estimators=50, max_depth=6, min_samples_leaf=5, random_state=42)
 model.fit(X_train, y_train)
 
 # ---------------------------------------------
 # 5. PREDIZIONE SU TEST SET
+st.markdown("### Passo 5: Predizione sul set di test")
+st.markdown("Calcoliamo la probabilità di churn e la classe prevista per ogni studente.")
 # ---------------------------------------------
 # Probabilità di churn (classe 1)
 probs = model.predict_proba(X_test)[:, 1]
 # Predizione binaria
 preds = model.predict(X_test)
 
 # ---------------------------------------------
 # 6. COSTRUZIONE RISULTATI CON PERCENTUALI E CATEGORIE
+st.markdown("### Passo 6: Creazione dei risultati e classificazione del rischio")
+st.markdown("Combiniamo le predizioni con le informazioni degli studenti e definiamo le fasce di rischio.")
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
+st.markdown("### Passo 7: Visualizzazione dei risultati")
+st.markdown("Possiamo filtrare gli studenti per livello di rischio e consultare i dettagli.")
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
+st.markdown("### Passo 8: Statistiche di sintesi sui gruppi di rischio")
+st.markdown("Calcoliamo alcune statistiche riassuntive per ciascuna categoria di rischio.")
 # ---------------------------------------------
 st.subheader("📈 Statistiche per Categoria di Rischio")
 st.dataframe(
     results.groupby("categoria_rischio")["churn_prob"]
            .agg(["count", "mean", "min", "max"])
            .rename(columns={"count": "Numero Studenti", "mean": "Media %", "min": "Min %", "max": "Max %"})
 )
 
 # ---------------------------------------------
 # 9. VALUTAZIONE DEL MODELLO
+st.markdown("### Passo 9: Valutazione delle prestazioni del modello")
+st.markdown("Utilizziamo metriche di classificazione per misurare l'accuratezza del modello.")
 # ---------------------------------------------
 st.subheader("📑 Performance del Modello")
 st.text(classification_report(y_test, preds))
 
EOF
)
