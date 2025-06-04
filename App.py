import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configurazione pagina
st.set_page_config(page_title="Analisi Churn Studenti", page_icon="📉", layout="wide")

st.title("📉 Analisi del Rischio Churn Studenti")
st.markdown("""
**Obiettivo**: Costruire un modello predittivo per identificare gli studenti a rischio di abbandono.
Questo tutorial mostra passo-passo come sviluppare un sistema di early warning per il churn accademico.
""")

# Sidebar per controlli
st.sidebar.header("⚙️ Parametri del Modello")
n_students = st.sidebar.slider("Numero di studenti nel dataset", 200, 1000, 500)
test_size = st.sidebar.slider("Percentuale dati test", 0.1, 0.4, 0.25)
random_state = st.sidebar.number_input("Random State", 1, 100, 42)

# ---------------------------------------------
# PASSO 1: CREAZIONE DATASET SINTETICO
# ---------------------------------------------
st.header("🔧 Passo 1: Creazione del Dataset Sintetico")
st.markdown("""
**Cosa facciamo**: Generiamo dati fittizi che simulano le caratteristiche degli studenti universitari.
**Perché**: Necessitiamo di dati realistici per addestrare il modello.
""")

with st.expander("🔍 Dettagli tecnici della generazione dati"):
    st.markdown("""
    - **Variabili demografiche**: età, sesso, regione
    - **Variabili accademiche**: media voti, esami superati, ore studio
    - **Variabili comportamentali**: partecipazione eventi, lavoro part-time
    - **Target**: churned (0=attivo, 1=abbandonato)
    """)

np.random.seed(random_state)

# Generazione dataset più realistico
studenti_df = pd.DataFrame({
    'student_id': np.arange(1, n_students + 1),
    'eta': np.random.normal(22, 3, n_students).astype(int).clip(18, 35),
    'ore_studio_settimanali': np.random.gamma(2, 10, n_students).astype(int).clip(0, 60),
    'numero_esami_superati': np.random.poisson(12, n_students).clip(0, 30),
    'media_voti': np.round(np.random.normal(24, 2.5, n_students), 1).clip(18, 30),
    'partecipazione_eventi': np.random.choice([0, 1], size=n_students, p=[0.6, 0.4]),
    'lavoro_part_time': np.random.choice([0, 1], size=n_students, p=[0.7, 0.3]),
    'carriera_in_corso': np.random.choice([0, 1], size=n_students, p=[0.8, 0.2]),
    'sesso': np.random.choice(['M', 'F'], size=n_students),
    'regione_residenza': np.random.choice(['Lombardia', 'Lazio', 'Campania', 'Sicilia', 'Veneto', 'Altro'], 
                                        size=n_students, p=[0.25, 0.20, 0.15, 0.12, 0.13, 0.15]),
    'tipo_corso': np.random.choice(['Triennale', 'Magistrale'], size=n_students, p=[0.7, 0.3]),
    'gruppo_corso': np.random.choice(['Economia', 'Ingegneria', 'Lettere', 'Scienze'], 
                                   size=n_students, p=[0.3, 0.25, 0.2, 0.25])
})

# Generazione target più realistica (correlata alle features)
churn_prob = (
    0.1 +  # probabilità base
    0.3 * (studenti_df['ore_studio_settimanali'] < 10) +  # poche ore studio
    0.2 * (studenti_df['media_voti'] < 22) +  # voti bassi
    0.15 * (studenti_df['numero_esami_superati'] < 8) +  # pochi esami
    0.1 * (studenti_df['partecipazione_eventi'] == 0) +  # non partecipa
    0.1 * (studenti_df['lavoro_part_time'] == 1)  # lavora
).clip(0, 0.8)

studenti_df['churned'] = np.random.binomial(1, churn_prob, n_students)

# Visualizzazione dataset
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Prime 10 righe del dataset")
    st.dataframe(studenti_df.head(10))

with col2:
    st.subheader("📈 Statistiche descrittive")
    st.dataframe(studenti_df.describe())

# Grafici esplorativi
st.subheader("🔍 Analisi Esplorativa dei Dati")
col1, col2, col3 = st.columns(3)

with col1:
    fig_churn = px.pie(values=studenti_df['churned'].value_counts().values, 
                       names=['Attivi', 'Abbandonati'], 
                       title='Distribuzione Churn')
    st.plotly_chart(fig_churn, use_container_width=True)

with col2:
    fig_eta = px.histogram(studenti_df, x='eta', color='churned', 
                          title='Distribuzione Età per Churn',
                          labels={'churned': 'Churn Status'})
    st.plotly_chart(fig_eta, use_container_width=True)

with col3:
    fig_voti = px.box(studenti_df, x='churned', y='media_voti', 
                     title='Media Voti vs Churn',
                     labels={'churned': 'Churn Status', 'media_voti': 'Media Voti'})
    st.plotly_chart(fig_voti, use_container_width=True)

# ---------------------------------------------
# PASSO 2: PREPROCESSING DEI DATI
# ---------------------------------------------
st.header("🔄 Passo 2: Preprocessing dei Dati")
st.markdown("""
**Cosa facciamo**: Trasformiamo i dati in un formato utilizzabile dal modello.
**Perché**: Gli algoritmi di ML richiedono dati numerici e ben strutturati.
""")

with st.expander("🔍 Dettagli del preprocessing"):
    st.markdown("""
    - **One-hot encoding**: Trasforma variabili categoriche in binarie
    - **Separazione X e y**: Features (X) e target (y)
    - **Esempio**: 'sesso_M' = 1 se maschio, 0 se femmina
    """)

# Separazione features e target
X = studenti_df.drop(columns=["churned", "student_id"])
y = studenti_df["churned"]

st.subheader("📋 Prima del preprocessing")
col1, col2 = st.columns(2)
with col1:
    st.write("**Features originali:**")
    st.dataframe(X.head())
with col2:
    st.write("**Target:**")
    st.dataframe(y.head())

# One-hot encoding
X_encoded = pd.get_dummies(X, columns=["sesso", "regione_residenza", "tipo_corso", "gruppo_corso"], drop_first=True)

st.subheader("📋 Dopo il preprocessing")
st.write(f"**Dimensioni**: {X_encoded.shape[0]} righe × {X_encoded.shape[1]} colonne")
st.dataframe(X_encoded.head())

# Mostra l'effetto dell'encoding
st.subheader("🎯 Esempio di One-Hot Encoding")
esempio_encoding = pd.DataFrame({
    'Originale': ['Lombardia', 'Lazio', 'Campania'],
    'regione_residenza_Lazio': [0, 1, 0],
    'regione_residenza_Campania': [0, 0, 1],
    'regione_residenza_Lombardia': [1, 0, 0]
})
st.dataframe(esempio_encoding)

# ---------------------------------------------
# PASSO 3: SPLIT TRAIN/TEST
# ---------------------------------------------
st.header("✂️ Passo 3: Suddivisione Train/Test")
st.markdown("""
**Cosa facciamo**: Dividiamo i dati in set di training e test.
**Perché**: Per valutare le prestazioni del modello su dati "non visti".
""")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=test_size, stratify=y, random_state=random_state
)

col1, col2 = st.columns(2)
with col1:
    st.metric("📚 Training Set", f"{len(X_train)} studenti", f"{len(X_train)/len(X_encoded)*100:.1f}%")
    st.metric("🎯 Churn Training", f"{y_train.sum()}", f"{y_train.mean()*100:.1f}%")

with col2:
    st.metric("🧪 Test Set", f"{len(X_test)} studenti", f"{len(X_test)/len(X_encoded)*100:.1f}%")
    st.metric("🎯 Churn Test", f"{y_test.sum()}", f"{y_test.mean()*100:.1f}%")

# Visualizzazione split
fig_split = px.bar(
    x=['Training', 'Test'],
    y=[len(X_train), len(X_test)],
    title='Suddivisione dei Dati',
    labels={'x': 'Set', 'y': 'Numero Studenti'}
)
st.plotly_chart(fig_split, use_container_width=True)

# ---------------------------------------------
# PASSO 4: ADDESTRAMENTO MODELLO
# ---------------------------------------------
st.header("🤖 Passo 4: Addestramento Random Forest")
st.markdown("""
**Cosa facciamo**: Addestriamo un classificatore Random Forest.
**Perché**: RF è robusto, gestisce bene overfitting e fornisce feature importance.
""")

# Parametri modello nella sidebar
st.sidebar.subheader("Parametri Random Forest")
n_estimators = st.sidebar.slider("Numero di alberi", 10, 200, 100)
max_depth = st.sidebar.slider("Profondità massima", 3, 20, 10)
min_samples_leaf = st.sidebar.slider("Min campioni per foglia", 1, 20, 5)

model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    random_state=random_state
)

with st.spinner("🔄 Addestramento in corso..."):
    model.fit(X_train, y_train)

st.success("✅ Modello addestrato con successo!")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

st.subheader("📊 Importanza delle Features")
fig_importance = px.bar(
    feature_importance.head(10),
    x='importance',
    y='feature',
    orientation='h',
    title='Top 10 Features più Importanti'
)
fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig_importance, use_container_width=True)

# ---------------------------------------------
# PASSO 5: PREDIZIONI
# ---------------------------------------------
st.header("🎯 Passo 5: Predizioni sul Test Set")
st.markdown("""
**Cosa facciamo**: Utilizziamo il modello per fare predizioni sui dati test.
**Perché**: Per valutare l'accuratezza del modello su dati non visti durante il training.
""")

# Predizioni
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Metriche di performance
accuracy = (y_pred == y_test).mean()
auc_score = roc_auc_score(y_test, y_proba)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🎯 Accuratezza", f"{accuracy:.3f}", f"{accuracy*100:.1f}%")
with col2:
    st.metric("📈 AUC Score", f"{auc_score:.3f}")
with col3:
    st.metric("🔍 Predizioni Totali", len(y_pred))

# Matrice di confusione
st.subheader("🔍 Matrice di Confusione")
cm = confusion_matrix(y_test, y_pred)
fig_cm = px.imshow(cm, 
                   text_auto=True, 
                   aspect="auto",
                   title="Matrice di Confusione",
                   labels=dict(x="Predetto", y="Reale", color="Conteggio"))
fig_cm.update_xaxes(tickmode='array', tickvals=[0, 1], ticktext=['Non Churn', 'Churn'])
fig_cm.update_yaxes(tickmode='array', tickvals=[0, 1], ticktext=['Non Churn', 'Churn'])
st.plotly_chart(fig_cm, use_container_width=True)

# Curva ROC
st.subheader("📈 Curva ROC")
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig_roc = px.line(x=fpr, y=tpr, title=f'Curva ROC (AUC = {auc_score:.3f})')
fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, 
                  line=dict(dash='dash', color='red'))
fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
st.plotly_chart(fig_roc, use_container_width=True)

# ---------------------------------------------
# PASSO 6: ANALISI RISULTATI
# ---------------------------------------------
st.header("📊 Passo 6: Analisi dei Risultati")
st.markdown("""
**Cosa facciamo**: Analizziamo le probabilità di churn e creiamo categorie di rischio.
**Perché**: Per identificare facilmente gli studenti a rischio e pianificare interventi.
""")

# Creazione dataset risultati
results_df = X_test.copy()
results_df['student_id'] = studenti_df.loc[X_test.index, 'student_id'].values
results_df['churn_real'] = y_test.values
results_df['churn_pred'] = y_pred
results_df['churn_prob'] = np.round(y_proba * 100, 1)

# Categorizzazione rischio
results_df['categoria_rischio'] = pd.cut(
    y_proba,
    bins=[-np.inf, 0.3, 0.7, np.inf],
    labels=["🟢 Basso Rischio", "🟡 Medio Rischio", "🔴 Alto Rischio"]
)

# Filtri interattivi
st.subheader("🎛️ Filtri Interattivi")
col1, col2 = st.columns(2)
with col1:
    risk_filter = st.selectbox(
        "Filtra per categoria di rischio:",
        ["Tutti"] + list(results_df['categoria_rischio'].unique())
    )
with col2:
    min_prob = st.slider("Probabilità minima di churn (%)", 0, 100, 0)

# Applicazione filtri
filtered_results = results_df.copy()
if risk_filter != "Tutti":
    filtered_results = filtered_results[filtered_results['categoria_rischio'] == risk_filter]
filtered_results = filtered_results[filtered_results['churn_prob'] >= min_prob]

# Visualizzazione risultati filtrati
st.subheader(f"📋 Risultati Filtrati ({len(filtered_results)} studenti)")
display_cols = ['student_id', 'churn_prob', 'categoria_rischio', 'churn_real', 'churn_pred']
st.dataframe(filtered_results[display_cols].sort_values('churn_prob', ascending=False))

# Distribuzione categorie di rischio
st.subheader("📊 Distribuzione Categorie di Rischio")
risk_counts = results_df['categoria_rischio'].value_counts()
fig_risk = px.pie(values=risk_counts.values, names=risk_counts.index, 
                  title='Distribuzione Studenti per Categoria di Rischio')
st.plotly_chart(fig_risk, use_container_width=True)

# Statistiche per categoria
st.subheader("📈 Statistiche per Categoria di Rischio")
risk_stats = results_df.groupby('categoria_rischio').agg({
    'churn_prob': ['count', 'mean', 'min', 'max'],
    'churn_real': 'mean'
}).round(2)
risk_stats.columns = ['Numero Studenti', 'Media Prob %', 'Min Prob %', 'Max Prob %', 'Churn Rate Reale']
st.dataframe(risk_stats)

# ---------------------------------------------
# PASSO 7: VALIDAZIONE MODELLO
# ---------------------------------------------
st.header("✅ Passo 7: Validazione del Modello")
st.markdown("""
**Cosa facciamo**: Valutiamo le prestazioni del modello con metriche dettagliate.
**Perché**: Per verificare che il modello sia affidabile e utilizzabile in produzione.
""")

# Report di classificazione
st.subheader("📊 Report di Classificazione")
class_report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(class_report).transpose()
st.dataframe(report_df.round(3))

# Analisi errori
st.subheader("🔍 Analisi degli Errori")
false_positives = results_df[(results_df['churn_real'] == 0) & (results_df['churn_pred'] == 1)]
false_negatives = results_df[(results_df['churn_real'] == 1) & (results_df['churn_pred'] == 0)]

col1, col2 = st.columns(2)
with col1:
    st.metric("❌ Falsi Positivi", len(false_positives), 
              f"{len(false_positives)/len(results_df)*100:.1f}%")
with col2:
    st.metric("❌ Falsi Negativi", len(false_negatives),
              f"{len(false_negatives)/len(results_df)*100:.1f}%")

# Distribuzione probabilità per classe reale
st.subheader("📊 Distribuzione Probabilità per Classe Reale")
fig_dist = px.histogram(results_df, x='churn_prob', color='churn_real',
                       title='Distribuzione Probabilità di Churn per Classe Reale',
                       labels={'churn_real': 'Classe Reale', 'churn_prob': 'Probabilità Churn (%)'})
st.plotly_chart(fig_dist, use_container_width=True)

# ---------------------------------------------
# PASSO 8: RACCOMANDAZIONI
# ---------------------------------------------
st.header("💡 Passo 8: Raccomandazioni e Prossimi Passi")
st.markdown("""
**Cosa abbiamo imparato**: Il modello può identificare studenti a rischio con buona accuratezza.
**Come utilizzarlo**: Implementare interventi mirati per le diverse categorie di rischio.
""")

# Studenti ad alto rischio
high_risk_students = results_df[results_df['categoria_rischio'] == '🔴 Alto Rischio']
st.subheader(f"🚨 Studenti ad Alto Rischio ({len(high_risk_students)} studenti)")

if len(high_risk_students) > 0:
    st.dataframe(high_risk_students[['student_id', 'churn_prob']].head(10))
    
    # Raccomandazioni specifiche
    st.markdown("""
    **Interventi Consigliati per Studenti ad Alto Rischio:**
    - 📞 Contatto diretto con tutor accademico
    - 📚 Sessioni di supporto allo studio personalizzate
    - 💼 Consulenza per gestione lavoro-studio
    - 🎯 Piani di studio individuali
    """)

# Insight dalle feature importance
st.subheader("🔍 Insight dalle Feature più Importanti")
top_features = feature_importance.head(5)
for idx, row in top_features.iterrows():
    st.markdown(f"**{row['feature']}**: {row['importance']:.3f}")

st.markdown("""
**Prossimi Passi:**
1. 🔄 Monitoraggio continuo delle performance
2. 📊 Raccolta feedback sugli interventi
3. 🔧 Ottimizzazione iperparametri
4. 📈 Integrazione con sistemi informativi esistenti
5. 🎯 Sviluppo di dashboard operativi
""")

# Footer
st.markdown("---")
st.markdown("🎓 **Analisi Churn Studenti** - Sviluppato per supportare la retention accademica")
