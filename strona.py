import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import sqlite3
import shap
import matplotlib.pyplot as plt
from datetime import datetime

# Konfiguracja strony
st.set_page_config(page_title="Kliniczny System Onkologiczny xAI", page_icon="🏥", layout="wide")

# ==========================================
# 0. INICJALIZACJA BAZY DANYCH SQLITE
# ==========================================
def init_db():
    os.makedirs('metrics', exist_ok=True)
    conn = sqlite3.connect('metrics/patients_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            patient_name TEXT,
            age REAL,
            tumor_size REAL,
            risk_score REAL,
            risk_category TEXT,
            survival_5y TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_to_db(name, age, tumor, risk, category, surv):
    conn = sqlite3.connect('metrics/patients_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO history (timestamp, patient_name, age, tumor_size, risk_score, risk_category, survival_5y)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, age, tumor, float(risk), category, surv))
    conn.commit()
    conn.close()

# Ładowanie artefaktów
@st.cache_resource
def load_artifacts():
    try:
        model = xgb.Booster()
        model.load_model('metrics/xgb_survival_model.json')
        scaler = joblib.load('metrics/scaler.pkl')
        poly = joblib.load('metrics/poly.pkl')
        top_features = joblib.load('metrics/top_features.pkl')
        num_cols = joblib.load('metrics/num_cols.pkl')
        cat_cols = joblib.load('metrics/cat_cols.pkl')
        explainer = joblib.load('metrics/shap_explainer.pkl')
        return model, scaler, poly, top_features, num_cols, cat_cols, explainer
    except Exception as e:
        st.error(f"Błąd ładowania artefaktów: {e}. Uruchom najpierw skrypt trenujący (benchmark.py).")
        return None, None, None, None, None, None, None

model, scaler, poly, top_features, num_cols, cat_cols, explainer = load_artifacts()

if 'risk_score' not in st.session_state:
    st.session_state.risk_score = None
if 'scaled_df' not in st.session_state:
    st.session_state.scaled_df = None
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {}

# Układ zakładek
tab1, tab2, tab3, tab4 = st.tabs(["ℹ️ Informacje", "📝 Formularz Pacjenta", "📊 Statystyki, xAI i Krzywa Przeżycia", "📂 Historia Pacjentów (DB)"])

with tab1:
    st.header("Kliniczny System Wspomagania Decyzji (CDSS)")
    st.markdown("""
    System wykorzystuje zaawansowane modele uczenia maszynowego (XGBoost Survival) zoptymalizowane za pomocą algorytmów bayesowskich.
    
    ### Kluczowe funkcje systemu:
    1. **xAI (SHAP):** Transparentność decyzji oparta o analizę wpływów lokalnych cech klinicznych.
    2. **Krzywe Przeżycia w Czasie:** Symulacja spadku prawdopodobieństwa przeżycia w horyzoncie 120 miesięcy.
    3. **Baza Danych SQLite:** Automatyczna archiwizacja wywiadów i konsultacji pacjentów.
    """)

with tab2:
    st.header("Wprowadź dane kliniczne pacjenta")
    
    with st.form("patient_form"):
        patient_name = st.text_input("Imię i nazwisko / ID pacjenta", value="Pacjent_001")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Wiek pacjenta", min_value=18, max_value=100, value=55)
            tumor_size = st.number_input("Rozmiar guza (mm)", min_value=1, max_value=200, value=25)
            nodes_examined = st.number_input("Liczba przebadanych węzłów chłonnych", min_value=0, max_value=50, value=12)
            nodes_positive = st.number_input("Liczba węzłów z przerzutami", min_value=0, max_value=50, value=2)
            
        with col2:
            estrogen = st.selectbox("Receptory Estrogenowe (ER)", ["Positive", "Negative"])
            progesterone = st.selectbox("Receptory Progesteronowe (PR)", ["Positive", "Negative"])
            grade = st.selectbox("Stopień złośliwości (Grade)", [
                "Well differentiated; Grade I", 
                "Moderately differentiated; Grade II", 
                "Poorly differentiated; Grade III", 
                "Undifferentiated; anaplastic; Grade IV"
            ])
            stage = st.selectbox("Kliniczne stadium (T Stage)", ["T1", "T2", "T3", "T4"])
        
        submitted = st.form_submit_button("Uruchom Pełną Analizę Kliniczną")
        
        if submitted and model is not None:
            input_data = pd.DataFrame([{
                'age': age,
                'tumor_size': tumor_size,
                'regional_node_examined': nodes_examined,
                'reginol_node_positive': nodes_positive,
                'estrogen_status': estrogen,
                'progesterone_status': progesterone,
                'grade': grade,
                't_stage': stage
            }])
            
            X_input_encoded = pd.get_dummies(input_data)
            for col in cat_cols:
                if col not in X_input_encoded.columns:
                    X_input_encoded[col] = 0
            
            X_num_input = input_data[num_cols]
            X_num_poly_input = pd.DataFrame(
                poly.transform(X_num_input), 
                columns=poly.get_feature_names_out(num_cols)
            )
            
            X_full_input = pd.concat([X_num_poly_input.reset_index(drop=True), X_input_encoded[cat_cols].reset_index(drop=True)], axis=1)
            for col in top_features:
                if col not in X_full_input.columns:
                    X_full_input[col] = 0
                    
            X_final_input = X_full_input[top_features]
            X_scaled_array = scaler.transform(X_final_input)
            X_scaled_df = pd.DataFrame(X_scaled_array, columns=top_features)
            
            dmatrix_input = xgb.DMatrix(X_scaled_df)
            risk = model.predict(dmatrix_input)[0]
            
            if risk < 0.8:
                risk_level = "Niskie Ryzyko"
                surv_5y = "92% - 96%"
            elif risk < 1.5:
                risk_level = "Umiarkowane Ryzyko"
                surv_5y = "75% - 85%"
            else:
                risk_level = "Wysokie Ryzyko"
                surv_5y = "Poniżej 60%"
            
            save_to_db(patient_name, age, tumor_size, risk, risk_level, surv_5y)
            
            st.session_state.risk_score = risk
            st.session_state.scaled_df = X_scaled_df
            st.session_state.patient_info = {'name': patient_name, 'age': age, 'tumor': tumor_size}
            st.success(f"Analiza dla {patient_name} zakończona pomyślnie i zapisana w bazie.")

with tab3:
    st.header("Wyniki, xAI oraz Krzywa Przeżycia")
    
    if st.session_state.risk_score is not None:
        risk = st.session_state.risk_score
        p_info = st.session_state.patient_info
        
        if risk < 0.8:
            risk_level, color, survival_5y, median_life, progress = "Niskie Ryzyko", "green", "92% - 96%", "Powyżej 10 lat", 0.25
        elif risk < 1.5:
            risk_level, color, survival_5y, median_life, progress = "Umiarkowane Ryzyko", "orange", "75% - 85%", "7 - 9 lat", 0.50
        else:
            risk_level, color, survival_5y, median_life, progress = "Wysokie Ryzyko", "red", "Poniżej 60%", "Mniej niż 5 lat", 0.85
            
        st.markdown(f"### Pacjent: **{p_info.get('name')}** | Kategoria: **<span style='color:{color}'>{risk_level}</span>** (HR: {risk:.2f})", unsafe_allow_html=True)
        st.progress(progress)
        
        st.markdown("---")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Szacowane przeżycie 5-letnie", survival_5y)
        with col_m2:
            st.metric("Przewidywana mediana życia", median_life)
        with col_m3:
            st.metric("Współczynnik ryzyka (HR)", f"{risk:.2f}")
            
        # Krzywa przeżycia w czasie
        st.markdown("---")
        st.subheader("📉 Indywidualna Krzywa Przeżycia w Czasie")
        months = np.arange(0, 121, 5)
        base_survival = np.exp(-0.002 * months)
        patient_survival = np.power(base_survival, risk) * 100
        
        fig_surv, ax_surv = plt.subplots(figsize=(8, 4))
        ax_surv.plot(months, patient_survival, color=color, linewidth=3, label=f"Pacjent ({risk_level})")
        ax_surv.axhline(50, color='gray', linestyle='--', alpha=0.7, label='Próg mediany (50%)')
        ax_surv.set_title("Prognozowane Prawdopodobieństwo Przeżycia (%) w Czasie")
        ax_surv.set_xlabel("Miesiące od diagnozy")
        ax_surv.set_ylabel("Przeżycie (%)")
        ax_surv.set_ylim(0, 105)
        ax_surv.legend()
        ax_surv.grid(True, alpha=0.3)
        st.pyplot(fig_surv)
        
        # Moduł xAI (SHAP)
        st.markdown("---")
        st.subheader("🔍 Moduł xAI (Analiza SHAP)")
        st.write("Wykres przedstawia wpływ poszczególnych czynników na podbicie lub obniżenie ryzyka pacjenta względem bazy populacyjnej:")
        try:
            shap_values = explainer(st.session_state.scaled_df)
            fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
            shap.plots.bar(shap_values[0], max_display=10, show=False)
            st.pyplot(fig_shap)
        except Exception as e:
            st.info(f"Błąd renderowania wykresu SHAP: {e}")
            
        st.markdown("---")
        st.subheader("🏥 Rekomendowane Następne Kroki")
        if risk_level == "Niskie Ryzyko":
            st.markdown("- Standardowy monitoring kontrolny co 12 miesięcy.\n- Profilaktyka i zdrowy styl życia.")
        elif risk_level == "Umiarkowane Ryzyko":
            st.markdown("- Konsultacja onkologiczna i rozważenie paneli genetycznych.\n- Częstotliwość kontroli co 6 miesięcy.")
        else:
            st.markdown("- Pilna konsultacja konsyliarna.\n- Zaawansowana diagnostyka obrazowa (PET-CT) oraz celowane leczenie systemowe.")
    else:
        st.info("Wypełnij formularz w zakładce 'Formularz Pacjenta', aby wygenerować pełny raport kliniczny.")

with tab4:
    st.header("📂 Historia Konsultacji (Baza Danych SQLite)")
    if os.path.exists('metrics/patients_history.db'):
        conn = sqlite3.connect('metrics/patients_history.db')
        df_history = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
        conn.close()
        
        if not df_history.empty:
            st.dataframe(df_history, use_container_width=True)
            if st.button("Wyczyść historię bazy danych"):
                conn = sqlite3.connect('metrics/patients_history.db')
                conn.execute("DELETE FROM history")
                conn.commit()
                conn.close()
                st.success("Wyczyszczono historię.")
                st.rerun()
        else:
            st.info("Baza danych jest pusta.")
    else:
        st.info("Brak utworzonej bazy danych.")