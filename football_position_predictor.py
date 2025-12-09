import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# Obsługa TensorFlow
try:
    import tensorflow as tf

    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    st.error("TensorFlow nie jest zainstalowany!")
    st.stop()

# Konfiguracja strony
st.set_page_config(
    page_title="Football Position Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS + Font Awesome
st.markdown("""
<!-- Font Awesome CDN -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<style>
    /* Główne tło gradientowe */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Główny kontener */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Karty z cieniami */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }

    /* Gradient card */
    .gradient-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 10px 30px rgba(102,126,234,0.3);
        margin-bottom: 1rem;
    }

    .gradient-card-pink {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 20px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 10px 30px rgba(240,147,251,0.3);
        margin-bottom: 1rem;
    }

    /* Tytuły */
    h1 {
        color: white !important;
        font-weight: 700 !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        margin-bottom: 2rem !important;
    }

    h2, h3 {
        color: #2d3748 !important;
        font-weight: 600 !important;
    }

    /* Przyciski */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.6);
    }

    /* Input fields */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }

    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
    }

    /* Labels */
    label {
        color: #2d3748 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Metryki */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #2d3748 !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #4a5568 !important;
    }

    [data-testid="stMetric"] {
        background: transparent !important;
        padding: 0 !important;
        margin-bottom: 1.5rem !important;
    }

    [data-testid="metric-container"] {
        background: transparent !important;
    }

    /* Wyrównanie kolumn */
    [data-testid="column"] {
        padding: 0 1rem !important;
    }

    /* Odstępy */
    [data-testid="stVerticalBlock"] > div {
        gap: 1rem;
    }

    /* Font Awesome ikony */
    .fas, .fa, .fa-solid {
        margin-right: 0.5rem;
        vertical-align: middle;
    }

    h2 .fas, h3 .fas, h2 .fa-solid, h3 .fa-solid {
        font-size: 1em;
    }

    /* Duże ikony modeli w kartach wyników */
    .metric-card .fa-solid {
        font-size: 3rem;
        color: #667eea;
        margin-right: 0;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* Success/Info/Warning */
    .stSuccess, .stInfo, .stWarning {
        background: rgba(255,255,255,0.95) !important;
        border-radius: 12px !important;
        border-left: 4px solid #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# Mapowanie pozycji
POSITION_NAMES = {
    'GK': 'Bramkarz (Goalkeeper)',
    'DF': 'Obrońca (Defender)',
    'MF': 'Pomocnik (Midfielder)',
    'FW': 'Napastnik (Forward)'
}


# Wskaźniki
def calculate_indicators(data):
    """Automatyczne obliczanie wskaźników"""

    if data['90s'] > 0:
        data['Gls_per90'] = data['Gls'] / data['90s']
        data['Ast_per90'] = data['Ast'] / data['90s']
        data['Sh_per90'] = data['Sh'] / data['90s']
        data['Tkl_per90'] = data['Tkl'] / data['90s']
        data['Int_per90'] = data['Int'] / data['90s']
        data['Carries_per90'] = data['Carries'] / data['90s']
    else:
        data['Gls_per90'] = 0
        data['Ast_per90'] = 0
        data['Sh_per90'] = 0
        data['Tkl_per90'] = 0
        data['Int_per90'] = 0
        data['Carries_per90'] = 0

    if data['Att'] > 0:
        data['Cmp%'] = (data['Cmp'] / data['Att']) * 100
    else:
        data['Cmp%'] = 0

    # offensive index - dzielone przez 90s
    if data['90s'] > 0:
        data['offensive_index'] = (data['Gls'] + data['Ast'] + (data['Sh'] / 10)) / data['90s']
    else:
        data['offensive_index'] = 0

    # defensive index - dzielone przez 90s
    if data['90s'] > 0:
        data['defensive_index'] = (data['Tkl'] + data['Int']) / data['90s']
    else:
        data['defensive_index'] = 0

    # passing quality
    if data['90s'] > 0 and data['Att'] > 0:
        data['passing_quality'] = (data['Cmp%'] / 100) * (data['Att'] / data['90s']) / 10
    else:
        data['passing_quality'] = 0

    # goal efficiency
    if data['Sh'] > 0:
        data['goal_efficiency'] = data['Gls'] / data['Sh']
    else:
        data['goal_efficiency'] = 0

    return data


# Wczytywanie modeli
@st.cache_resource
def load_models():
    """Wczytuje wszystkie modele"""
    models = {}
    errors = []

    try:
        with open('feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        models['feature_cols'] = feature_cols
    except Exception as e:
        errors.append(f"Feature columns: {e}")
        return None, errors

    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        models['scaler'] = scaler
    except Exception as e:
        st.warning(f"Problem ze scalerem: {e}")
        models['scaler'] = None

    try:
        with open('random_forest_models.pkl', 'rb') as f:
            rf_models = pickle.load(f)
        models['rf_models'] = rf_models
    except Exception as e:
        st.warning(f"Nie można wczytać Random Forest: {e}")
        models['rf_models'] = None

    try:
        with open('xgboost_models.pkl', 'rb') as f:
            xgb_models = pickle.load(f)
        models['xgb_models'] = xgb_models
    except Exception as e:
        st.warning(f"Nie można wczytać XGBoost: {e}")
        models['xgb_models'] = None

    try:
        mlp_model = tf.keras.models.load_model('mlp_model.keras', compile=False)
        models['mlp_model'] = mlp_model
    except Exception as e:
        st.warning(f"Nie można wczytać MLP: {e}")
        models['mlp_model'] = None

    available_models = sum([
        models.get('rf_models') is not None,
        models.get('xgb_models') is not None,
        models.get('mlp_model') is not None
    ])

    if available_models == 0:
        return None, ["Nie udało się wczytać żadnego modelu!"]

    return models, []


def get_predicted_position(predictions_dict):
    return max(predictions_dict, key=predictions_dict.get)


def predict_all_models(input_data, models):
    """Wykonuje predykcję"""

    feature_cols = models['feature_cols']
    df = pd.DataFrame([input_data])
    df = df[feature_cols]

    if models['scaler'] is not None:
        X_scaled = models['scaler'].transform(df)
    else:
        X_scaled = df.values

    results = {}

    if models['rf_models'] is not None:
        try:
            rf_preds = {}
            for pos, model in models['rf_models'].items():
                rf_preds[pos] = float(model.predict(X_scaled)[0])
            rf_position = get_predicted_position(rf_preds)
            results['Random Forest'] = {
                'position': rf_position,
                'probabilities': rf_preds,
                'confidence': rf_preds[rf_position]
            }
        except Exception as e:
            st.error(f"Błąd Random Forest: {e}")

    if models['xgb_models'] is not None:
        try:
            xgb_preds = {}
            for pos, model in models['xgb_models'].items():
                xgb_preds[pos] = float(model.predict(X_scaled)[0])
            xgb_position = get_predicted_position(xgb_preds)
            results['XGBoost'] = {
                'position': xgb_position,
                'probabilities': xgb_preds,
                'confidence': xgb_preds[xgb_position]
            }
        except Exception as e:
            st.error(f"Błąd XGBoost: {e}")

    if models['mlp_model'] is not None:
        try:
            mlp_pred = models['mlp_model'].predict(X_scaled, verbose=0)[0]
            mlp_preds = {
                'GK': float(mlp_pred[0]),
                'DF': float(mlp_pred[1]),
                'MF': float(mlp_pred[2]),
                'FW': float(mlp_pred[3])
            }
            mlp_position = get_predicted_position(mlp_preds)
            results['MLP'] = {
                'position': mlp_position,
                'probabilities': mlp_preds,
                'confidence': mlp_preds[mlp_position]
            }
        except Exception as e:
            st.error(f"Błąd MLP: {e}")

    return results


# === GŁÓWNA APKA ===

# Tytuł
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <h1><i class="fas fa-futbol"></i> Football Position Predictor</h1>
    <p style='color: rgba(255,255,255,0.9); font-size: 1.1rem;'>Predykcja pozycji piłkarza przy użyciu Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Wczytanie modeli
with st.spinner('Wczytywanie modeli...'):
    models, errors = load_models()

if models is None:
    st.error("Nie udało się wczytać modeli!")
    for error in errors:
        st.error(error)
    st.stop()

# Status modeli - musi to być bo bywa różnie
available = sum([
    models.get('rf_models') is not None,
    models.get('xgb_models') is not None,
    models.get('mlp_model') is not None
])

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(f"""
    <div class='metric-card' style='text-align: center;'>
        <p style='margin: 0; color: #667eea; font-weight: 600;'>Status systemu</p>
        <h2 style='margin: 0.5rem 0;'>{available}/3 modeli aktywnych</h2>
        <div style='width: 100%; background: #e2e8f0; border-radius: 10px; height: 8px; margin-top: 1rem;'>
            <div style='width: {available / 3 * 100}%; background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; border-radius: 10px;'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# FORMULARZ
with st.form("player_stats"):
    st.markdown("""
    <div class='gradient-card'>
        <h2 style='color: white; margin: 0;'><i class="fas fa-user"></i> Statystyki gracza</h2>
        <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>Wprowadź podstawowe dane - wskaźniki zostaną obliczone automatycznie</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### <i class='fas fa-user'></i> Podstawowe", unsafe_allow_html=True)
        age = st.number_input("Wiek", min_value=15.0, max_value=45.0, value=25.0, step=0.1)
        mins_90s = st.number_input("90s (pełne mecze)", min_value=0.0, max_value=50.0, value=30.0, step=1.0)

        st.markdown("### <i class='fas fa-shield-alt'></i> Defensywne", unsafe_allow_html=True)
        tackles = st.number_input("Wślizgi", min_value=0, max_value=200, value=30, step=1)
        interceptions = st.number_input("Przechwyty", min_value=0, max_value=200, value=20, step=1)

    with col2:
        st.markdown("### <i class='fas fa-futbol'></i> Ofensywne", unsafe_allow_html=True)
        goals = st.number_input("Gole", min_value=0, max_value=100, value=5, step=1)
        assists = st.number_input("Asysty", min_value=0, max_value=50, value=3, step=1)
        xg = st.number_input("Expected Goals (xG)", min_value=0.0, max_value=50.0, value=4.5, step=0.1)
        xag = st.number_input("Expected Assists (xAG)", min_value=0.0, max_value=30.0, value=2.5, step=0.1)
        shots = st.number_input("Strzały", min_value=0, max_value=300, value=40, step=1)
        sot = st.number_input("Strzały celne", min_value=0, max_value=150, value=15, step=1)

    with col3:
        st.markdown("### <i class='fas fa-chart-bar'></i> Podania", unsafe_allow_html=True)
        passes_cmp = st.number_input("Podania celne", min_value=0, max_value=3000, value=500, step=10)
        passes_att = st.number_input("Próby podań", min_value=0, max_value=3500, value=600, step=10)
        prgp = st.number_input("Progresywne podania", min_value=0, max_value=300, value=60, step=1)

        st.markdown("### <i class='fas fa-bolt'></i> Prowadzenie", unsafe_allow_html=True)
        carries = st.number_input("Prowadzenia", min_value=0, max_value=1500, value=300, step=10)
        prgc = st.number_input("Progresywne prowadzenia", min_value=0, max_value=300, value=50, step=1)
        prgr = st.number_input("Progresywne przyjęcia", min_value=0, max_value=300, value=70, step=1)

    submitted = st.form_submit_button("PRZEWIDUJ POZYCJĘ", use_container_width=True)

# PREDYKCJA
if submitted:
    basic_data = {
        'Age': age, '90s': mins_90s, 'Gls': float(goals), 'Ast': float(assists),
        'xG': xg, 'xAG': xag, 'Sh': float(shots), 'SoT': float(sot),
        'Tkl': float(tackles), 'Int': float(interceptions), 'Cmp': float(passes_cmp),
        'Att': float(passes_att), 'Carries': float(carries),
        'PrgC': float(prgc), 'PrgP': float(prgp), 'PrgR': float(prgr)
    }

    with st.spinner('Obliczanie wskaźników...'):
        input_data = calculate_indicators(basic_data.copy())

    # Obliczone wskaźniki
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='gradient-card-pink'>
        <h2 style='color: white; margin: 0;'>Obliczone wskaźniki</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Gole/90", f"{input_data['Gls_per90']:.2f}")
        st.metric("Asysty/90", f"{input_data['Ast_per90']:.2f}")

    with col2:
        st.metric("Strzały/90", f"{input_data['Sh_per90']:.2f}")
        st.metric("Celność podań", f"{input_data['Cmp%']:.1f}%")

    with col3:
        st.metric("Indeks ofensywny", f"{input_data['offensive_index']:.1f}")
        st.metric("Indeks defensywny", f"{input_data['defensive_index']:.1f}")

    with col4:
        st.metric("Wślizgi/90", f"{input_data['Tkl_per90']:.2f}")
        st.metric("Przechwyty/90", f"{input_data['Int_per90']:.2f}")

    # Predykcja
    with st.spinner('Wykonywanie predykcji...'):
        results = predict_all_models(input_data, models)

    if not results:
        st.error("Nie udało się wykonać predykcji.")
        st.stop()

    st.markdown("<br>", unsafe_allow_html=True)

    # Wyniki
    st.markdown("""
    <div class='gradient-card'>
        <h2 style='color: white; margin: 0;'>Wyniki predykcji</h2>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(len(results))

    model_icons = {
        'Random Forest': '<i class="fa-solid fa-tree"></i>',
        'XGBoost': '<i class="fa-solid fa-rocket"></i>',
        'MLP': '<i class="fa-solid fa-brain"></i>'
    }

    for idx, (model_name, data) in enumerate(results.items()):
        with cols[idx]:
            pos = data['position']
            conf = data['confidence']

            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>
                    {model_icons.get(model_name)}
                </div>
                <h3 style='margin: 0;'>{model_name}</h3>
                <div style='margin: 1rem 0;'>
                    <h2 style='margin: 0;'>{POSITION_NAMES[pos]}</h2>
                </div>
                <p style='color: #667eea; font-size: 1.5rem; font-weight: 700; margin: 0;'>{conf:.1%}</p>
                <p style='color: #718096; margin: 0;'>pewność</p>
            </div>
            """, unsafe_allow_html=True)

    # Konsensus
    st.markdown("<br>", unsafe_allow_html=True)
    positions = [data['position'] for data in results.values()]

    if len(set(positions)) == 1:
        st.markdown(f"""
        <div class='gradient-card' style='text-align: center;'>
            <h2 style='color: white; margin: 0;'>KOŃCOWA DECYZJA</h2>
            <h1 style='color: white; margin: 1rem 0;'>{POSITION_NAMES[positions[0]]}</h1>
            <p style='color: rgba(255,255,255,0.9);'>Wszystkie modele są zgodne!</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        from collections import Counter

        vote_count = Counter(positions)
        most_common = vote_count.most_common(1)[0]

        if most_common[1] >= 2:
            st.markdown(f"""
            <div class='gradient-card-pink' style='text-align: center;'>
                <h2 style='color: white; margin: 0;'>KOŃCOWA DECYZJA</h2>
                <h1 style='color: white; margin: 1rem 0;'>{POSITION_NAMES[most_common[0]]}</h1>
                <p style='color: rgba(255,255,255,0.9);'>Większość modeli ({most_common[1]}/{len(results)}) wskazuje tę pozycję</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Modele nie są zgodne - gracz może być uniwersalny!")

    # Szczegóły
    with st.expander("Zobacz szczegółowe prawdopodobieństwa"):
        for model_name, data in results.items():
            st.markdown(f"#### {model_name}")
            cols = st.columns(4)
            probs = data['probabilities']
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

            for idx, (pos, prob) in enumerate(sorted_probs):
                with cols[idx]:
                    if pos == data['position']:
                        st.success(f"**{pos}**")
                        st.metric("", f"{prob:.2%}", "✓ WYBRANE")
                    else:
                        st.write(f"{pos}")
                        st.metric("", f"{prob:.2%}")
            st.markdown("---")

    # Wykres porównawczy
    if len(results) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-card'>
            <h2 style='margin-top: 0;'><i class="fas fa-chart-area"></i> Wykres porównawczy</h2>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()
        positions_order = ['GK', 'DF', 'MF', 'FW']
        colors = ['#667eea', '#764ba2', '#f093fb']

        for idx, (model_name, data) in enumerate(results.items()):
            probs = data['probabilities']
            values = [probs[pos] for pos in positions_order]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=positions_order,
                fill='toself',
                name=model_name,
                line=dict(color=colors[idx % len(colors)], width=2)
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(102,126,234,0.2)'),
                bgcolor='rgba(255,255,255,0.9)'
            ),
            showlegend=True,
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, color='#2d3748')
        )
        st.plotly_chart(fig, use_container_width=True)

# Fancy footer
st.markdown("""
<div style='text-align: center; margin-top: 3rem; color: rgba(255,255,255,0.7);'>
    <p><i class="fas fa-code"></i> Dashboard by Karolina | Machine Learning Position Predictor</p>
</div>
""", unsafe_allow_html=True)