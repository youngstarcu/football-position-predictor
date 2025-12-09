# football-position-predictor

# URUCHOMIENIE DASHBOARDU
````````bash
pip install numpy==1.26.4 tensorflow==2.19.0 scikit-learn xgboost pandas streamlit plotly
````````

## Podstawowe uruchomienie 
````````bash
streamlit run football_position_predictor.py
````````

# ŚRODOWISKO WIRTUALNE (opcjonalne)
## WINDOWS:

### 1. Stwórz środowisko wirtualne
````````bash
py -m venv football_env
````````

### 2. Aktywuj środowisko
````````bash
football_env\Scripts\activate
````````
### 3. Zainstaluj zależności
````````bash
pip install numpy==1.26.4 tensorflow==2.19.0 scikit-learn xgboost pandas streamlit plotly
````````
### 4. Uruchom dashboard
````````bash
streamlit run football_position_predictor.py
````````
### 5. Dezaktywuj środowisko (gdy skończysz)
````````bash
deactivate
````````