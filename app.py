import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from polygon import RESTClient  # Für Live-Daten
from datetime import datetime, timedelta
from scipy import optimize  # Für besseren Optimizer
import io  # Für Exports

# Black-Scholes + Greeks (erweitert mit Error-Handling)
def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    try:
        if T <= 0 or sigma <= 0:
            intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
            return intrinsic, 0, 0, 0, 0, 0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)
        vega = S * np.sqrt(T) * norm.pdf(d1)
        return price, delta, gamma, theta, vega, rho
    except Exception as e:
        st.error(f"Fehler in Black-Scholes: {e}")
        return 0, 0, 0, 0, 0, 0

# Multi-Leg P/L Calc (bis 6 Legs jetzt)
@st.cache_data
def multi_leg_payoff(legs, stock_prices, T_for_calc=None):
    payoff = np.zeros_like(stock_prices)
    for leg in legs:
        S, K, T, r, sigma, opt_type, position = leg
        if T_for_calc is not None:
            T = T_for_calc  # Für Expiry T=0 überschreiben
        price, _, _, _, _, _ = black_scholes_greeks(S, K, T, r, sigma, opt_type)
        intrinsic = np.maximum(stock_prices - K, 0) if opt_type == 'call' else np.maximum(K - stock_prices, 0)
        leg_payoff = (intrinsic - price) * (1 if position == 'long' else -1)
        payoff += leg_payoff
    return payoff

# Erweiterte Strategie-Vorschläge (mehr hinzugefügt)
def suggest_strategies(forecast_price, forecast_vol, risk_tolerance):
    strategies = []
    if forecast_price == "Up":
        strategies.extend([{"name": "Long Call", "legs": 1}, {"name": "Bull Call Spread", "legs": 2}, {"name": "Covered Call", "legs": 2}, {"name": "Diagonal Call Spread", "legs": 2}])
    elif forecast_price == "Down":
        strategies.extend([{"name": "Long Put", "legs": 1}, {"name": "Bear Put Spread", "legs": 2}, {"name": "Cash Secured Put", "legs": 1}, {"name": "Diagonal Put Spread", "legs": 2}])
    elif forecast_price == "Neutral":
        strategies.extend([{"name": "Long Straddle", "legs": 2}, {"name": "Iron Condor", "legs": 4}, {"name": "Butterfly Spread", "legs": 3}, {"name": "Calendar Spread", "legs": 2}])
    if forecast_vol == "Increase":
        strategies.append({"name": "Long Strangle", "legs": 2})
    # Filter nach Risk
    if risk_tolerance == "Low":
        strategies = [s for s in strategies if s['legs'] <= 2]
    elif risk_tolerance == "Medium":
        strategies = [s for s in strategies if s['legs'] <= 4]
    return strategies[:15]  # Mehr Vorschläge

# Polygon Live-Data Fetch (dynamischer für Options Snapshot)
@st.cache_data
def fetch_live_options(ticker, api_key, strike=None, expiry=None, opt_type='C'):
    try:
        client = RESTClient(api_key)
        contract = f'O:{ticker}{expiry or "240101"}{opt_type}{strike or "00150000":08d}'
        today = datetime.today().strftime('%Y-%m-%d')
        aggs = client.get_aggs(contract, 1, 'day', today, today)
        return aggs.close if aggs else 0.0
    except Exception as e:
        st.error(f"Fehler bei Polygon: {e}")
        return 0.0

# Monte-Carlo für Prob-Profit
@st.cache_data
def monte_carlo_prob_profit(legs, S, target_return, num_sim=1000):
    try:
        sim_prices = S * np.exp((st.session_state.r - 0.5 * st.session_state.sigma**2) * st.session_state.T + st.session_state.sigma * np.sqrt(st.session_state.T) * np.random.standard_normal(num_sim))
        payoffs = multi_leg_payoff(legs, sim_prices)
        return np.mean(payoffs > (target_return / 100) * S) * 100
    except:
        return 0.0

# SciPy Optimizer für Strike
def optimize_strike(legs, target_price):
    def objective(K):
        opt_legs = legs.copy()
        opt_legs[0] = (*opt_legs[0][:2], K[0], *opt_legs[0][3:])
        return -multi_leg_payoff(opt_legs, np.array([target_price]))[0]  # Maximieren
    bounds = [(legs[0][1] * 0.9, legs[0][1] * 1.1)]
    result = optimize.minimize(objective, [legs[0][1]], bounds=bounds)
    return result.x[0], -result.fun

# Custom CSS für dunkles Theme mit metallischen Akzenten und metallisch-blauer Schrift
st.markdown("""
<style>
    /* Dunkler Hintergrund */
    .stApp {
        background-color: #121212; /* Dunkelgrau */
        color: #4FC3F7; /* Metallisch-blau für Haupttext (light blue metallic) */
    }
    
    /* Metallische Akzente (z.B. Buttons, Slider) */
    .stButton > button {
        background-color: #A9A9A9; /* Silber-Metallic */
        color: #000000; /* Schwarzer Text */
        border: 1px solid #C0C0C0; /* Hellerer Silber-Rand */
        box-shadow: 0 2px 4px rgba(192,192,192,0.5); /* Leichter Metallic-Schatten */
    }
    
    .stButton > button:hover {
        background-color: #C0C0C0; /* Helleres Silber beim Hover */
    }
    
    .stSlider .stSlider {
        background-color: #333333; /* Dunkler Slider-Hintergrund */
    }
    
    .stSlider .stSlider > div > div > div {
        background-color: #A9A9A9; /* Metallic-Slider-Handle */
    }
    
    /* Tabellen und Metrics mit blauer Schrift */
    .stDataFrame {
        background-color: #1E1E1E; /* Dunkler Tabellen-Hintergrund */
        color: #4FC3F7; /* Metallisch-blau */
    }
    
    .stMetric {
        background-color: #1E1E1E;
        border: 1px solid #A9A9A9; /* Metallic-Rand */
        color: #4FC3F7; /* Metallisch-blau für Metrics */
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0A0A0A; /* Noch dunklerer Sidebar */
        color: #4FC3F7; /* Blau in Sidebar */
    }
    
    /* Plots (Matplotlib) */
    .stPlotlyChart, figure {
        background-color: #1E1E1E;
    }
    
    /* Überschriften und Caption */
    h1, h2, h3, h4, h5, h6 {
        color: #81D4FA; /* Helleres Blau für Überschriften (metallic-effect) */
    }
    
    .stCaption {
        color: #4FC3F7;
    }
    
    /* Selectbox und Text-Inputs */
    .stSelectbox > div > div, .stTextInput > div > div > input {
        color: #4FC3F7; /* Blau für Inputs */
    }
</style>
""", unsafe_allow_html=True)

# Streamlit App
st.set_page_config(page_title="OptionsAssistent Mega 2026 Pro", layout="wide")
st.title("💥 OptionsAssistent Mega 2026 Pro")
st.markdown("---")
st.caption("*Mega-Edition: Greeks, Multi-Leg, Optimizer, Live-Daten. Keine Finanzberatung – DYOR! Stand: Feb 2026*")

# API-Key Input
api_key = st.sidebar.text_input("Polygon API-Key (für Live-Daten)", type="password", help="Gib hier deinen Polygon API-Schlüssel ein, um Live-Daten für Optionspreise zu holen. Erhalte einen Key auf polygon.io.")

# Wizard-Schritte (mit Session State)
if 'step' not in st.session_state:
    st.session_state.step = 1

if st.session_state.step == 1:
    st.header("Schritt 1: Basis & Live-Daten")
    ticker = st.text_input("Underlying Ticker (z.B. AAPL)", "AAPL", help="Der Ticker-Symbol des zugrunde liegenden Assets, z.B. AAPL für Apple-Aktien. Mehr Infos: https://www.investopedia.com/terms/t/tickersymbol.asp")
    S = st.number_input("Aktueller Kurs (S) €", value=150.0, help="Der aktuelle Marktpreis des Underlying Assets (z.B. Aktie). Das ist der Spot-Preis.")
    expiry_str = st.text_input("Expiry-Datum (YYYYMMDD, optional)", help="Das Verfallsdatum der Option im Format YYYYMMDD. Für Live-Daten.")
    strike_input = st.number_input("Strike für Live-Query (optional)", value=150.0, help="Der Strike-Preis für die spezifische Option in der Live-Query.")
    opt_type_live = st.selectbox("Option-Typ für Live (C/P)", ["C", "P"], help="C für Call, P für Put.")
    if api_key and st.button("Live-Preis holen"):
        live_price = fetch_live_options(ticker, api_key, int(strike_input * 1000), expiry_str, opt_type_live)
        if live_price > 0:
            st.success(f"Live-Optionspreis: {live_price:.2f}")
        else:
            st.warning("Kein Live-Preis gefunden – nutze manuellen Input.")
    T_days = st.slider("Zeit bis Verfall (T) Tage", 1, 1825, 182, help="Die verbleibende Zeit bis zum Verfall der Option in Tagen (z.B. 182 für ca. 6 Monate).")
    T = T_days / 365.0  # Intern in Jahren umwandeln
    r = st.slider("Risikofreier Zins (r) %", 0.0, 10.0, 3.0, help="Der risikofreie Zinssatz in Prozent (z.B. EZB-Leitzins oder LIBOR).") / 100
    sigma = st.slider("Volatilität (sigma) %", 5.0, 100.0, 25.0, help="Die implizite Volatilität des Underlyings in Prozent (Maß für Preisschwankungen).") / 100
    if st.button("Weiter"):
        st.session_state.ticker, st.session_state.S, st.session_state.T, st.session_state.r, st.session_state.sigma = ticker, S, T, r, sigma
        st.session_state.T_days = T_days  # Speichere Tage für Ausgabe
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.header("Schritt 2: Forecast & Risk")
    forecast_price = st.selectbox("Preisverlauf?", ["Up", "Down", "Neutral"], help="Deine Erwartung für den Preis des Underlyings: Up (steigt), Down (fällt) oder Neutral (stabil).")
    forecast_vol = st.selectbox("Volatilität?", ["Increase", "Decrease", "Stable"], help="Deine Erwartung für die Volatilität: Increase (steigt), Decrease (fällt) oder Stable (gleichbleibend).")
    risk_tolerance = st.selectbox("Risiko-Toleranz?", ["Low", "Medium", "High"], help="Dein Risikoprofil: Low (niedrig), Medium (mittel) oder High (hoch). Beeinflusst die Strategievorschläge.")
    target_return = st.number_input("Ziel-Return %", value=20.0, help="Dein angestrebter Renditeprozentsatz für den Trade.")
    if st.button("Strategien optimieren"):
        st.session_state.strategies = suggest_strategies(forecast_price, forecast_vol, risk_tolerance)
        st.session_state.forecast_price, st.session_state.forecast_vol, st.session_state.risk_tolerance, st.session_state.target_return = forecast_price, forecast_vol, risk_tolerance, target_return
        st.session_state.step = 3
        st.rerun()

elif st.session_state.step == 3:
    st.header("Schritt 3: Strategie-Builder & Analyse")
    strategies = st.session_state.strategies
    selected = st.selectbox("Strategie wählen", [s['name'] for s in strategies], help="Wähle eine der vorgeschlagenen Optionsstrategien aus.")
    legs = []
    num_legs = min(next(s['legs'] for s in strategies if s['name'] == selected), 6)  # Bis 6
    for i in range(num_legs):
        with st.expander(f"Leg {i+1} (erweiterbar für Mobile)"):
            col = st.columns(4)
            opt_type = col[0].selectbox(f"Typ", ["call", "put"], help="Der Typ der Option: Call (Kaufrecht) oder Put (Verkaufsrecht).")
            position = col[1].selectbox(f"Position", ["long", "short"], help="Deine Position: Long (kaufen) oder Short (verkaufen).")
            K = col[2].slider(f"Strike €", 50.0, 300.0, st.session_state.S, help="Der Ausübungspreis (Strike) der Option.")
            legs.append((st.session_state.S, K, st.session_state.T, st.session_state.r, st.session_state.sigma, opt_type, position))

    # Berechnungen
    stock_prices = np.linspace(st.session_state.S * 0.5, st.session_state.S * 1.5, 200)
    payoffs_current = multi_leg_payoff(legs, stock_prices)
    payoffs_expiry = multi_leg_payoff(legs, stock_prices, T_for_calc=0)  # Fix: T=0 für Expiry
    total_price = sum(black_scholes_greeks(*leg[:-1])[0] * (1 if leg[-1] == 'long' else -1) for leg in legs)
    max_profit = np.max(payoffs_current)
    max_loss = np.min(payoffs_current)
    prob_profit = monte_carlo_prob_profit(legs, st.session_state.S, st.session_state.target_return)  # Fix: Monte-Carlo

    st.metric("Gesamt-Prämie", f"€ {total_price:.2f}")
    st.metric("Max Profit", f"€ {max_profit:.2f}")
    st.metric("Max Loss", f"€ {max_loss:.2f}")
    st.metric("Profit-Wahrscheinlichkeit (Monte-Carlo)", f"{prob_profit:.1f}%")

    # Greeks-Tabelle
    greeks_data = []
    for i, leg in enumerate(legs):
        _, delta, gamma, theta, vega, rho = black_scholes_greeks(*leg[:-1])
        greeks_data.append({"Leg": i+1, "Delta": delta, "Gamma": gamma, "Theta (pro Tag)": theta / 365, "Vega": vega, "Rho": rho})
    st.subheader("Greeks")
    st.dataframe(pd.DataFrame(greeks_data))

    # Payoff-Plot
    st.subheader("Payoff-Kurve (Current vs. Expiry)")
    fig, ax = plt.subplots()
    ax.plot(stock_prices, payoffs_current, label="Current P/L", color='blue')
    ax.plot(stock_prices, payoffs_expiry, label="Expiry P/L", linestyle='--', color='green')
    ax.axhline(0, color='black', linestyle='--')
    ax.axvline(st.session_state.S, color='red', linestyle='--', label='Aktueller Kurs')
    ax.legend()
    ax.grid(True)
    ax.set_facecolor('#1E1E1E')  # Dunkler Plot-Hintergrund
    fig.patch.set_facecolor('#121212')
    st.pyplot(fig)

    # Sensitivität
    st.subheader("Sensitivitäts-Analyse")
    scenarios = pd.DataFrame()
    for vol in [st.session_state.sigma * 0.8, st.session_state.sigma, st.session_state.sigma * 1.2]:
        for t_days in [st.session_state.T_days // 2, st.session_state.T_days]:
            t = t_days / 365.0
            temp_legs = [(*l[:2], t, *l[3:]) for l in legs]
            payoff = multi_leg_payoff(temp_legs, np.array([st.session_state.S]))[0]
            scenarios = scenarios._append({"Vol %": vol*100, "T Tage": t_days, "P/L €": payoff}, ignore_index=True)
    st.dataframe(scenarios)

    # Optimizer
    if st.button("Optimieren für Ziel-Return"):
        target_price = st.session_state.S * (1 + st.session_state.target_return / 100)
        best_K, best_payoff = optimize_strike(legs, target_price)
        st.success(f"Bester Strike: {best_K:.2f} € | Erwarteter P/L: {best_payoff:.2f} €")

    # Export
    if st.button("CSV Export"):
        df = pd.DataFrame({"Stock Prices": stock_prices, "Current Payoff": payoffs_current, "Expiry Payoff": payoffs_expiry})
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "payoffs.csv", "text/csv")

    if st.button("Zurück"):
        st.session_state.step = 1
        st.rerun()

# Footer
st.markdown("---")
st.caption("Mega-Built by xAI & dich. Test's, tweak's, und lass uns upgraden – z.B. mit mehr APIs oder Backtesting! 🚀")
