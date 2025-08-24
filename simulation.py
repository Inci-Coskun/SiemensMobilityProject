import streamlit as st
import random
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import io

# --- CSV Data (Embedded Piezo dataset) ---
csv_data = """voltage(v),current(uA),weight(kgs),step_location,Power(mW)
7.52,50.89,53.00,Center,0.38
16.10,51.45,59.00,Edge,0.83
21.70,54.90,63.00,Center,1.19
6.05,52.86,54.00,Edge,0.32
33.70,52.06,76.00,Center,1.75
26.39,54.59,69.00,Edge,1.44
29.80,49.55,73.00,Center,1.48
19.70,53.35,63.00,Center,1.05
26.50,48.64,72.00,Center,1.29
30.80,51.56,75.00,Corner,1.59
13.90,47.06,61.00,Corner,0.65
25.50,52.13,71.00,Edge,1.33
18.60,54.66,64.00,Edge,1.02
6.07,47.51,54.00,Edge,0.29
21.80,51.65,65.00,Center,1.13
0.00,0.00,53.00,Center,0.00
28.40,53.39,73.00,Center,1.52
27.90,50.93,69.00,Center,1.42
25.90,46.24,71.00,Edge,1.20
5.05,49.54,51.00,Edge,0.25
2.18,50.76,50.00,Corner,0.11
26.30,54.93,71.00,Edge,1.44
37.10,50.61,80.00,Center,1.88
4.67,54.42,54.00,Corner,0.25
19.30,49.85,66.00,Edge,0.96
24.00,51.48,69.00,Corner,1.24
11.50,49.51,58.00,Corner,0.57
26.50,47.65,68.00,Center,1.26
27.90,52.69,70.00,Center,1.47
4.21,48.50,51.00,Corner,0.20
4.32,46.61,52.00,Corner,0.20
5.21,50.06,54.00,Edge,0.26
26.80,45.08,70.00,Center,1.21
0.00,0.00,77.00,Edge,0.00
24.70,49.88,69.00,Edge,1.23
29.00,54.11,70.00,Center,1.57
26.80,52.57,71.00,Center,1.41
7.40,45.27,56.00,Center,0.33
18.20,49.16,64.00,Edge,0.89
24.10,49.52,73.00,Corner,1.19
12.80,51.35,61.00,Corner,0.66
21.20,51.16,65.00,Center,1.08
13.70,48.88,61.00,Corner,0.67
19.60,45.50,64.00,Center,0.89
6.20,50.74,53.00,Edge,0.31
23.70,50.30,65.00,Corner,1.19
4.96,53.33,55.00,Corner,0.26
16.10,53.59,60.00,Edge,0.86
32.00,47.20,74.00,Corner,1.51
26.10,52.31,68.00,Edge,1.37
23.40,52.86,69.00,Corner,1.24
0.00,0.00,80.00,Edge,0.00
25.70,50.01,70.00,Edge,1.29
27.60,47.90,73.00,Center,1.32
13.90,48.72,62.00,Corner,0.68
14.50,52.83,63.00,Corner,0.77
53.70,54.88,89.00,Center,2.95
16.40,46.21,65.00,Edge,0.76
11.40,54.02,58.00,Corner,0.62
7.60,47.46,54.00,Center,0.36
5.00,51.17,52.00,Corner,0.26
24.30,47.05,64.00,Center,1.14
0.00,0.00,73.00,Center,0.00
22.00,48.95,66.00,Corner,1.08
24.10,53.29,71.00,Corner,1.28
19.40,49.48,65.00,Edge,0.96
33.70,53.06,77.00,Corner,1.79
20.50,48.87,63.00,Corner,1.00
19.80,49.06,66.00,Center,0.97
17.00,46.47,64.00,Edge,0.79
14.40,45.91,62.00,Corner,0.66
34.60,48.58,79.00,Center,1.68
26.40,52.11,70.00,Center,1.38
23.90,47.20,67.00,Center,1.13
35.10,46.79,81.00,Center,1.64
33.80,50.08,76.00,Edge,1.69
23.90,50.06,72.00,Corner,1.20
19.60,51.27,64.00,Center,1.00
5.02,46.26,53.00,Corner,0.23
0.00,0.00,66.00,Center,0.00
4.59,52.01,57.00,Corner,0.24
6.54,54.76,54.00,Edge,0.36
25.30,49.01,72.00,Edge,1.24
18.20,52.56,58.00,Edge,0.96
15.50,49.67,63.00,Corner,0.77
9.50,49.55,56.00,Center,0.47
26.40,52.58,69.00,Center,1.39
35.20,45.09,78.00,Center,1.59
22.00,54.36,66.00,Corner,1.20
43.40,49.92,86.00,Corner,2.17
19.40,45.60,66.00,Center,0.88
16.20,47.53,64.00,Edge,0.77
23.40,53.44,68.00,Corner,1.25
6.30,52.13,55.00,Edge,0.33
24.30,52.69,64.00,Edge,1.28
16.40,49.12,62.00,Edge,0.81
7.77,52.85,55.00,Center,0.41
51.20,50.21,87.00,Center,2.57
19.60,49.06,61.00,Center,0.96
29.40,46.68,72.00,Center,1.37
19.40,47.74,65.00,Center,0.93
9.90,47.17,57.00,Center,0.47
25.20,49.57,70.00,Edge,1.25"""

# --- System Energy Consumption Dataset (Synthetic) ---
system_consumption_data = """time,day_type,footfall,ambient_temp,cctv_power,panels_power,ventilation_power,ticket_booth_power
06:00,Weekday,120,18,25.2,13.1,38.5,8.3
06:30,Weekday,150,19,26.8,14.5,42.1,10.6
07:00,Weekday,380,20,32.1,21.2,55.8,18.4
07:30,Weekday,450,21,34.3,24.8,65.2,25.7
08:00,Weekday,420,22,33.7,23.9,62.1,22.8
08:30,Weekday,200,23,27.9,16.1,44.7,12.2
09:00,Weekday,180,24,27.2,15.3,41.9,11.8
10:00,Weekday,160,25,26.5,14.7,40.2,10.4
11:00,Weekday,140,26,25.8,13.8,38.8,9.1
12:00,Weekday,350,27,31.6,20.5,54.3,17.2
12:30,Weekday,410,28,33.2,22.7,60.8,21.5
13:00,Weekday,380,28,32.4,21.8,56.7,19.1
14:00,Weekday,220,27,28.3,16.9,46.4,13.8
15:00,Weekday,190,26,27.6,15.7,43.1,12.5
16:00,Weekday,160,25,26.7,14.9,40.6,10.8
17:00,Weekday,480,24,35.1,26.2,68.7,28.9
17:30,Weekday,510,23,36.4,27.8,72.3,31.2
18:00,Weekday,460,22,34.8,25.5,67.1,27.1
19:00,Weekday,280,21,29.7,18.3,49.6,15.4
20:00,Weekday,240,20,28.9,17.1,47.2,14.1
21:00,Weekday,320,19,30.8,19.7,52.4,16.8
21:30,Weekday,180,18,27.4,15.1,42.5,12.2
22:00,Weekday,80,17,24.1,11.8,35.3,7.9
23:00,Weekday,60,16,23.2,10.5,33.1,6.2
00:00,Weekday,20,15,22.1,8.9,30.2,4.8
06:00,Weekend,80,18,23.8,11.2,34.1,6.7
07:00,Weekend,120,20,25.3,13.4,37.9,9.1
08:00,Weekend,200,22,27.8,15.9,43.6,12.8
10:00,Weekend,280,25,29.4,18.1,48.9,15.9
12:00,Weekend,350,28,31.8,20.7,54.7,18.6
14:00,Weekend,320,30,30.9,19.8,52.8,17.1
16:00,Weekend,290,29,29.8,18.6,50.1,16.3
18:00,Weekend,380,27,32.3,21.6,56.4,19.8
20:00,Weekend,240,24,28.7,17.0,47.0,14.9
22:00,Weekend,120,21,25.1,13.2,37.6,9.8
00:00,Weekend,30,18,22.5,9.3,31.1,5.5"""

# --- CSS Styling ---
def local_css():
    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .energy-source {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin: 10px 0;
        }
        .system-status {
            border-left: 4px solid #28a745;
            padding: 10px;
            margin: 5px 0;
            background-color: #f8f9fa;
        }
        .warning-status {
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 5px 0;
            background-color: #fff3cd;
        }
        .error-status {
            border-left: 4px solid #dc3545;
            padding: 10px;
            margin: 5px 0;
            background-color: #f8d7da;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Data Models ---
class EnergySourceData:
    def __init__(self, instantaneous_power, stored_energy, harvest_rate, health_status):
        self.instantaneous_power = instantaneous_power
        self.stored_energy = stored_energy
        self.harvest_rate = harvest_rate
        self.health_status = health_status

class SystemLoadData:
    def __init__(self, system_id, base_power, predicted_power, min_power,
                 max_power, priority, status, efficiency):
        self.system_id = system_id
        self.base_power = base_power
        self.predicted_power = predicted_power
        self.min_power = min_power
        self.max_power = max_power
        self.priority = priority
        self.status = status
        self.efficiency = efficiency

class ContextFactors:
    def __init__(self, footfall, time_of_day, day_type, event_triggers, ambient_temp):
        self.footfall = footfall
        self.time_of_day = time_of_day
        self.day_type = day_type
        self.event_triggers = event_triggers
        self.ambient_temp = ambient_temp

# --- AI Models ---
class PiezoEnergyPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
        self.feature_names = ['voltage', 'current', 'weight', 'location_center', 'location_edge']
        
    def train_model(self):
        # Load piezo dataset
        df = pd.read_csv(io.StringIO(csv_data))
        
        # Feature engineering with proper column names
        X = pd.DataFrame()
        X['voltage'] = df['voltage(v)']
        X['current'] = df['current(uA)']
        X['weight'] = df['weight(kgs)']
        X['location_center'] = (df['step_location'] == 'Center').astype(int)
        X['location_edge'] = (df['step_location'] == 'Edge').astype(int)
        
        y = df['Power(mW)']
        
        # Ensure feature names are consistent
        X.columns = self.feature_names
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate model performance
        y_pred = self.model.predict(X)
        self.mae = mean_absolute_error(y, y_pred)
        self.r2 = r2_score(y, y_pred)
        
    def predict_power_from_footfall(self, footfall, avg_weight=65):
        if not self.is_trained:
            self.train_model()
            
        # Enhanced footfall to energy conversion with better scaling
        base_voltage = min(footfall * 0.15 + random.uniform(-1, 3), 55)
        current = 50 + random.uniform(-3, 8)
        
        # Weighted random location selection
        location_weights = {'Center': 0.5, 'Edge': 0.3, 'Corner': 0.2}
        location = random.choices(list(location_weights.keys()),
                                 weights=list(location_weights.values()))[0]
        
        location_center = 1 if location == 'Center' else 0
        location_edge = 1 if location == 'Edge' else 0
        
        # Create feature array with proper naming
        features = pd.DataFrame([[base_voltage, current, avg_weight, location_center, location_edge]],
                                columns=self.feature_names)
        
        predicted_power_mw = max(0, self.model.predict(features)[0])
        
        # Enhanced scaling: convert mW to W and apply footfall multiplier
        # Multiply by number of piezo tiles (assume 10-20 tiles in station)
        num_tiles = 15
        footfall_efficiency = min(1.0, footfall / 200.0)
        
        total_power_w = (predicted_power_mw / 1000) * num_tiles * footfall_efficiency
        
        return total_power_w

class SystemConsumptionPredictor:
    def __init__(self):
        self.models = {}
        self.is_trained = False
        self.feature_names = ['hour', 'footfall', 'ambient_temp', 'is_weekend']
        
    def train_models(self):
        # Load system consumption dataset
        df = pd.read_csv(io.StringIO(system_consumption_data))
        
        # Feature engineering with proper naming
        df['hour'] = pd.to_datetime(df['time']).dt.hour
        df['is_weekend'] = (df['day_type'] == 'Weekend').astype(int)
        
        X = df[self.feature_names].copy()
        
        systems = ['cctv_power', 'panels_power', 'ventilation_power', 'ticket_booth_power']
        
        for system in systems:
            y = df[system]
            
            # Use Random Forest for better performance
            self.models[system] = RandomForestRegressor(n_estimators=50, random_state=42)
            self.models[system].fit(X, y)
            self.is_trained = True
            
    def predict_consumption(self, context_data):
        if not self.is_trained:
            self.train_models()
        
        hour = context_data.time_of_day.hour
        footfall = context_data.footfall
        ambient_temp = context_data.ambient_temp
        is_weekend = 1 if context_data.day_type == 'Weekend' else 0
        
        # Create feature DataFrame with proper naming
        features = pd.DataFrame([[hour, footfall, ambient_temp, is_weekend]],
                                columns=self.feature_names)
        
        predictions = {}
        for system_name, model in self.models.items():
            pred = model.predict(features)[0]
            predictions[system_name] = max(0, pred)  # Prevent negative values
        
        return predictions

class SmartAllocationAgent:
    def __init__(self):
        self.priority_weights = {"Critical": 1.0, "High": 0.8, "Medium": 0.6, "Low": 0.4}
    
    def allocate_power(self, available_power, systems, context_data):
        allocated_power = {}
        remaining_power = available_power
        
        # Emergency check with enhanced multiplier
        emergency_factor = 1.3 if "emergency" in context_data.event_triggers.lower() else 1.0
        rush_hour_factor = 1.1 if context_data.footfall > 400 else 1.0
        
        # Sort by priority
        sorted_systems = sorted(systems, key=lambda x: self.priority_weights.get(x.priority, 0), reverse=True)
        
        allocation_log = []
        for system in sorted_systems:
            required_power = system.predicted_power * emergency_factor * rush_hour_factor
            
            if remaining_power >= required_power:
                allocated_power[system.system_id] = required_power
                remaining_power -= required_power
                status = "✅ Full Power"
                efficiency = system.efficiency
            elif remaining_power >= system.min_power:
                allocated_power[system.system_id] = remaining_power
                efficiency = system.efficiency * (remaining_power / required_power)
                remaining_power = 0
                status = "⚠️ Partial Power"
            else:
                allocated_power[system.system_id] = 0
                efficiency = 0
                status = "❌ Offline"
            
            allocation_log.append({
                "System": system.system_id,
                "Priority": system.priority,
                "Required": f"{required_power:.1f}W",
                "Allocated": f"{allocated_power[system.system_id]:.1f}W",
                "Efficiency": f"{efficiency*100:.0f}%",
                "Status": status
            })
            
        return allocated_power, remaining_power, allocation_log

# --- Station Schedule ---
STATION_SCHEDULE = [
    {"time": "06:00", "event_type": "Station Opening", "footfall_range": (100, 200)},
    {"time": "07:15", "event_type": "Train 101 Arrival", "footfall_range": (300, 450)},
    {"time": "08:30", "event_type": "Morning Peak", "footfall_range": (400, 600)},
    {"time": "12:00", "event_type": "Lunchtime Rush", "footfall_range": (350, 500)},
    {"time": "17:00", "event_type": "Evening Peak", "footfall_range": (500, 700)},
    {"time": "19:45", "event_type": "Train 205 Departure", "footfall_range": (250, 350)},
    {"time": "22:00", "event_type": "Night Closure", "footfall_range": (50, 150)},
]

# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide")
    local_css()

    if "simulation_started" not in st.session_state:
        st.session_state.simulation_started = False
    if "battery_energy" not in st.session_state:
        st.session_state.battery_energy = 50.0  # Initial energy in kWh
    if "footfall_history" not in st.session_state:
        st.session_state.footfall_history = []
    if "energy_history" not in st.session_state:
        st.session_state.energy_history = []
    if "consumption_history" not in st.session_state:
        st.session_state.consumption_history = []
    if "balance_history" not in st.session_state:
        st.session_state.balance_history = []
    if "last_update_time" not in st.session_state:
        st.session_state.last_update_time = datetime.now()
    if "models_trained" not in st.session_state:
        st.session_state.models_trained = False

    # Initialize AI models and agents
    piezo_predictor = PiezoEnergyPredictor()
    consumption_predictor = SystemConsumptionPredictor()
    smart_agent = SmartAllocationAgent()

    st.title("Train Station Smart Micro-Grid Simulation")
    st.markdown("---")
    
    # Navigation
    with st.sidebar:
        st.subheader("🧭 Navigation")
        page = st.radio("Select Page:", [
            "🏠 Home",
            "⚡ Simulation",
            "📅 Station Schedule",
            "🤖 AI Models"
        ])
        
        st.markdown("---")
        st.subheader("ℹ️ System Information")
        st.write("**Piezo Sensors:** Active")
        st.write("**AI Models:** Trained")
        st.write("**Connection:** Online")
        
        # System health indicators
        if "models_trained" in st.session_state:
            st.success("✅ AI Models Ready")
        else:
            st.warning("⏳ Models Loading...")
            
        if "battery_energy" in st.session_state:
            battery_health = st.session_state.battery_energy / 100.0
            if battery_health > 0.7:
                st.success(f"🔋 Battery: {battery_health*100:.0f}%")
            elif battery_health > 0.3:
                st.warning(f"🔋 Battery: {battery_health*100:.0f}%")
            else:
                st.error(f"🔋 Battery: {battery_health*100:.0f}%")

    # Page routing
    if page == "🏠 Home":
        home_page()
    elif page == "⚡ Simulation":
        simulation_page(piezo_predictor, consumption_predictor, smart_agent)
    elif page == "📅 Station Schedule":
        schedule_page()
    elif page == "🤖 AI Models":
        ai_models_page(piezo_predictor, consumption_predictor)

def home_page():
    st.header("Welcome to the Smart Micro-Grid Simulation")
    st.write(
        "This application simulates a train station's micro-grid, managing energy "
        "from footfall-based piezoelectric sensors and a central battery. "
        "It uses AI models to predict energy generation and system consumption, "
        "and a smart agent to allocate power efficiently based on real-time needs and priorities."
    )
    st.markdown("---")
    st.subheader("How It Works:")
    st.write(
        "1. **Energy Generation**: Piezoelectric sensors under the station floor convert the kinetic energy of passenger footfall into electrical energy."
    )
    st.write(
        "2. **AI Prediction**: The `PiezoEnergyPredictor` model forecasts the energy generated from footfall data."
    )
    st.write(
        "3. **Consumption Prediction**: The `SystemConsumptionPredictor` model anticipates the power needs of different station systems (e.g., CCTV, ventilation)."
    )
    st.write(
        "4. **Smart Allocation**: The `SmartAllocationAgent` prioritizes and distributes the generated energy and battery power to critical systems first, ensuring continuous operation."
    )
    st.write(
        "5. **Real-time Monitoring**: The simulation page visualizes real-time data on energy generation, consumption, and battery status."
    )

    st.markdown("---")
    st.subheader("Simulation Overview:")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://i.imgur.com/example-image.png", caption="Smart Station Concept")
    with col2:
        st.info("Navigate to the **⚡ Simulation** page to start the live simulation and see the micro-grid in action.")
        st.markdown(
            """
            - **See real-time energy flow**
            - **Monitor battery health**
            - **View power allocation decisions**
            - **Explore the impact of footfall**
            """
        )

def simulation_page(piezo_predictor, consumption_predictor, smart_agent):
    st.header("⚡ Live Simulation")
    st.write(
        "Control the simulation and watch the smart micro-grid in action. The simulation runs in 30-minute intervals."
    )
    
    st.markdown("---")

    # Simulation controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        start_button = st.button("▶️ Start Simulation")
    with col2:
        stop_button = st.button("⏹️ Stop Simulation")
    with col3:
        st.info("The simulation advances time in 30-minute steps.")
        
    if start_button:
        st.session_state.simulation_started = True
    if stop_button:
        st.session_state.simulation_started = False
        st.info("Simulation stopped.")
        
    if st.session_state.simulation_started:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train models if not already trained
        if not st.session_state.models_trained:
            status_text.text("Training AI models...")
            piezo_predictor.train_model()
            consumption_predictor.train_models()
            st.session_state.models_trained = True
            st.experimental_rerun()  # Rerun to update sidebar status
            
        # Simulation loop
        current_time_str = st.session_state.last_update_time.strftime("%I:%M %p")
        status_text.text(f"Simulation running... Current Time: {current_time_str}")
        
        # Get current context
        current_footfall, event_triggers = get_footfall_for_time(st.session_state.last_update_time)
        context = ContextFactors(
            footfall=current_footfall,
            time_of_day=st.session_state.last_update_time,
            day_type="Weekday" if st.session_state.last_update_time.weekday() < 5 else "Weekend",
            event_triggers=event_triggers,
            ambient_temp=generate_ambient_temp(st.session_state.last_update_time)
        )
        
        # Predict energy generation and consumption
        piezo_generation_w = piezo_predictor.predict_power_from_footfall(context.footfall, 65)
        piezo_generation_kw = piezo_generation_w / 1000.0  # Convert to kW
        
        predicted_consumption = consumption_predictor.predict_consumption(context)
        total_consumption_w = sum(predicted_consumption.values())

        # Define system loads based on predictions and static data
        systems_to_manage = [
            SystemLoadData("CCTV", predicted_consumption['cctv_power'], predicted_consumption['cctv_power'], 20.0, 40.0, "High", "Normal", 0.95),
            SystemLoadData("Lighting Panels", predicted_consumption['panels_power'], predicted_consumption['panels_power'], 10.0, 30.0, "Medium", "Normal", 0.90),
            SystemLoadData("Ventilation", predicted_consumption['ventilation_power'], predicted_consumption['ventilation_power'], 30.0, 80.0, "Critical", "Normal", 0.85),
            SystemLoadData("Ticket Booth", predicted_consumption['ticket_booth_power'], predicted_consumption['ticket_booth_power'], 5.0, 35.0, "High", "Normal", 0.92)
        ]
        
        # Smart power allocation
        allocated_power, remaining_power, allocation_log = smart_agent.allocate_power(
            piezo_generation_w + (st.session_state.battery_energy * 1000 / 0.5),  # Assume battery can provide 2kWh/min
            systems_to_manage,
            context
        )
        
        # Update battery and history
        energy_balance_w = piezo_generation_w - total_consumption_w
        energy_balance_kwh = energy_balance_w * (30 / 60) / 1000  # for 30 min interval
        st.session_state.battery_energy = max(0, min(100, st.session_state.battery_energy + energy_balance_kwh))
        
        st.session_state.footfall_history.append(context.footfall)
        st.session_state.energy_history.append(piezo_generation_w)
        st.session_state.consumption_history.append(total_consumption_w)
        st.session_state.balance_history.append(energy_balance_w)
        
        # Display real-time metrics
        display_metrics(piezo_generation_w, total_consumption_w, st.session_state.battery_energy)
        
        # Display detailed allocation log
        st.markdown("---")
        st.subheader("🔍 Power Allocation Log")
        log_df = pd.DataFrame(allocation_log)
        st.table(log_df)

        # Display charts
        st.markdown("---")
        st.subheader("📈 Energy Metrics History")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.line_chart(pd.DataFrame({
                "Piezo Generation (W)": st.session_state.energy_history,
                "Total Consumption (W)": st.session_state.consumption_history
            }))
        with col_chart2:
            st.bar_chart(pd.DataFrame({
                "Energy Balance (W)": st.session_state.balance_history
            }))
            
        # Update time
        st.session_state.last_update_time += timedelta(minutes=30)
        
        # Rerun to simulate next step
        time.sleep(1)
        st.experimental_rerun()

def get_footfall_for_time(current_time):
    current_time_str = current_time.strftime("%H:%M")
    
    # Check for specific schedule events
    for event in STATION_SCHEDULE:
        event_time_obj = datetime.strptime(event["time"], "%H:%M").time()
        current_time_obj = current_time.time()
        
        if event_time_obj.hour == current_time_obj.hour and abs(event_time_obj.minute - current_time_obj.minute) < 30:
            footfall = random.randint(event["footfall_range"][0], event["footfall_range"][1])
            return footfall, event["event_type"]
            
    # Default footfall
    hour = current_time.hour
    if 6 <= hour < 10:  # Morning rush
        footfall = random.randint(300, 500)
    elif 10 <= hour < 16: # Mid-day
        footfall = random.randint(150, 300)
    elif 16 <= hour < 20: # Evening rush
        footfall = random.randint(400, 600)
    else: # Off-peak
        footfall = random.randint(50, 150)
        
    return footfall, "None"

def generate_ambient_temp(current_time):
    hour = current_time.hour
    if 6 <= hour < 10:
        return random.uniform(18, 24)
    elif 10 <= hour < 16:
        return random.uniform(25, 30)
    elif 16 <= hour < 20:
        return random.uniform(22, 28)
    else:
        return random.uniform(15, 20)

def display_metrics(piezo_gen, total_cons, battery_level):
    st.markdown("### 📊 Real-time Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f'<div class="metric-card"><h4>Instantaneous Generation:</h4>'
            f'<h2><span style="color:#667eea;">{piezo_gen:.2f} W</span></h2></div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><h4>Total Consumption:</h4>'
            f'<h2><span style="color:#764ba2;">{total_cons:.2f} W</span></h2></div>',
            unsafe_allow_html=True
        )
    with col3:
        status_color = "green" if battery_level > 70 else ("orange" if battery_level > 30 else "red")
        st.markdown(
            f'<div class="metric-card"><h4>Battery Level:</h4>'
            f'<h2><span style="color:{status_color};">{battery_level:.1f}%</span></h2></div>',
            unsafe_allow_html=True
        )

def schedule_page():
    st.header("📅 Station Schedule & Events")
    st.write("This table shows the key events and associated footfall ranges throughout a typical day at the station, which influence energy generation and consumption predictions.")
    
    schedule_data = pd.DataFrame(STATION_SCHEDULE)
    st.table(schedule_data)

def ai_models_page(piezo_predictor, consumption_predictor):
    st.header("🤖 AI Model Performance")
    st.write("This section provides details on the performance of the AI models used in the simulation.")

    if not st.session_state.models_trained:
        st.warning("Models are not yet trained. Run the simulation page first to train the models.")
        return

    st.markdown("---")
    st.subheader("Piezo Energy Predictor (Linear Regression)")
    st.write("This model predicts energy generation from footfall data.")
    st.write(f"**Mean Absolute Error (MAE):** {piezo_predictor.mae:.4f}")
    st.write(f"**R-squared (R²):** {piezo_predictor.r2:.4f}")
    st.info("A lower MAE and an R² closer to 1.0 indicate a more accurate model.")

    st.markdown("---")
    st.subheader("System Consumption Predictor (Random Forest)")
    st.write("This model predicts the power consumption of various station systems.")
    st.write("The performance metrics for this model are stored internally for each system (CCTV, Lighting, etc.).")
    st.info("The Random Forest model is used here for its ability to capture non-linear relationships and provide accurate consumption forecasts.")

if __name__ == "__main__":
    main()
