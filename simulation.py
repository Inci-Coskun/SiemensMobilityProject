# app.py
# Run with: streamlit run app.py

import streamlit as st
import random
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import io

# ───────────────────────────────────────────────────────────────────────────────
# Page config MUST be the first Streamlit command
st.set_page_config(page_title="AI-Powered Piezoelectric Energy", page_icon="⚡", layout="wide")
# ───────────────────────────────────────────────────────────────────────────────

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

# --- System Energy Consumption Dataset (Synthetic) - Updated for 3 systems ---
system_consumption_data = """time,day_type,footfall,ambient_temp,cctv_power,public_address_power,ticket_booth_power
06:00,Weekday,120,18,8.2,4.1,2.8
06:30,Weekday,150,19,8.8,4.5,3.6
07:00,Weekday,380,20,10.1,6.2,5.4
07:30,Weekday,450,21,11.3,7.8,6.7
08:00,Weekday,420,22,10.7,6.9,5.8
08:30,Weekday,200,23,8.9,5.1,3.2
09:00,Weekday,180,24,8.7,4.3,3.1
10:00,Weekday,160,25,8.5,4.7,2.8
11:00,Weekday,140,26,8.2,3.8,2.6
12:00,Weekday,350,27,10.6,6.5,4.7
12:30,Weekday,410,28,11.2,7.7,6.1
13:00,Weekday,380,28,10.4,6.8,5.1
14:00,Weekday,220,27,8.9,4.9,3.8
15:00,Weekday,190,26,8.6,4.7,3.2
16:00,Weekday,160,25,8.3,4.2,2.8
17:00,Weekday,480,24,11.5,8.2,7.9
17:30,Weekday,510,23,11.8,8.8,8.2
18:00,Weekday,460,22,11.2,7.5,6.9
19:00,Weekday,280,21,9.2,5.3,4.1
20:00,Weekday,240,20,8.9,4.8,3.8
21:00,Weekday,320,19,9.8,5.7,4.2
21:30,Weekday,180,18,8.4,4.1,3.1
22:00,Weekday,80,17,7.8,2.8,2.2
23:00,Weekday,60,16,7.2,2.5,1.8
00:00,Weekday,20,15,6.8,1.9,1.2
06:00,Weekend,80,18,7.8,2.2,1.7
07:00,Weekend,120,20,8.3,3.4,2.8
08:00,Weekend,200,22,8.8,4.9,3.5
10:00,Weekend,280,25,9.4,5.1,4.2
12:00,Weekend,350,28,10.8,6.7,5.1
14:00,Weekend,320,30,9.9,5.8,4.6
16:00,Weekend,290,29,9.5,5.2,4.3
18:00,Weekend,380,27,10.3,6.6,5.8
20:00,Weekend,240,24,8.7,4.0,3.9
22:00,Weekend,120,21,8.1,3.2,2.5
00:00,Weekend,30,18,7.2,2.3,1.8"""

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
        .system-status { border-left: 4px solid #28a745; padding: 10px; margin: 5px 0; background-color: #f8f9fa; }
        .warning-status { border-left: 4px solid #ffc107; padding: 10px; margin: 5px 0; background-color: #fff3cd; }
        .error-status { border-left: 4px solid #dc3545; padding: 10px; margin: 5px 0; background-color: #f8d7da; }
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
    def __init__(self, system_id, base_power, predicted_power, min_power, max_power, priority, status, efficiency):
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
        self.mae = None
        self.r2 = None
        
    def train_model(self):
        df = pd.read_csv(io.StringIO(csv_data))
        X = pd.DataFrame()
        X['voltage'] = df['voltage(v)']
        X['current'] = df['current(uA)']
        X['weight'] = df['weight(kgs)']
        X['location_center'] = (df['step_location'] == 'Center').astype(int)
        X['location_edge'] = (df['step_location'] == 'Edge').astype(int)
        y = df['Power(mW)']
        X.columns = self.feature_names
        self.model.fit(X, y)
        self.is_trained = True
        y_pred = self.model.predict(X)
        self.mae = mean_absolute_error(y, y_pred)
        self.r2 = r2_score(y, y_pred)
        
    def predict_power_from_footfall(self, footfall, avg_weight=65):
        if not self.is_trained:
            self.train_model()

        # scale voltage/current with noise; clamp voltage
        base_voltage = min(max(0, footfall * 0.3 + random.uniform(-2, 5)), 55)
        current = 50 + random.uniform(-5, 10)

        location_weights = {'Center': 0.5, 'Edge': 0.3, 'Corner': 0.2}
        location = random.choices(list(location_weights.keys()),
                                  weights=list(location_weights.values()))[0]
        location_center = 1 if location == 'Center' else 0
        location_edge = 1 if location == 'Edge' else 0

        features = pd.DataFrame([[base_voltage, current, avg_weight, location_center, location_edge]],
                                columns=self.feature_names)

        predicted_power_mw = max(0, float(self.model.predict(features)[0]))

        # Aggregate across tiles with footfall effects
        num_tiles = 25
        footfall_efficiency = min(1.2, footfall / 150.0)
        footfall_boost = max(1.0, footfall / 100.0)

        total_power_w = (predicted_power_mw / 1000) * num_tiles * footfall_efficiency * footfall_boost

        # enforce a tiny baseline proportional to footfall
        return max(total_power_w, footfall * 0.05)

class SystemConsumptionPredictor:
    def __init__(self):
        self.models = {}
        self.is_trained = False
        self.feature_names = ['hour', 'footfall', 'ambient_temp', 'is_weekend']
        
    def train_models(self):
        df = pd.read_csv(io.StringIO(system_consumption_data))
        df['hour'] = pd.to_datetime(df['time']).dt.hour
        df['is_weekend'] = (df['day_type'] == 'Weekend').astype(int)
        X = df[self.feature_names].copy()

        for system in ['cctv_power', 'public_address_power', 'ticket_booth_power']:
            y = df[system]
            self.models[system] = RandomForestRegressor(n_estimators=80, random_state=42)
            self.models[system].fit(X, y)
        self.is_trained = True
        
    def predict_consumption(self, context_data: ContextFactors):
        if not self.is_trained:
            self.train_models()
        hour = context_data.time_of_day.hour
        footfall = context_data.footfall
        ambient_temp = context_data.ambient_temp
        is_weekend = 1 if context_data.day_type == 'Weekend' else 0
        features = pd.DataFrame([[hour, footfall, ambient_temp, is_weekend]], columns=self.feature_names)
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = max(0.0, float(model.predict(features)[0]))
        return predictions

class SmartAllocationAgent:
    def __init__(self):
        self.priority_weights = {"Critical": 1.0, "High": 0.8, "Medium": 0.6, "Low": 0.4}
        
    def allocate_power(self, available_power, systems, context_data: ContextFactors):
        allocated_power = {}
        remaining_power = available_power
        emergency_factor = 1.3 if "emergency" in str(context_data.event_triggers).lower() else 1.0
        rush_hour_factor = 1.1 if context_data.footfall > 400 else 1.0
        sorted_systems = sorted(systems, key=lambda x: self.priority_weights.get(x.priority, 0), reverse=True)
        allocation_log = []

        for system in sorted_systems:
            required_power = system.predicted_power * emergency_factor * rush_hour_factor
            if remaining_power >= required_power:
                alloc = required_power
                status = "✅ Full Power"
                efficiency = system.efficiency
            elif remaining_power >= system.min_power:
                alloc = remaining_power
                status = "⚠️ Partial Power"
                efficiency = system.efficiency * (alloc / max(required_power, 1e-6))
            else:
                alloc = 0.0
                status = "❌ Offline"
                efficiency = 0.0

            allocated_power[system.system_id] = alloc
            remaining_power = max(0.0, remaining_power - alloc)

            allocation_log.append({
                "System": system.system_id,
                "Priority": system.priority,
                "Required": f"{required_power:.1f} W",
                "Allocated": f"{alloc:.1f} W",
                "Efficiency": f"{efficiency*100:.0f}%",
                "Status": status
            })
        return allocated_power, remaining_power, allocation_log

# --- Station Schedule ---
STATION_SCHEDULE = [
    {"time": "06:00", "event_type": "Station Opening", "footfall_range": (100, 200)},
    {"time": "07:15", "event_type": "Train 101 Arrival", "footfall_range": (400, 600)},
    {"time": "07:20", "event_type": "Train 101 Departure", "footfall_range": (80, 180)},
    {"time": "08:00", "event_type": "Train 102 Arrival", "footfall_range": (420, 650)},
    {"time": "08:05", "event_type": "Train 102 Departure", "footfall_range": (100, 200)},
    {"time": "10:30", "event_type": "Maintenance", "footfall_range": (50, 100)},
    {"time": "12:30", "event_type": "Train 201 Arrival", "footfall_range": (380, 580)},
    {"time": "12:35", "event_type": "Train 201 Departure", "footfall_range": (80, 180)},
    {"time": "14:00", "event_type": "Cleaning", "footfall_range": (60, 120)},
    {"time": "16:45", "event_type": "Train 301 Arrival", "footfall_range": (450, 700)},
    {"time": "16:50", "event_type": "Train 301 Departure", "footfall_range": (120, 220)},
    {"time": "17:30", "event_type": "Train 302 Arrival", "footfall_range": (480, 720)},
    {"time": "17:35", "event_type": "Train 302 Departure", "footfall_range": (100, 200)},
    {"time": "21:00", "event_type": "Train 401 Arrival", "footfall_range": (300, 500)},
    {"time": "21:05", "event_type": "Train 401 Departure", "footfall_range": (80, 180)},
    {"time": "00:00", "event_type": "Station Closure", "footfall_range": (0, 20)},
]

# --- Helpers ---
def get_context_data(sim_time: datetime) -> ContextFactors:
    day_type = "Weekday" if sim_time.weekday() < 5 else "Weekend"
    event = "Normal Operation"
    footfall_min, footfall_max = 80, 250
    for scheduled_event in STATION_SCHEDULE:
        e_hour, e_min = map(int, scheduled_event['time'].split(':'))
        e_dt = sim_time.replace(hour=e_hour, minute=e_min, second=0, microsecond=0)
        if abs((sim_time - e_dt).total_seconds()) < 300:
            event = scheduled_event['event_type']
            footfall_min, footfall_max = scheduled_event['footfall_range']
            break
    footfall = random.randint(footfall_min, footfall_max)
    ambient_temp = 18 + random.uniform(-2, 8)
    return ContextFactors(footfall, sim_time, day_type, event, ambient_temp)

def generate_system_loads(context_data: ContextFactors, consumption_predictor: SystemConsumptionPredictor):
    preds = consumption_predictor.predict_consumption(context_data)
    systems = [
        SystemLoadData(
            system_id="CCTV Security",
            base_power=preds['cctv_power'] * 0.7,
            predicted_power=preds['cctv_power'] * 0.7 * (1 + random.uniform(-0.1, 0.1)),
            min_power=preds['cctv_power'] * 0.5,
            max_power=preds['cctv_power'] * 0.9,
            priority="Critical",
            status="Active",
            efficiency=0.95
        ),
        SystemLoadData(
            system_id="Ticket Booth",
            base_power=preds['ticket_booth_power'] * 0.6,
            predicted_power=preds['ticket_booth_power'] * 0.6 * (1 + random.uniform(-0.1, 0.1)),
            min_power=preds['ticket_booth_power'] * 0.4,
            max_power=preds['ticket_booth_power'] * 0.8,
            priority="High",
            status="Active",
            efficiency=0.92
        ),
        SystemLoadData(
            system_id="Public Address",
            base_power=preds['public_address_power'] * 0.5,
            predicted_power=preds['public_address_power'] * 0.5 * (1 + random.uniform(-0.2, 0.2)),
            min_power=preds['public_address_power'] * 0.2,
            max_power=preds['public_address_power'] * 0.8,
            priority="Low",
            status="Active",
            efficiency=0.85
        ),
    ]
    return systems

def get_next_event(sim_time: datetime):
    for event in STATION_SCHEDULE:
        e_hour, e_min = map(int, event['time'].split(':'))
        e_dt = sim_time.replace(hour=e_hour, minute=e_min, second=0, microsecond=0)
        if e_dt > sim_time:
            return f"{event['event_type']} - {event['time']}"
    return "No more events today"

# --- Controller (render + battery update) ---
class AIController:
    def __init__(self):
        pass

    def run_control_loop(
        self,
        predicted_energy: float,
        actual_power_used: float,
        total_available: float,
        allocation_log: list,
        context_data: ContextFactors,
        system_loads: list
    ):
        # Update battery based on actual usage vs generated
        if actual_power_used > predicted_energy:
            deficit = actual_power_used - predicted_energy
            discharge = min(deficit, st.session_state.battery_energy)
            st.session_state.battery_energy -= discharge
        elif predicted_energy > actual_power_used:
            surplus = predicted_energy - actual_power_used
            st.session_state.battery_energy = min(st.session_state.battery_energy + surplus * 0.85, 100.0)

        st.markdown("---")
        st.subheader("⚡ Energy Status")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Generated", f"{predicted_energy:.2f} W")
        with c2: st.metric("Consumption", f"{actual_power_used:.2f} W")
        with c3: st.metric("Battery", f"{st.session_state.battery_energy:.0f}%")
        with c4: st.metric("Available", f"{total_available:.2f} W")

        left, right = st.columns(2)
        with left:
            st.subheader("📊 System Power Allocation")
            st.dataframe(pd.DataFrame(allocation_log), use_container_width=True)
        with right:
            st.subheader("🌡️ Environmental Factors")
            st.metric("Pedestrian Traffic", f"{context_data.footfall} people")
            st.metric("Ambient Temperature", f"{context_data.ambient_temp:.1f} °C")
            st.metric("Current Event", context_data.event_triggers)
            st.metric("Day Type", context_data.day_type)

        with st.expander("🔍 Detailed AI Analysis Log", expanded=False):
            if 'piezo_mae' in st.session_state and 'piezo_r2' in st.session_state:
                st.info("🤖 Piezoelectric Model Performance")
                st.write(f"MAE = {st.session_state.piezo_mae:.3f} mW, R² = {st.session_state.piezo_r2:.3f}")
            st.write(f"Footfall: {context_data.footfall} → Predicted Energy: {predicted_energy:.2f} W")
            st.write(f"Footfall Efficiency: {min(1.0, context_data.footfall/200.0)*100:.0f}%")
            st.info("🏭 Consumption Predictions")
            for s in system_loads:
                st.write(f"- {s.system_id}: {s.predicted_power:.1f} W (Priority: {s.priority})")
            st.info("🧠 Allocation Strategy")
            st.write("Priority-based optimization with rush-hour & emergency multipliers.")

        # Log history
        st.session_state.historical_data.append({
            "timestamp": st.session_state.simulated_time.strftime('%Y-%m-%d %H:%M:%S'),
            "footfall": context_data.footfall,
            "event": context_data.event_triggers,
            "energy_generated": predicted_energy,
            "total_available": total_available,
            "total_allocated": actual_power_used,
            "remaining_power": max(0.0, total_available - actual_power_used),
            "ambient_temp": context_data.ambient_temp,
            "battery_level": st.session_state.battery_energy
        })

# ───────────────────────────────────────────────────────────────────────────────
# Pages
def home_page():
    st.header("🔋 What is Piezoelectric Energy?")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(
            "https://earimediaprodweb.azurewebsites.net/Api/v1/Multimedia/51e01af1-0782-49a1-9415-0d44220dd5d2/Rendition/low-res/Content/Public",
            caption="Piezoelectric energy concept",
        )
    with c2:
        st.write("Piezoelectric materials turn footsteps into electricity. "
                 "This app simulates a smart micro-grid that collects this energy and powers station systems.")
    st.markdown("---")
    st.header("📱 About This Application")
    st.write(
        "Real-time simulation of an AI-powered micro-grid. "
        "Shows piezoelectric energy harvesting, ML-based demand forecasting, and intelligent power allocation."
    )
    st.markdown("### 🎯 User Guide")
    st.markdown("- **🏠 Home** — overview")
    st.markdown("- **⚡ Simulation** — real-time controller decisions")
    st.markdown("- **📅 Station Schedule** — events the AI uses")
    st.markdown("- **🤖 AI Models** — model details & metrics")
    st.markdown("### 🔬 Technical Features")
    st.markdown("- Linear Regression (generation) + Random Forest (consumption)")
    st.markdown("- Priority-based allocation (CCTV > Ticket Booth > PA)")
    st.markdown("- Dynamic adaptation to footfall & events")
    st.markdown("- Battery charge/discharge optimization")

def station_schedule_page():
    st.header("📅 Daily Station Schedule")
    st.write("Static schedule that drives realistic event patterns.")
    schedule_df = pd.DataFrame(STATION_SCHEDULE)
    schedule_df.rename(columns={'time': 'Time', 'event_type': 'Event Type', 'footfall_range': 'Pedestrian Traffic Range'}, inplace=True)
    st.dataframe(schedule_df, use_container_width=True)
    st.markdown("---")
    st.subheader("📊 Schedule Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🚂 Peak Hours:**")
        st.write("- 07:15–08:05 Morning rush")
        st.write("- 12:30–12:35 Lunch traffic")
        st.write("- 16:45–17:35 Evening rush")
    with col2:
        st.markdown("**🔧 Maintenance / Low Activity:**")
        st.write("- 10:30 Maintenance")
        st.write("- 14:00 Cleaning")
        st.write("- 00:00–06:00 Closed")
    st.markdown("---")
    st.subheader("📈 Expected Footfall by Hour")
    schedule_data = []
    for e in STATION_SCHEDULE:
        hour = int(e['time'].split(':')[0])
        avg_footfall = sum(e['footfall_range']) / 2
        schedule_data.append({'Hour': hour, 'Expected Footfall': avg_footfall})
    st.bar_chart(pd.DataFrame(schedule_data).set_index('Hour'))

def ai_models_page():
    """AI models page."""
    st.header("🤖 AI Models and Performance")
    
    st.markdown("### 1. 🔋 Piezoelectric Energy Prediction Model")
    st.write("**Model Type:** Linear Regression")
    st.write("**Purpose:** Predict piezoelectric energy generation from pedestrian traffic")
    
    if "piezo_mae" in st.session_state and "piezo_r2" in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Absolute Error (MAE)", f"{st.session_state.piezo_mae:.3f}")
        with col2:
            st.metric("R² Score", f"{st.session_state.piezo_r2:.3f}")
    
    st.markdown("**🔧 Features:**")
    st.write("- Voltage (V), Current (μA), Weight (kg)")
    st.write("- Step Location (Center/Edge/Corner)")
    st.write("- Footfall multiplier effect")
    st.write("- 15 piezoelectric tiles simulation")
    
    st.markdown("**⚡ Energy Conversion Logic:**")
    st.code("""
# Enhanced energy conversion with realistic scaling
base_voltage = footfall * 0.15 + noise
predicted_power_mw = model.predict(features)
total_power_w = (predicted_power_mw / 1000) * num_tiles * efficiency
""")
    
    st.markdown("---")
    
    st.markdown("### 2. 🏭 System Consumption Prediction Model")
    st.write("**Model Type:** Random Forest Regression")
    st.write("**Purpose:** Predict CCTV, public address, and ticket booth consumption")
    
    st.markdown("**🔧 Features:**")
    st.write("- Hour, Pedestrian traffic, Ambient temperature")
    st.write("- Day type (Weekday/Weekend)")
    st.write("- System-specific optimization")
    
    st.markdown("**🎯 Model Benefits:**")
    st.write("- **Non-linear patterns:** Random Forest captures complex relationships")
    st.write("- **Feature importance:** Automatically weighs most relevant factors")
    st.write("- **Robust predictions:** Handles outliers and missing data well")
    
    st.markdown("---")
    
    st.markdown("### 3. 🧠 Smart Power Allocation Agent")
    st.write("**Model Type:** Rule-based + Priority Optimization")
    st.write("**Purpose:** Optimally distribute available power based on system priorities")
    
    st.markdown("**⚖️ Priority System:**")
    st.write("- **Critical (CCTV):** 100% weight - Security system")
    st.write("- **High (Ticket Booth):** 80% weight - Essential services") 
    st.write("- **Low (Public Address):** 40% weight - Communication system")
    
    st.markdown("**🔄 Dynamic Adjustments:**")
    st.write("- **Emergency factor:** 1.3x power allocation during emergencies")
    st.write("- **Rush hour factor:** 1.1x during high traffic periods (>400 people)")
    st.write("- **Efficiency tracking:** Real-time system performance monitoring")

def skip_time(minutes, piezo, consumption, agent):
    """
    Skip ahead in simulation by a given number of minutes, updating battery, allocation, and history.
    Only record actual events in history. Battery never below 20%.
    """
    step = 1 if minutes <= 15 else 5 if minutes <= 60 else 10
    remaining = minutes

    while remaining > 0:
        current_step = min(step, remaining)
        context_data = get_context_data(st.session_state.simulated_time)

        # Predict energy per minute
        predicted_energy_per_min = piezo.predict_power_from_footfall(context_data.footfall) / 60

        # System consumption
        system_loads = generate_system_loads(context_data, consumption)
        actual_power_used_per_min = sum([s.predicted_power for s in system_loads]) / 60

        predicted_energy = predicted_energy_per_min * current_step
        actual_power_used = actual_power_used_per_min * current_step

        # Ensure battery >= 20%
        max_available = predicted_energy + (st.session_state.battery_energy - 20)
        if actual_power_used > max_available:
            scale_factor = max_available / actual_power_used
            actual_power_used *= scale_factor

        # Update battery
        st.session_state.battery_energy += predicted_energy - actual_power_used
        st.session_state.battery_energy = max(20, min(100, st.session_state.battery_energy))

        # Determine if current time is an event
        current_time_str = st.session_state.simulated_time.strftime("%H:%M")
        is_event = any(current_time_str == e["time"] for e in STATION_SCHEDULE)

        # Record only events
        if is_event:
            st.session_state.historical_data.append({
                "timestamp": st.session_state.simulated_time,
                "footfall": context_data.footfall if context_data.footfall > 0 else "No Footfall",
                "energy_generated": predicted_energy if predicted_energy > 0 else "No Generation",
                "total_allocated": actual_power_used if actual_power_used > 0 else "No Allocation",
                "battery_level": st.session_state.battery_energy
            })

        # Advance time
        st.session_state.simulated_time += timedelta(minutes=current_step)
        remaining -= current_step


def simulation_page():
    st.header("⚡ Real-Time Simulation")
    st.markdown("This simulation shows the AI controller's real-time power allocation decisions.")
    st.markdown("---")

    # ── Session state init ─────────────────────────────────────────────
    if "running" not in st.session_state:
        st.session_state.running = False
    if "battery_energy" not in st.session_state:
        st.session_state.battery_energy = 50.0
    if "historical_data" not in st.session_state:
        st.session_state.historical_data = []
    if "simulated_time" not in st.session_state:
        st.session_state.simulated_time = datetime(2025, 8, 18, 6, 0, 0)

    # ── Models & Agent ────────────────────────────────────────────────
    if "piezo" not in st.session_state:
        st.session_state.piezo = PiezoEnergyPredictor()
        st.session_state.piezo.train_model()
    if "consumption" not in st.session_state:
        st.session_state.consumption = SystemConsumptionPredictor()
        st.session_state.consumption.train_models()
    if "agent" not in st.session_state:
        st.session_state.agent = SmartAllocationAgent()

    piezo = st.session_state.piezo
    consumption = st.session_state.consumption
    agent = st.session_state.agent
    controller = AIController()

    # ── Controls ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    # Start/Stop
    with col1:
        if st.session_state.running:
            if st.button("⏹️ Stop Simulation", type="secondary", use_container_width=True):
                st.session_state.running = False
                st.balloons()
                st.rerun()
        else:
            if st.button("▶️ Start Simulation", type="primary", use_container_width=True):
                st.session_state.running = True
                st.session_state.simulated_time = datetime(2025, 8, 18, 6, 0, 0)
                st.session_state.battery_energy = 50.0
                st.session_state.historical_data = []
                st.rerun()

    # Reset
    with col2:
        if st.button("🔄 Reset System", type="secondary", use_container_width=True):
            st.session_state.running = True
            st.session_state.battery_energy = 50.0
            st.session_state.historical_data = []
            st.session_state.simulated_time = datetime(2025, 8, 18, 6, 0, 0)
            st.rerun()

    # Skip Options
    with col3:
        st.markdown("**⏭️ Skip Options**")
        event_names = [f"{e['time']} - {e['event_type']}" for e in STATION_SCHEDULE]
        chosen_event = st.selectbox("Choose event", event_names, index=0)
        if st.button("⏩ Skip to Event", use_container_width=True, key="skip_event"):
            now = st.session_state.simulated_time
            h, m = map(int, chosen_event.split(" - ")[0].split(":"))
            target_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if target_dt <= now:
                target_dt += timedelta(days=1)
            delta_minutes = int((target_dt - now).total_seconds() // 60)
            skip_time(delta_minutes, piezo, consumption, agent)
            st.success(f"⏩ Skipped to {chosen_event} → {target_dt}")
            st.rerun()

        preset = st.selectbox("Quick skip", ["5 min", "15 min", "30 min", "1 hour"], index=1)
        if st.button("⏭️ Skip Preset", use_container_width=True, key="skip_preset"):
            add_minutes = {"5 min": 5, "15 min": 15, "30 min": 30, "1 hour": 60}[preset]
            skip_time(add_minutes, piezo, consumption, agent)
            st.success(f"⏭️ Skipped ahead by {add_minutes} minutes → {st.session_state.simulated_time}")
            st.rerun()

    # Simulation speed
    with col4:
        speed_label = st.selectbox("⚡ Simulation Speed", ["Slow", "Normal", "Fast"], index=0)
        speed_cfg = {"Slow": (4, 300), "Normal": (2, 300), "Fast": (0.2, 300)}
        base_sleep, base_step = speed_cfg[speed_label]

    # ── Station Status ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏢 Station Status")
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    context_data = get_context_data(st.session_state.simulated_time)
    with status_col1:
        st.metric("🕐 Simulation Time", st.session_state.simulated_time.strftime("%H:%M:%S"))
    with status_col2:
        st.metric("👥 Pedestrian Count", f"{context_data.footfall} people")
    with status_col3:
        st.metric("🌡️ Temperature", f"{context_data.ambient_temp:.1f} °C")
    with status_col4:
        st.metric("📋 Next Event", get_next_event(st.session_state.simulated_time))

    # Battery status
    battery_percentage = st.session_state.battery_energy
    if battery_percentage > 70:
        st.success(f"🔋 Battery Status: Excellent ({battery_percentage:.0f}%)")
    elif battery_percentage > 30:
        st.warning(f"🔋 Battery Status: Good ({battery_percentage:.0f}%)")
    else:
        st.error(f"🔋 Battery Status: Low ({battery_percentage:.0f}%)")

    # ── Core Simulation Loop ─────────────────────────────────────────
    if st.session_state.running:
        predicted_energy = piezo.predict_power_from_footfall(context_data.footfall)
        system_loads = generate_system_loads(context_data, consumption)
        total_load_power = sum([s.predicted_power for s in system_loads])

        # Ensure battery never drops below 20%
        min_battery = 20
        available_battery = max(0, st.session_state.battery_energy - min_battery)
        if total_load_power > predicted_energy + available_battery:
            scale_factor = (predicted_energy + available_battery) / total_load_power
            for s in system_loads:
                s.predicted_power *= scale_factor
            total_load_power = sum([s.predicted_power for s in system_loads])

        # Update battery
        st.session_state.battery_energy += predicted_energy - total_load_power
        st.session_state.battery_energy = max(min_battery, min(100, st.session_state.battery_energy))

        # Agent allocation
        total_available = predicted_energy + st.session_state.battery_energy
        allocated_power, remaining_power, allocation_log = agent.allocate_power(
            total_available, system_loads, context_data
        )
        actual_power_used = sum(allocated_power.values())

        # Store history (avoid duplicate timestamps)
        history_entry = {
            "timestamp": st.session_state.simulated_time,
            "footfall": context_data.footfall or 0,
            "energy_generated": predicted_energy or 0,
            "total_allocated": actual_power_used or 0,
            "battery_level": st.session_state.battery_energy
        }
        unique_history = {row["timestamp"]: row for row in st.session_state.historical_data}
        unique_history[history_entry["timestamp"]] = history_entry
        st.session_state.historical_data = list(unique_history.values())

        # Controller actions
        controller.run_control_loop(
            predicted_energy=predicted_energy,
            actual_power_used=actual_power_used,
            total_available=total_available,
            allocation_log=allocation_log,
            context_data=context_data,
            system_loads=system_loads
        )

        # Advance time
        st.session_state.simulated_time += timedelta(seconds=base_step)
        time.sleep(base_sleep)
        st.rerun()
    else:
        st.info("⏸️ Simulation paused. Click 'Start Simulation' to resume.")

    # ── History & Stats ─────────────────────────────────────────────
    if st.session_state.historical_data:
        history_df = pd.DataFrame(st.session_state.historical_data)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df = history_df.sort_values('timestamp')

        with st.expander("📈 Complete Simulation Results", expanded=False):
            st.dataframe(history_df.fillna(0), use_container_width=True)

        st.subheader("📊 Simulation Statistics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Avg Generation", f"{history_df['energy_generated'].mean():.2f} Wh")
        with c2:
            st.metric("Max Pedestrian Traffic", int(history_df['footfall'].max()))
        with c3:
            total_gen = history_df['energy_generated'].sum()
            total_cons = history_df['total_allocated'].sum()
            eff = min(100, (total_gen / total_cons) * 100) if total_cons > 0 else 100
            st.metric("Energy Usage Efficiency", f"{eff:.1f}%")
        with c4:
            st.metric("Final Battery", f"{history_df['battery_level'].iloc[-1]:.0f}%")

        if len(history_df) > 1:
            st.subheader("📈 Energy Flow Over Time")
            chart_data = history_df[['timestamp','energy_generated','total_allocated','battery_level']].copy()
            chart_data.set_index('timestamp', inplace=True)
            st.line_chart(chart_data)

        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.historical_data = []
            st.session_state.battery_energy = 50.0
            st.session_state.simulated_time = datetime(2025, 8, 18, 6, 0, 0)
            st.rerun()


# ───────────────────────────────────────────────────────────────────────────────
# Main App
def app_main():
    local_css()
    st.title("⚡ AI-Powered Piezoelectric Energy Management")
    st.markdown("*Train Station Smart Micro-Grid Simulation*")
    st.markdown("---")

    with st.sidebar:
        st.subheader("🧭 Navigation")
        page = st.radio("Select Page:", ["🏠 Home", "⚡ Simulation", "📅 Station Schedule", "🤖 AI Models"])
        st.markdown("---")
        st.subheader("ℹ️ System Information")
        st.write("**Piezo Sensors:** Active")
        st.write("**AI Models:** Trained" if "models_trained" in st.session_state else "**AI Models:** Loading…")
        if "battery_energy" in st.session_state:
            b = st.session_state.battery_energy / 100.0
            if b > 0.7: st.success(f"🔋 Battery: {b*100:.0f}%")
            elif b > 0.3: st.warning(f"🔋 Battery: {b*100:.0f}%")
            else: st.error(f"🔋 Battery: {b*100:.0f}%")

    if page == "🏠 Home":
        home_page()
    elif page == "⚡ Simulation":
        simulation_page()
    elif page == "📅 Station Schedule":
        station_schedule_page()
    elif page == "🤖 AI Models":
        ai_models_page()

# Entrypoint
if __name__ == "__main__":
    app_main()
