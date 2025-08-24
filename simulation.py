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

# --- Custom CSS to hide the GitHub icon ---
# This CSS is applied globally to the Streamlit app's header.
hide_github_icon = """
#GithubIcon {
    display: none !important;
}
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)

# --- CSV Data (Embedded Piezo dataset) ---
# This data represents readings from a piezoelectric floor tile.
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
# This data models the power consumption of various station systems.
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

# --- AI Models (functions now instead of class) ---
@st.cache_resource
def get_piezo_predictor():
    """Trains and caches the Piezoelectric energy prediction model."""
    df = pd.read_csv(io.StringIO(csv_data))
    
    X = pd.DataFrame()
    X['voltage'] = df['voltage(v)']
    X['current'] = df['current(uA)']
    X['weight'] = df['weight(kgs)']
    X['location_center'] = (df['step_location'] == 'Center').astype(int)
    X['location_edge'] = (df['step_location'] == 'Edge').astype(int)
    
    y = df['Power(mW)']
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return model, mae, r2

@st.cache_resource
def get_consumption_predictor():
    """Trains and caches the system consumption prediction models."""
    df = pd.read_csv(io.StringIO(system_consumption_data))
    
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    df['is_weekend'] = (df['day_type'] == 'Weekend').astype(int)
    
    X = df[['hour', 'footfall', 'ambient_temp', 'is_weekend']].copy()
    
    models = {}
    systems = ['cctv_power', 'panels_power', 'ventilation_power', 'ticket_booth_power']
    
    for system in systems:
        y = df[system]
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        models[system] = model
            
    return models

def predict_piezo_power_from_footfall(model, footfall, avg_weight=65):
    """Predicts power generation based on footfall using the trained model."""
    base_voltage = min(footfall * 0.25 + random.uniform(-1, 3), 55)
    current = 50 + random.uniform(-3, 8)
    
    location_weights = {'Center': 0.5, 'Edge': 0.3, 'Corner': 0.2}
    location = random.choices(list(location_weights.keys()), 
                             weights=list(location_weights.values()))[0]
    
    location_center = 1 if location == 'Center' else 0
    location_edge = 1 if location == 'Edge' else 0
    
    features = pd.DataFrame([[base_voltage, current, avg_weight, location_center, location_edge]], 
                             columns=['voltage', 'current', 'weight', 'location_center', 'location_edge'])
    
    predicted_power_mw = max(0, model.predict(features)[0])
    
    power_per_step = (predicted_power_mw / 1000) * 0.5 
    total_power_w = power_per_step * footfall
    total_power_w = total_power_w + random.uniform(-0.5, 0.5)
    total_power_w = min(total_power_w, 50.0)
    
    return max(total_power_w, 0.1)

def predict_system_consumption(models, context_data):
    """Predicts power consumption for all systems."""
    hour = context_data.time_of_day.hour
    footfall = context_data.footfall
    ambient_temp = context_data.ambient_temp
    is_weekend = 1 if context_data.day_type == 'Weekend' else 0
    
    features = pd.DataFrame([[hour, footfall, ambient_temp, is_weekend]], 
                             columns=['hour', 'footfall', 'ambient_temp', 'is_weekend'])
    
    predictions = {}
    for system_name, model in models.items():
        pred = model.predict(features)[0]
        predictions[system_name] = max(0, pred)
            
    return predictions

def allocate_power(available_power, systems, context_data):
    """Allocates available power to systems based on priority."""
    priority_weights = {"Critical": 1.0, "High": 0.8, "Medium": 0.6, "Low": 0.4}
    allocated_power = {}
    remaining_power = available_power
    
    emergency_factor = 1.3 if "emergency" in context_data.event_triggers.lower() else 1.0
    rush_hour_factor = 1.1 if context_data.footfall > 400 else 1.0
    
    sorted_systems = sorted(systems, 
                             key=lambda x: priority_weights.get(x.priority, 0), 
                             reverse=True)
    
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

# --- Mock Data Generators ---
def get_context_data(sim_time):
    """Generates a context object for the simulation."""
    day_type = "Weekday" if sim_time.weekday() < 5 else "Weekend"
    
    event = "Normal Operation"
    footfall_min = 80
    footfall_max = 250
    
    for scheduled_event in STATION_SCHEDULE:
        event_time_str = scheduled_event['time']
        event_hour, event_minute = map(int, event_time_str.split(':'))
        event_dt = sim_time.replace(hour=event_hour, minute=event_minute, second=0, microsecond=0)
        time_diff = abs((sim_time - event_dt).total_seconds())
        
        if time_diff < 300:
            event = scheduled_event['event_type']
            footfall_min, footfall_max = scheduled_event['footfall_range']
            break
    
    footfall = random.randint(footfall_min, footfall_max)
    ambient_temp = 18 + random.uniform(-2, 8)
    
    return ContextFactors(footfall, sim_time, day_type, event, ambient_temp)

def generate_system_loads(context_data, consumption_models):
    """Generates system load data based on predictions."""
    predictions = predict_system_consumption(consumption_models, context_data)
    
    systems = []
    
    systems.append(SystemLoadData(
        system_id="CCTV Security",
        base_power=predictions['cctv_power'],
        predicted_power=predictions['cctv_power'] * (1 + random.uniform(-0.1, 0.1)),
        min_power=predictions['cctv_power'] * 0.7,
        max_power=predictions['cctv_power'] * 1.3,
        priority="Critical",
        status="Active",
        efficiency=0.95
    ))
    
    systems.append(SystemLoadData(
        system_id="Info Panels",
        base_power=predictions['panels_power'],
        predicted_power=predictions['panels_power'] * (1 + random.uniform(-0.15, 0.15)),
        min_power=predictions['panels_power'] * 0.5,
        max_power=predictions['panels_power'] * 1.2,
        priority="High",
        status="Active",
        efficiency=0.88
    ))
    
    systems.append(SystemLoadData(
        system_id="Ventilation",
        base_power=predictions['ventilation_power'],
        predicted_power=predictions['ventilation_power'] * (1 + random.uniform(-0.2, 0.2)),
        min_power=predictions['ventilation_power'] * 0.6,
        max_power=predictions['ventilation_power'] * 1.4,
        priority="Medium",
        status="Active",
        efficiency=0.82
    ))
    
    systems.append(SystemLoadData(
        system_id="Ticket Booth",
        base_power=predictions['ticket_booth_power'],
        predicted_power=predictions['ticket_booth_power'] * (1 + random.uniform(-0.1, 0.1)),
        min_power=predictions['ticket_booth_power'] * 0.8,
        max_power=predictions['ticket_booth_power'] * 1.1,
        priority="High",
        status="Active",
        efficiency=0.92
    ))
    
    return systems

def get_next_event(sim_time):
    """Find the next scheduled event."""
    for event in STATION_SCHEDULE:
        event_time_str = event['time']
        event_hour, event_minute = map(int, event_time_str.split(':'))
        event_dt = sim_time.replace(hour=event_hour, minute=event_minute, second=0)
        
        if event_dt > sim_time:
            return f"{event['event_type']} - {event_time_str}"
    
    return "No more events today"

def display_simulation_status(placeholder, simulated_time, battery_energy, historical_data, piezo_model, consumption_models, piezo_mae, piezo_r2):
    """Renders the simulation dashboard within the given placeholder."""
    
    with placeholder.container():
        st.header("⚡ Smart Micro-Grid Simulation")
        
        current_time_str = simulated_time.strftime('%Y-%m-%d %H:%M:%S')
        next_event = get_next_event(simulated_time)
        
        st.info(f"**Current Simulation Time:** {current_time_str} | **Next Event:** {next_event}")
        
        # --- Run one step of the simulation loop ---
        context_data = get_context_data(simulated_time)
        predicted_energy = predict_piezo_power_from_footfall(piezo_model, context_data.footfall)
        system_loads = generate_system_loads(context_data, consumption_models)
        total_available = predicted_energy + battery_energy
        allocated_power, remaining_power, allocation_log = allocate_power(total_available, systems=system_loads, context_data=context_data)
        
        energy_generation_rate = predicted_energy * 0.85
        battery_charge_rate = min(remaining_power * 0.1, 5.0)
        new_battery_energy = min(battery_energy + battery_charge_rate, 100.0)

        # --- Main Display Area ---
        st.markdown("---")
        
        # Energy Source Status
        st.markdown(
            f"""
            <div class="energy-source">
                <h3>🔋 Piezoelectric Energy Source</h3>
                <p><strong>Instantaneous Generation:</strong> {predicted_energy:.2f} W</p>
                <p><strong>Battery Storage:</strong> {battery_energy:.2f} W</p>
                <p><strong>Total Available:</strong> {total_available:.2f} W</p>
                <p><strong>Generation Efficiency:</strong> 85%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 System Power Allocation")
            allocation_df = pd.DataFrame(allocation_log)
            st.dataframe(allocation_df, use_container_width=True)
            
            total_allocated = sum(allocated_power.values())
            if total_allocated <= total_available * 0.8:
                st.success(f"✅ Power Balance: Optimal ({total_allocated:.1f}W / {total_available:.1f}W)")
            elif total_allocated <= total_available:
                st.warning(f"⚠️ Power Balance: Tight ({total_allocated:.1f}W / {total_available:.1f}W)")
            else:
                st.error(f"❌ Power Balance: Deficit ({total_allocated:.1f}W / {total_available:.1f}W)")
            
        with col2:
            st.subheader("🌡️ Environmental Factors")
            st.metric("Pedestrian Traffic", f"{context_data.footfall} people")
            st.metric("Ambient Temperature", f"{context_data.ambient_temp:.1f}°C")
            st.metric("Current Event", context_data.event_triggers)
            st.metric("Day Type", context_data.day_type)
            
            generation_efficiency = (predicted_energy / max(context_data.footfall * 0.01, 0.1)) * 100
            st.metric("Generation Efficiency", f"{generation_efficiency:.1f}%")

        with st.expander("🔍 Detailed AI Analysis Log", expanded=False):
            st.info("🤖 Piezoelectric Energy Prediction Model Running...")
            st.write(f"Model Performance: MAE={piezo_mae:.3f}, R²={piezo_r2:.3f}")
            st.write(f"Footfall: {context_data.footfall} → Predicted Energy: {predicted_energy:.2f}W")
            
            st.info("🏭 System Consumption Prediction Model Running...")
            for system in system_loads:
                st.write(f"- {system.system_id}: {system.predicted_power:.1f}W (Priority: {system.priority})")
            
            st.info("🧠 Smart Allocation Agent Making Decisions...")
            st.write(f"Total Available Power: {total_available:.2f}W")
            st.write("Allocation Strategy: Priority-based optimization with efficiency consideration")

        historical_data.append({
            "timestamp": simulated_time.strftime('%Y-%m-%d %H:%M:%S'),
            "footfall": context_data.footfall,
            "event": context_data.event_triggers,
            "energy_generated": predicted_energy,
            "total_available": total_available,
            "total_allocated": sum(allocated_power.values()),
            "remaining_power": remaining_power,
            "ambient_temp": context_data.ambient_temp,
            "battery_level": new_battery_energy
        })

        return simulated_time + timedelta(seconds=300), new_battery_energy, historical_data

def run_simulation(start_time, initial_battery, piezo_model, consumption_models, piezo_mae, piezo_r2):
    """Main simulation loop using a placeholder for dynamic updates."""
    
    st.session_state.running = True
    
    placeholder = st.empty()
    simulated_time = start_time
    battery_energy = initial_battery
    historical_data = st.session_state.historical_data
    
    while st.session_state.running:
        simulated_time, battery_energy, historical_data = display_simulation_status(
            placeholder, simulated_time, battery_energy, historical_data, piezo_model, consumption_models, piezo_mae, piezo_r2
        )
        
        # Adjust sleep time based on events for dynamic simulation speed
        next_event_time = min([simulated_time.replace(hour=int(e['time'].split(':')[0]), minute=int(e['time'].split(':')[1]), second=0) for e in STATION_SCHEDULE if simulated_time.replace(hour=int(e['time'].split(':')[0]), minute=int(e['time'].split(':')[1]), second=0) > simulated_time] or [simulated_time + timedelta(hours=1)])
        time_to_sleep = (next_event_time - simulated_time).total_seconds() / 1500 if (next_event_time - simulated_time).total_seconds() > 0 else 2
        time.sleep(time_to_sleep)

def show_simulation_results(historical_data):
    """Displays the final simulation results and statistics."""
    st.info("⏸️ Simulation paused. Click 'Start' button to resume.")
    
    if historical_data:
        with st.expander("📈 Complete Simulation Results", expanded=True):
            history_df = pd.DataFrame(historical_data)
            st.dataframe(history_df, use_container_width=True)
            
            st.subheader("📊 Simulation Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_generation = history_df['energy_generated'].mean()
                st.metric("Avg Energy Generation", f"{avg_generation:.2f}W")
            with col2:
                max_footfall = history_df['footfall'].max()
                st.metric("Max Pedestrian Traffic", f"{max_footfall} people")
            with col3:
                total_generated = history_df['energy_generated'].sum()
                total_consumed = history_df['total_allocated'].sum()
                efficiency = (total_consumed / total_generated) * 100 if total_generated > 0 else 0
                st.metric("Energy Usage Efficiency", f"{efficiency:.1f}%")
            with col4:
                final_battery = historical_data[-1]['battery_level']
                st.metric("Final Battery Level", f"{final_battery:.1f}W")
            
            if len(history_df) > 1:
                st.subheader("📈 Energy Flow Over Time")
                chart_data = history_df[['timestamp', 'energy_generated', 'total_allocated', 'battery_level']].copy()
                chart_data['timestamp'] = pd.to_datetime(chart_data['timestamp'])
                chart_data.set_index('timestamp', inplace=True)
                st.line_chart(chart_data)

# --- Page Functions ---
def home_page():
    st.header("🔋 What is Piezoelectric Energy?")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://earimediaprodweb.azurewebsites.net/Api/v1/Multimedia/51e01af1-0782-49a1-9415-0d44220dd5d2/Rendition/low-res/Content/Public", caption="Piezoelectric energy concept")
    with col2:
        st.write("Piezoelectric materials generate electrical charge when mechanical stress is applied to them. Simply put, they can convert human footsteps into usable electrical energy. This simulation demonstrates a micro-grid system that harvests this energy from a busy environment like a train station and intelligently distributes it to various systems.")

    st.markdown("---")
    st.header("📱 About This Application")
    st.write("This application is a real-time simulation of an AI-powered micro-grid system. It demonstrates how piezoelectric energy harvesting can be efficiently managed using machine learning models and intelligent power allocation algorithms.")
    st.markdown("### 🎯 User Guide:")
    st.markdown("- **🏠 Home:** You are here! Provides basic concepts and application overview.")
    st.markdown("- **⚡ Simulation:** The heart of the application. Click 'Start Simulation' to see real-time AI controller power allocation decisions.")
    st.markdown("- **📅 Station Schedule:** Shows the static schedule of daily activities like train arrivals and maintenance.")
    st.markdown("- **🤖 AI Models:** Details and performance metrics of the machine learning models used.")
    st.markdown("### 🔬 Technical Features:")
    st.markdown("- **Real-time ML Prediction:** Linear regression for energy generation, Random Forest for consumption")
    st.markdown("- **Priority-based Allocation:** Critical systems (CCTV) get power first")
    st.markdown("- **Dynamic Adaptation:** AI adjusts to changing footfall and environmental conditions")
    st.markdown("- **Battery Management:** Intelligent charging and discharge optimization")

def station_schedule_page():
    st.header("📅 Daily Station Schedule")
    st.write("Static schedule of all important station events that the AI uses to predict energy demand surges. The simulation follows this schedule to create realistic event patterns.")
    schedule_df = pd.DataFrame(STATION_SCHEDULE)
    schedule_df.rename(columns={'time': 'Time', 'event_type': 'Event Type', 'footfall_range': 'Pedestrian Traffic Range'}, inplace=True)
    st.dataframe(schedule_df, use_container_width=True)
    st.markdown("---")
    st.subheader("📊 Schedule Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🚂 Peak Hours:**")
        st.write("- 07:15-08:05: Morning rush")
        st.write("- 12:30-12:35: Lunch traffic")
        st.write("- 16:45-17:35: Evening rush")
    with col2:
        st.markdown("**🔧 Maintenance Hours:**")
        st.write("- 10:30: Scheduled cleaning")
        st.write("- 14:00: General maintenance")
        st.write("- 00:00-06:00: Station closed")
    st.markdown("---")
    st.subheader("📈 Traffic Pattern Insights")
    schedule_data = []
    for event in STATION_SCHEDULE:
        hour = int(event['time'].split(':')[0])
        avg_footfall = sum(event['footfall_range']) / 2
        schedule_data.append({'Hour': hour, 'Expected Footfall': avg_footfall})
    schedule_chart_df = pd.DataFrame(schedule_data)
    st.bar_chart(schedule_chart_df.set_index('Hour'))

def ai_models_page():
    st.header("🤖 AI Models and Performance")
    st.write("This page provides an overview of the machine learning models that power the simulation.")
    
    with st.spinner("Training AI models..."):
        piezo_model, piezo_mae, piezo_r2 = get_piezo_predictor()
        consumption_models = get_consumption_predictor()
    
    st.subheader("⚡ Piezoelectric Energy Prediction Model")
    st.write("This model predicts the energy generated from footsteps based on voltage, current, and weight. It is a **Linear Regression** model trained on real-world piezoelectric data.")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Absolute Error (MAE)", f"{piezo_mae:.3f}")
    with col2:
        st.metric("R² Score", f"{piezo_r2:.3f}")
    st.info("MAE indicates the average error in prediction. R² close to 1 indicates a good fit.")
    
    st.subheader("🏭 System Consumption Prediction Model")
    st.write("This model predicts the power consumption of various station systems (CCTV, info panels, etc.) based on contextual factors like time of day, footfall, and ambient temperature. It uses a **Random Forest Regressor** for higher accuracy.")
    st.info("The Random Forest model learns complex, non-linear relationships to provide more accurate consumption forecasts.")
    
    st.subheader("🧠 Smart Power Allocation Agent")
    st.write("This is the core logic that orchestrates the entire system. It is a rule-based AI agent that prioritizes power distribution based on system criticality. It ensures essential systems stay online even when energy generation is low.")
    st.markdown("""- **Critical:** CCTV Security (highest priority)
- **High:** Info Panels, Ticket Booth
- **Medium:** Ventilation
- **Low:** Other non-essential systems (not simulated)""")

def simulation_page():
    if "running" not in st.session_state:
        st.session_state.running = False
    if "simulated_time" not in st.session_state:
        st.session_state.simulated_time = datetime(2025, 8, 18, 6, 0, 0)
    if "battery_energy" not in st.session_state:
        st.session_state.battery_energy = 50.0
    if "historical_data" not in st.session_state:
        st.session_state.historical_data = []

    st.header("⚡ Smart Micro-Grid Simulation")

    col1, col2, col3 = st.columns(3)
    start_button = col1.button("▶️ Start Simulation", type="primary", disabled=st.session_state.running)
    stop_button = col2.button("⏹️ Pause Simulation", type="secondary", disabled=not st.session_state.running)
    reset_button = col3.button("🔄 Reset Simulation", type="secondary")

    if start_button:
        st.session_state.running = True
    elif stop_button:
        st.session_state.running = False
    elif reset_button:
        st.session_state.running = False
        st.session_state.historical_data = []
        st.session_state.battery_energy = 50.0
        st.session_state.simulated_time = datetime(2025, 8, 18, 6, 0, 0)

    with st.spinner("Training AI models..."):
        piezo_model, piezo_mae, piezo_r2 = get_piezo_predictor()
        consumption_models = get_consumption_predictor()

    if st.session_state.running:
        run_simulation(
            st.session_state.simulated_time,
            st.session_state.battery_energy,
            piezo_model,
            consumption_models,
            piezo_mae,
            piezo_r2
        )
    else:
        show_simulation_results(st.session_state.historical_data)


# --- Main App Logic ---
local_css()

st.sidebar.title("🚉 AI Energy Management")
page = st.sidebar.radio("Go to", ("🏠 Home", "⚡ Simulation", "📅 Station Schedule", "🤖 AI Models"))

if page == "🏠 Home":
    home_page()
elif page == "⚡ Simulation":
    simulation_page()
elif page == "📅 Station Schedule":
    station_schedule_page()
elif page == "🤖 AI Models":
    ai_models_page()
