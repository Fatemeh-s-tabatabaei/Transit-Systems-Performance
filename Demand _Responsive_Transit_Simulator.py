import numpy as np
import pandas as pd
import os

# === STEP 1: Passenger demand simulation ===
class StopsDemandSimulator:
    def __init__(self, num_stops=(2, 5), sim_time=300, network_rate=60, seed=None):
        # if seed is not None:
        #     np.random.seed(seed)
        self.num_stops = num_stops
        self.sim_time = sim_time
        self.network_rate = network_rate
        self.bus_stops = {(i, j): [] for i in range(num_stops[0]) for j in range(num_stops[1])}

    def simulate_network(self):
        current_time = 0.0
        mean_interarrival = 60 / self.network_rate
        while True:
            wait = np.random.exponential(scale=mean_interarrival)
            current_time += wait
            if current_time >= self.sim_time:
                break
            chosen_line = np.random.choice([0, 1], p=[0.7, 0.3])
            chosen_stop = np.random.randint(0, self.num_stops[1])
            self.bus_stops[(chosen_line, chosen_stop)].append(round(current_time, 2))

        for key in self.bus_stops:
            self.bus_stops[key] = sorted([t for t in self.bus_stops[key] if 0 <= t <= self.sim_time])
        return self.bus_stops

# === STEP 2: Initialization ===
simulator = StopsDemandSimulator(num_stops=(2, 5), sim_time=180, network_rate=60, seed=42)
bus_stop_arrivals = simulator.simulate_network()
destinations = ['D1', 'D2', 'D3']
demand, global_passenger_id = {}, 0
for stop, times in bus_stop_arrivals.items():
    demand[stop] = []
    for t in times:
        dest = np.random.choice(destinations)
        demand[stop].append({'passenger_id': global_passenger_id, 'arrival_time': t, 'destination': dest, 'served': False})
        global_passenger_id += 1

if (0, 0) not in demand:
    demand[(0, 0)] = []
demand[(0, 0)].append({'passenger_id': global_passenger_id, 'arrival_time': 27.6, 'destination': 'D3', 'served': False})
global_passenger_id += 1

# === STEP 3: Vehicle operation ===
events = []
pickup_times = {}  # Track when each passenger was picked up
class Vehicle:
    def __init__(self, vehicle_id, start_time, start_stop, chosen_destination):
        self.vehicle_id = vehicle_id
        self.current_time = start_time
        self.current_stop = start_stop
        self.chosen_destination = chosen_destination
        self.visited_stops = {start_stop}
        self.current_load = 0
        self.max_capacity = 15
        events.append({"Time": round(start_time, 2), "vehicle_id": vehicle_id, "stop_id": start_stop,
                       "passenger_arrival": None, "passenger_pickup": start_time, "delay": None,
                       "#riders": self.current_load, "destination": chosen_destination, "event": 'start'})

    def travel_to(self, next_stop):
        travel_time = max(1, np.random.normal(5, 1))
        self.current_time += travel_time
        self.current_stop = next_stop
        self.visited_stops.add(next_stop)
        return travel_time

    def pickup_passengers(self, demand):
        picked = []
        max_wait_time = 20
        if self.current_stop in demand:
            for passenger in demand[self.current_stop]:
                if (passenger['destination'] == self.chosen_destination and
                    not passenger['served'] and
                    passenger['arrival_time'] <= self.current_time):
                    
                    waited_time = self.current_time - passenger['arrival_time']
                    if waited_time <= max_wait_time and self.current_load + len(picked) < self.max_capacity:
                        passenger['served'] = True
                        picked.append(passenger)
                        pickup_times[passenger['passenger_id']] = self.current_time
        self.current_load += len(picked)
        return picked

# === STEP 4: Vehicle dispatch ===
vehicles, vehicle_count, current_time = [], 0, 0.0
sim_time = 300
last_vehicle_start, last_vehicle_dest = None, None
while current_time < sim_time:
    # interarrival = np.random.exponential(scale=1/4)
    interarrival = 5
    current_time += interarrival
    if current_time >= sim_time:
        break
    for i in range(3):
        start_row = np.random.choice([0, 1])
        start_stop = (start_row, 0)
        # chosen_dest = np.random.choice(destinations)
        chosen_dest = destinations[i]
        if last_vehicle_dest == chosen_dest and current_time < last_vehicle_start + 10:
            current_time = last_vehicle_start + 10
        v = Vehicle(vehicle_id=vehicle_count, start_time=round(current_time, 2),
                    start_stop=start_stop, chosen_destination=chosen_dest)
        vehicles.append(v)
        vehicle_count += 1
    last_vehicle_start = current_time
    last_vehicle_dest = chosen_dest

# === STEP 5: Route execution ===
for v in vehicles:
    for col in range(0, 4):
        candidate_stops = []
        for row in [0, 1]:
            s = (row, col)
            if s not in v.visited_stops and s in demand:
                if any(p['destination'] == v.chosen_destination and not p['served'] for p in demand[s]):
                    candidate_stops.append(s)
        if candidate_stops:
            dists = [abs(v.current_stop[0] - s[0]) + abs(v.current_stop[1] - s[1]) for s in candidate_stops]
            chosen_stop = candidate_stops[np.argmin(dists)]
            v.travel_to(chosen_stop)
            picked = v.pickup_passengers(demand)
            if picked:
                delays = [round(v.current_time - p['arrival_time'], 2) for p in picked]
                avg_delay = round(sum(delays) / len(delays), 2)
                events.append({"Time": round(v.current_time, 2), "vehicle_id": v.vehicle_id, "stop_id": chosen_stop,
                               "passenger_arrival": ','.join(str(p['arrival_time']) for p in picked),
                               "passenger_pickup": round(v.current_time, 2), "delay": avg_delay,
                               "#riders": v.current_load, "destination": v.chosen_destination, "event": 'pickup'})
    v.travel_to((v.current_stop[0], 4))
    v.current_time += np.random.normal(10, 2)
    events.append({"Time": round(v.current_time, 2), "vehicle_id": v.vehicle_id,
                   "stop_id": "Destination " + v.chosen_destination, "passenger_dropoff": round(v.current_time, 2),
                   "passenger_pickup": None, "delay": None, "#riders": v.current_load,
                   "destination": v.chosen_destination, "event": 'dropoff'})

# === STEP 6: Post-processing ===
df = pd.DataFrame(events)
df.sort_values(by='Time', inplace=True)

# Calculate delays and metrics
served_delays = []
timeout_delays = []
for stop, passengers in demand.items():
    for p in passengers:
        if p['served']:
            waited = pickup_times.get(p['passenger_id'], None)
            if waited is not None:
                waited = waited - p['arrival_time']
                served_delays.append(min(waited, 20))
        else:
            timeout_delays.append(20)

all_waits = served_delays + timeout_delays
mean_wait = round(np.mean(all_waits), 2)
std_wait = round(np.std(all_waits), 2)
cv_wait = round(std_wait / mean_wait, 4) if mean_wait > 0 else None
total_ridership = len(served_delays)
timed_out_count = len(timeout_delays)

print("Average waiting time:", mean_wait)
print("Std deviation:", std_wait)
print("Coefficient of variation:", cv_wait)
print("Total ridership:", total_ridership)
print("Timed-out passengers:", timed_out_count)


demand_ls = [60, 120, 180, 240]
dispatch_interval_ls = [5, 10, 15, 20] 
num_replication = 10

Ave_wt = []
CoV_wt = []
ridership = []
time_out_riders = []
folder_name = "./results/DRT/"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

for demand_level in demand_ls:
    for dispatch_interval in dispatch_interval_ls:
        Ave_wt.clear()
        CoV_wt.clear()
        ridership.clear()
        time_out_riders.clear()
        
        for rep in range(num_replication):
            # --- Step 1: Passenger Demand Simulation ---
            simulator = StopsDemandSimulator(num_stops=(2, 5), sim_time=180, network_rate=demand_level, seed=rep)
            bus_stop_arrivals = simulator.simulate_network()
            destinations = ['D1', 'D2', 'D3']
            demand = {}
            global_passenger_id = 0
            for stop, times in bus_stop_arrivals.items():
                demand[stop] = []
                for t in times:
                    dest = np.random.choice(destinations)
                    demand[stop].append({'passenger_id': global_passenger_id, 'arrival_time': t,
                                         'destination': dest, 'served': False})
                    global_passenger_id += 1
            # Optionally add a fixed passenger at (0,0)
            if (0, 0) not in demand:
                demand[(0, 0)] = []
            demand[(0, 0)].append({'passenger_id': global_passenger_id, 'arrival_time': 27.6,
                                   'destination': 'D3', 'served': False})
            global_passenger_id += 1

            # --- Step 3: Vehicle Operation & Dispatch ---
            events = []
            pickup_times = {}

            # Define the vehicle class as before
            class Vehicle:
                def __init__(self, vehicle_id, start_time, start_stop, chosen_destination):
                    self.vehicle_id = vehicle_id
                    self.current_time = start_time
                    self.current_stop = start_stop
                    self.chosen_destination = chosen_destination
                    self.visited_stops = {start_stop}
                    self.current_load = 0
                    events.append({"Time": round(start_time, 2), "vehicle_id": vehicle_id,
                                   "stop_id": start_stop, "passenger_arrival": None,
                                   "passenger_pickup": start_time, "delay": None,
                                   "#riders": self.current_load, "destination": chosen_destination, "event": 'start'})
                def travel_to(self, next_stop):
                    travel_time = max(1, np.random.normal(5, 1))
                    self.current_time += travel_time
                    self.current_stop = next_stop
                    self.visited_stops.add(next_stop)
                    return travel_time
                def pickup_passengers(self, demand):
                    picked = []
                    max_wait_time = 20
                    if self.current_stop in demand:
                        for passenger in demand[self.current_stop]:
                            if (passenger['destination'] == self.chosen_destination and
                                not passenger['served'] and
                                passenger['arrival_time'] <= self.current_time):
                                waited_time = self.current_time - passenger['arrival_time']
                                if waited_time <= max_wait_time:
                                    passenger['served'] = True
                                    picked.append(passenger)
                                    pickup_times[passenger['passenger_id']] = self.current_time
                    self.current_load += len(picked)
                    return picked

            # Dispatch vehicles using the dispatch_interval parameter
            vehicles = []
            vehicle_count = 0
            current_time = 0.0
            sim_time = 300
            last_vehicle_start = 0
            last_vehicle_dest = None
            while current_time < sim_time:
                # Use the dispatch_interval from the outer loop to set interarrival times
                current_time += dispatch_interval
                if current_time >= sim_time:
                    break
                for i in range(3):
                    start_row = np.random.choice([0, 1])
                    start_stop = (start_row, 0)
                    chosen_dest = destinations[i]
                    if last_vehicle_dest == chosen_dest and current_time < last_vehicle_start + 10:
                        current_time = last_vehicle_start + 10
                    v = Vehicle(vehicle_id=vehicle_count, start_time=round(current_time, 2),
                                start_stop=start_stop, chosen_destination=chosen_dest)
                    vehicles.append(v)
                    vehicle_count += 1
                last_vehicle_start = current_time
                last_vehicle_dest = chosen_dest

            # --- Step 5: Execute Routes ---
            for v in vehicles:
                for col in range(0, 4):
                    candidate_stops = []
                    for row in [0, 1]:
                        s = (row, col)
                        if s not in v.visited_stops and s in demand:
                            if any(p['destination'] == v.chosen_destination and not p['served'] for p in demand[s]):
                                candidate_stops.append(s)
                    if candidate_stops:
                        dists = [abs(v.current_stop[0] - s[0]) + abs(v.current_stop[1] - s[1]) for s in candidate_stops]
                        chosen_stop = candidate_stops[np.argmin(dists)]
                        v.travel_to(chosen_stop)
                        picked = v.pickup_passengers(demand)
                        if picked:
                            delays = [round(v.current_time - p['arrival_time'], 2) for p in picked]
                            avg_delay = round(sum(delays) / len(delays), 2)
                            events.append({"Time": round(v.current_time, 2), "vehicle_id": v.vehicle_id,
                                           "stop_id": chosen_stop, "passenger_arrival": ','.join(str(p['arrival_time']) for p in picked),
                                           "passenger_pickup": round(v.current_time, 2), "delay": avg_delay,
                                           "#riders": v.current_load, "destination": v.chosen_destination, "event": 'pickup'})
                # Complete route by heading to drop-off
                v.travel_to((v.current_stop[0], 4))
                v.current_time += np.random.normal(10, 2)
                events.append({"Time": round(v.current_time, 2), "vehicle_id": v.vehicle_id,
                               "stop_id": "Destination " + v.chosen_destination, "passenger_dropoff": round(v.current_time, 2),
                               "passenger_pickup": None, "delay": None, "#riders": v.current_load,
                               "destination": v.chosen_destination, "event": 'dropoff'})
            
            # --- Step 6: Post-processing ---
            served_delays = []
            timeout_delays = []
            for stop, passengers in demand.items():
                for p in passengers:
                    if p['served']:
                        waited = pickup_times.get(p['passenger_id'], None)
                        if waited is not None:
                            waited = waited - p['arrival_time']
                            served_delays.append(min(waited, 20))
                    else:
                        timeout_delays.append(20)
            all_waits = served_delays + timeout_delays
            mean_wait = round(np.mean(all_waits), 2) if all_waits else 0
            std_wait = round(np.std(all_waits), 2) if all_waits else 0
            cv_wait = round(std_wait / mean_wait, 4) if mean_wait > 0 else None
            total_ridership = len(served_delays)
            timed_out_rider = len(timeout_delays)
            
            Ave_wt.append(mean_wait)
            CoV_wt.append(cv_wait)
            ridership.append(total_ridership)
            time_out_riders.append(timed_out_rider)
        
        # Save the results for current (demand, dispatch interval) combination
        np.savetxt(os.path.join(folder_name, f"{demand_level}_{dispatch_interval}_wt.txt"), Ave_wt)
        np.savetxt(os.path.join(folder_name, f"{demand_level}_{dispatch_interval}_cov.txt"), CoV_wt)
        np.savetxt(os.path.join(folder_name, f"{demand_level}_{dispatch_interval}_ride.txt"), ridership)
        print(f"DRT: Results for dispatch interval {dispatch_interval} and demand level {demand_level}:")
        print("  Average waiting time:", np.mean(Ave_wt))
        print("  Coefficient of variation:", np.mean(CoV_wt))
        print("  Total ridership:", np.mean(ridership))
        print("  Time out ridership:", np.mean(time_out_riders))
        print("====================================================")
