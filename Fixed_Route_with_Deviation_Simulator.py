import numpy as np
import pandas as pd
import os

# Utility function: vertical vs horizontal travel time
def compute_travel_time(from_stop, to_stop):
    if from_stop[0] == to_stop[0]:  # same row → horizontal
        return max(1, np.random.normal(5, 1))
    elif from_stop[1] == to_stop[1]:  # same column → vertical
        return max(1, np.random.normal(4, 1))
    else:
        return max(1, np.random.normal(5, 0.5))

# Passenger demand simulator
class StopsDemandSimulator:
    def __init__(self, num_stops=(2, 5), sim_time=300, network_rate=60, seed=None):
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
            chosen_line = np.random.choice(range(self.num_stops[0]), p=[0.7, 0.3])
            chosen_stop = np.random.randint(0, self.num_stops[1])
            self.bus_stops[(chosen_line, chosen_stop)].append(round(current_time, 2))

        effective_start = 30
        effective_end = self.sim_time - 30
        for key in self.bus_stops:
            filtered = [t for t in self.bus_stops[key] if effective_start <= t <= effective_end]
            self.bus_stops[key] = sorted(filtered)
        return self.bus_stops

# Grid bus class
class GridBusWithTimeout:
    def __init__(self, bus_id, fixed_route, gen_time=0, deviation_paths=None, slack_time=50):
        self.bus_id = bus_id
        self.fixed_route = fixed_route
        self.route = fixed_route.copy()
        self.gen_time = gen_time
        self.travel_times = [compute_travel_time(self.route[i], self.route[i + 1]) for i in range(len(self.route) - 1)]
        self.current_stop_index = 0
        self.deviation_paths = deviation_paths or {}
        self.slack_time = slack_time
        self.in_deviation = False
        self.deviation_index = 0
        self.deviation_route = []
        self.completed_deviation = False
        self.deviation_travel_times = []
        self.total_dwell_time = 0

    def apply_dwell_time(self):
        self.total_dwell_time += np.random.uniform(0.3, 0.5)

    def next_arrival_time(self):
        if self.in_deviation and self.deviation_index < len(self.deviation_route) - 1:
            return self.gen_time + sum(self.travel_times[:self.current_stop_index + 1]) + \
                   sum(self.deviation_travel_times[:self.deviation_index + 1]) + self.total_dwell_time
        elif self.current_stop_index < len(self.travel_times):
            return self.gen_time + sum(self.travel_times[:self.current_stop_index + 1]) + self.total_dwell_time
        else:
            return None

    def update(self, current_time, station_passengers):
        if self.in_deviation:
            next_arrival = self.next_arrival_time()
            if current_time >= next_arrival:
                self.deviation_index += 1
                if self.deviation_index >= len(self.deviation_route):
                    self.in_deviation = False
                    self.completed_deviation = True
                return self.deviation_route[self.deviation_index - 1], next_arrival
            return None, None

        elif self.current_stop_index < len(self.fixed_route) - 1:
            next_arrival = self.next_arrival_time()
            if current_time >= next_arrival:
                stop = self.fixed_route[self.current_stop_index + 1]

                # Check for possible deviation
                if stop in self.deviation_paths:
                    deviation_route = self.deviation_paths[stop]
                    flex_stop = deviation_route[1]

                    if flex_stop in station_passengers and station_passengers[flex_stop]:
                        deviation_time = sum(compute_travel_time(deviation_route[i], deviation_route[i+1]) for i in range(len(deviation_route) - 1))
                        if deviation_time <= self.slack_time:
                            self.start_deviation(deviation_route)
                            return self.deviation_route[0], current_time

                self.current_stop_index += 1
                return stop, next_arrival
            return None, None

        elif self.current_stop_index < len(self.fixed_route) - 1:
            next_arrival = self.next_arrival_time()
            if current_time >= next_arrival:
                stop = self.fixed_route[self.current_stop_index + 1]
                self.current_stop_index += 1
                for dev_list in self.deviation_paths.values():
                    flex_stop = dev_list[1]
                    if flex_stop in station_passengers:
                        station_passengers[flex_stop] = [pt for pt in station_passengers[flex_stop] if current_time - pt <= max_wait_time]
                return stop, next_arrival
        return None, None

    def start_deviation(self, deviation_route):
        self.in_deviation = True
        self.deviation_index = 0
        self.deviation_route = deviation_route
        self.deviation_travel_times = [compute_travel_time(deviation_route[i], deviation_route[i+1])
                                       for i in range(len(deviation_route) - 1)]

# === Simulation parameters ===
fixed_route = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
deviation_paths = {
    (0, 1): [(0, 0),(1, 1), (0, 1)],
    (0, 3): [(0, 2),(1, 3), (0, 3)],
}
max_wait_time = 20

# Scenario parameters
demand_ls = [60, 120, 180, 240]
headway_ls = [5, 10, 15, 20]
num_replication = 10
folder_name = "./results/FRD/"
os.makedirs(folder_name, exist_ok=True)

for demand_level in demand_ls:
    for headway in headway_ls:
        Ave_wt, CoV_wt, ridership, timed_out_rider_ls = [], [], [], []

        for rep in range(num_replication):
            simulator = StopsDemandSimulator(num_stops=(2, 5), sim_time=300, network_rate=demand_level, seed=rep)
            bus_stop_arrivals = simulator.simulate_network()
            station_passengers = bus_stop_arrivals.copy()
            sim_start, sim_end = 30, 270
            active_buses, service_records, timeout_records = [], [], []

            for current_time in range(sim_start, sim_end + 1):
                if (current_time - sim_start) % headway == 0:
                    bus_id = len(active_buses) + 1
                    new_bus = GridBusWithTimeout(bus_id=bus_id, fixed_route=fixed_route,
                                                 gen_time=current_time, deviation_paths=deviation_paths, slack_time=8)
                    active_buses.append(new_bus)

                for bus in active_buses:
                    arrival_stop, arrival_time = bus.update(current_time, station_passengers)
                    if arrival_stop is not None:
                        waiting_list = station_passengers.get(arrival_stop, [])
                        timed_out = [pt for pt in waiting_list if pt <= arrival_time and (arrival_time - pt) > max_wait_time]
                        for pt in timed_out:
                            timeout_records.append({"stop ID": arrival_stop, "delay": round(min(arrival_time - pt, max_wait_time), 2)})

                        served_passengers = [pt for pt in waiting_list if pt <= arrival_time and (arrival_time - pt) <= max_wait_time]
                        if served_passengers:
                            delays = [round(arrival_time - pt, 2) for pt in served_passengers]
                            avg_delay = round(sum(delays) / len(delays), 2)
                            bus.apply_dwell_time()
                            service_records.append({"time": arrival_time, "bus ID": bus.bus_id, "stop ID": arrival_stop,
                                                     "passengers arrivals": served_passengers, "delay": delays,
                                                     "average delay": avg_delay, "# riders": len(served_passengers)})
                        station_passengers[arrival_stop] = [pt for pt in waiting_list if pt > arrival_time]

            served_waits = [min(max_wait_time, d) for record in service_records for d in record["delay"]]
            timeout_waits = [entry["delay"] for entry in timeout_records]
            all_waits = served_waits + timeout_waits

            mean_wait = round(np.mean(all_waits), 2) if all_waits else 0
            std_wait = round(np.std(all_waits), 2) if all_waits else 0
            cv_wait = round(std_wait / mean_wait, 4) if mean_wait > 0 else None
            total_ridership = sum(record["# riders"] for record in service_records)

            Ave_wt.append(mean_wait)
            CoV_wt.append(cv_wait)
            ridership.append(total_ridership)

            # Revised timeout calculation: includes any unserved passengers still at stops after simulation ends
            unserved_final = sum(len(q) for q in station_passengers.values())
            total_timeout = len(timeout_waits) + unserved_final
            all_served_pts = set(pt for record in service_records for pt in record["passengers arrivals"])
            all_timed_out_pts = set(entry["delay"] for entry in timeout_records)  # assumes 'delay' uniquely identifies pt
            tracked_stops = [(0,0), (0,1), (0,2), (0,3), (0,4), (1,1), (1,3)]
            all_observed_pts = set(pt for stop, pts in bus_stop_arrivals.items() if stop in tracked_stops for pt in pts)
            unserved_pts = all_observed_pts - all_served_pts - all_timed_out_pts
            timed_out_rider_ls.append(len(all_timed_out_pts) + len(unserved_pts))

        np.savetxt(os.path.join(folder_name, f"{demand_level}_{headway}_wt.txt"), Ave_wt)
        np.savetxt(os.path.join(folder_name, f"{demand_level}_{headway}_cov.txt"), CoV_wt)
        np.savetxt(os.path.join(folder_name, f"{demand_level}_{headway}_ride.txt"), ridership)
        np.savetxt(os.path.join(folder_name, f"{demand_level}_{headway}_timeout.txt"), timed_out_rider_ls)

        print(f"FRD: Results for headway {headway} and demand level {demand_level}:")
        print("  Average waiting time:", np.mean(Ave_wt))
        print("  Coefficient of variation:", np.mean(CoV_wt))
        print("  Total ridership:", np.mean(ridership))
        print("  Total timeout ridership:", np.mean(timed_out_rider_ls))
        print("====================================================")
