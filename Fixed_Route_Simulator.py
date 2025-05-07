import numpy as np
import pandas as pd
import random
import os
# demand_level = 180
# headway_line0 = 15
# num_replication = 10
# Ave_wt = []
# CoV_wt = []
# rider = []
class StopsDemandSimulator:
    def __init__(self, num_stops=(2, 5), sim_time=180, network_rate=60, seed=None):
        # if seed is not None:
        #     np.random.seed(seed)
        self.num_stops = num_stops
        self.sim_time = sim_time
        self.network_rate = network_rate  
        self.bus_stops = {(i, j): [] for i in range(0, num_stops[0]) for j in range(num_stops[1])}

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

    def __repr__(self):
        return str(self.bus_stops)

# simulator = StopsDemandSimulator(num_stops=(2, 5), sim_time=300, network_rate=demand_level, seed=None)
# bus_stop_arrivals = simulator.simulate_network()

class Bus:
    def __init__(self, bus_id, route, gen_time=0, travel_times=None):
        self.bus_id = bus_id
        self.route = route
        self.gen_time = gen_time
        self.travel_times = travel_times
        self.current_stop_index = 0
        self.total_dwell_time = 0  # Track cumulative dwell time

    def next_arrival_time(self):
        if self.current_stop_index < len(self.travel_times):
            return self.gen_time + sum(self.travel_times[:self.current_stop_index+1]) + self.total_dwell_time
        else:
            return None

    def update(self, current_time):
        next_arrival = self.next_arrival_time()
        if next_arrival is not None and current_time >= next_arrival:
            arrival_stop = self.route[self.current_stop_index+1]
            self.current_stop_index += 1
            return arrival_stop, next_arrival
        return None, None

    def add_dwell_time(self):
        additional_dwell = np.random.uniform(0.3, 0.5)
        self.total_dwell_time += additional_dwell
        return additional_dwell

def generate_travel_times(n_segments):
    return [np.random.normal(5, 1) for _ in range(n_segments)]


## Control Parameters
demand_ls = [60,120,180,240]
hd_ls = [5,10,15,20]
# demand_level = 180
# headway_line0 = 15
num_replication = 10

folder_name = "./results/FR/"
### Main Code
for demand_level in demand_ls:
    for headway_line0 in hd_ls:
        Ave_wt = []
        CoV_wt = []
        rider = []
        time_out_rider_ls = []
        for i in range(num_replication):
            simulator = StopsDemandSimulator(num_stops=(2, 5), sim_time=300, network_rate=demand_level, seed=None)
            bus_stop_arrivals = simulator.simulate_network()
            sim_start = 30
            sim_end = 270
            active_buses = []
            service_records = []
            timeout_delays = []
            station_passengers = bus_stop_arrivals.copy()
            line0_bus_count = 0
            line1_bus_count = 0
            max_wait_time = 20
            time_out_rider = 0
            for current_time in range(sim_start, sim_end + 1):
                if (current_time - sim_start) % headway_line0 == 0:
                    line0_bus_count += 1
                    bus_id = f"{line0_bus_count:02d}"
                    route = [(0, j) for j in range(5)]
                    travel_times = generate_travel_times(len(route) - 1)
                    new_bus = Bus(bus_id=bus_id, route=route, gen_time=current_time, travel_times=travel_times)
                    active_buses.append(new_bus)

                for bus in active_buses:
                    arrival_stop, arrival_time = bus.update(current_time)
                    if arrival_stop is not None:
                        waiting_list = station_passengers.get(arrival_stop, [])
                        served_passengers = [pt for pt in waiting_list if pt <= arrival_time and (arrival_time - pt) <= max_wait_time]
                        timed_out_passengers = [pt for pt in waiting_list if pt <= arrival_time and (arrival_time - pt) > max_wait_time]
                        time_out_rider += len(timed_out_passengers)

                        timeout_delays.extend([max_wait_time for _ in timed_out_passengers])

                        if served_passengers:
                            delays = [round(arrival_time - pt, 2) for pt in served_passengers]
                            avg_delay = round(sum(delays)/len(delays), 2) if delays else 0

                            dwell_time = bus.add_dwell_time()

                            record = {
                                "time": arrival_time,
                                "bus ID": bus.bus_id,
                                "stop ID": arrival_stop,
                                "passengers arrivals": served_passengers,
                                "delay": delays,
                                "average delay": avg_delay,
                                "# riders": len(served_passengers),
                                "dwell time": round(dwell_time, 2)
                            }
                            service_records.append(record)

                        remaining = [pt for pt in waiting_list if pt > arrival_time]
                        station_passengers[arrival_stop] = remaining
            time_out_rider_ls.append(time_out_rider)
            # === Post-processing metrics ===
            all_delays = [delay for record in service_records for delay in record['delay']]
            capped_delays = [min(20, d) for d in all_delays]
            all_waits = capped_delays + timeout_delays

            mean_wait = round(np.mean(all_waits), 2)
            std_wait = round(np.std(all_waits), 2)
            cv_wait = round(std_wait / mean_wait, 4) if mean_wait > 0 else None
            total_ridership = len(capped_delays)
            Ave_wt.append(mean_wait)
            CoV_wt.append(cv_wait)
            rider.append(total_ridership)

        np.savetxt(os.path.join(folder_name,f"{demand_level}_{headway_line0}_wt.txt"),Ave_wt)
        np.savetxt(os.path.join(folder_name,f"{demand_level}_{headway_line0}_cov.txt"),CoV_wt)
        np.savetxt(os.path.join(folder_name,f"{demand_level}_{headway_line0}_ride.txt"),rider)
        print(f"Results of headway:{headway_line0}, demand_level:{demand_level} for FR")
        print("Average waiting time:", np.mean(Ave_wt))
        print("Coefficient of variation:", np.mean(CoV_wt))
        print("Total ridership:", np.mean(rider))
        print("Total timeout ridership",np.mean(time_out_rider_ls))
        print("====================================================")

# DataFrame for further analysis if needed
# df_service = pd.DataFrame(service_records)
