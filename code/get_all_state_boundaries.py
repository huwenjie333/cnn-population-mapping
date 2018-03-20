import csv
import numpy as np
import sys

def get_state_boundaries():
	csv_file = "/home/timhu/dfd-pop/data/India_pov_pop_Feb4_density_class.csv"

	state_boundaries = dict()

	with open(csv_file, "r") as f:
		village_reader = csv.reader(f)

		count = 0
		for row in village_reader:
			count += 1
			if count == 1:
				continue
			vlat = float(row[7])
			vlon = float(row[6])
			sid = int(row[2])
        
			if sid not in state_boundaries:
				state_boundaries[sid] = {"lat_min": sys.float_info.max, "lat_max": sys.float_info.min, "lon_min": sys.float_info.max, "lon_max": sys.float_info.min}
        
			if vlat < state_boundaries[sid]["lat_min"]:
				state_boundaries[sid]["lat_min"] = vlat
			elif vlat > state_boundaries[sid]["lat_max"]:
				state_boundaries[sid]["lat_max"] = vlat
            
			if vlon < state_boundaries[sid]["lon_min"]:
				state_boundaries[sid]["lon_min"] = vlon
			elif vlon > state_boundaries[sid]["lon_max"]:
				state_boundaries[sid]["lon_max"] = vlon

	return state_boundaries


