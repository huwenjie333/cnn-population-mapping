import csv
import numpy as np
import sys

csv_file = "/home/timhu/dfd-pop/data/annos_csv/India_pov_pop_Feb3_density_class.csv"

# the coordinates to search around
lat = 22.2759570
lon = 73.1990113
# the radius to search (in degrees)
radius = 0.5
# the state id to match
state_id = 24

output_path = "filtered_villages.npy"

def dist(lat1, lon1, lat2, lon2):
	return np.sqrt(np.square(lat2 - lat1) + np.square(lon2 - lon1))

filtered = []
with open(csv_file, "r") as f:
	village_reader = csv.reader(f)

	count = 0
	for row in village_reader:
		count += 1
		if count == 1:
			headers_row = row
			continue
		vlat = float(row[7])
		vlon = float(row[6])
		sid = int(row[2])
		if sid != state_id:
			continue
		distance = dist(vlat, vlon, lat, lon)
		if distance < radius:
			filtered.append(row)
		#if count == 100:
		#	break

	print("Found", len(filtered), "villages within radius from searching", count, "villages")

np.save(output_path, filtered)

