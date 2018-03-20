import csv
import numpy as np
import sys

csv_file = "/home/timhu/dfd-pop/data/annos_csv/India_pov_pop_Feb3_density_class.csv"
lat = 22.2759570
lon = 73.1990113

def dist(lat1, lon1, lat2, lon2):
	return np.sqrt(np.square(lat2 - lat1) + np.square(lon2 - lon1))

with open(csv_file, "r") as f:
	village_reader = csv.reader(f)

	# record the village that is the closest
	min_dist = sys.float_info.max
	min_village = None
	headers_row = None

	count = 0
	for row in village_reader:
		count += 1
		if count == 1:
			headers_row = row
			continue
		vlat = float(row[7])
		vlon = float(row[6])
		distance = dist(vlat, vlon, lat, lon)
		if distance < min_dist:
			min_dist = distance
			min_village = row
		#if count == 100:
		#	break

	print("Results from searching:", count, "villages")
	print("Tried to find closest village to lat, lon: ", lat, lon)
	print("min dist:", min_dist)
	print("vlat, vlon:", min_village[7], min_village[6])
	print("state id:", min_village[2])
	print("headers:", headers_row)
	print("min village:", min_village)
