import csv
import numpy as np
import sys

csv_file = "/home/timhu/dfd-pop/data/annos_csv/state24_jpgpaths_density_nolaps_12k_Mar6.csv"

# the state id to match
state_id = 24

output_path = "gujarat_villages.npy"

filtered = []
with open(csv_file, "r") as f:
	village_reader = csv.reader(f)

	count = 0
	for row in village_reader:
		count += 1
		if count == 1:
			headers_row = row
			print(headers_row)
			continue
		sid = int(row[2])
		if sid != state_id:
			continue
		filtered.append(row)
		if count < 5:
			print(row)
	print("Found", len(filtered), "villages within state", state_id)

#np.save(output_path, filtered)

