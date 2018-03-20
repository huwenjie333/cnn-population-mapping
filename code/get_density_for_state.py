import csv
import numpy as np
import sys

def get_state_density(sid):
	csv_file = "/home/timhu/dfd-pop/data/India_pov_pop_Feb4_density_class.csv"

	total_area = 0.0
	total_pop = 0.0
	with open(csv_file, "r") as f:
		village_reader = csv.reader(f)

		count = 0
		for row in village_reader:
			count += 1
			if count == 1:
				continue
			curr_sid = int(row[2])
			if curr_sid != sid:
				continue

			density = float(row[8])
			area = float(row[5])

			total_area += area
			total_pop += density*area

	overall_density = total_pop / float(total_area)

	return overall_density


