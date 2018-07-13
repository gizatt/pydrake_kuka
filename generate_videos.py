import os
import random

# Fullscreen meshlab on right monitor for this to work
for k in range(100, 200):
	n_objects = random.randint(5, 10)
	os.system("python kuka_pydrake_sim.py -T 60 --seed %d --hacky_save_video -N %d" % (k, n_objects))
