# Pydrake Kuka Sandbox and Utilities

[![Build Status](https://travis-ci.org/gizatt/pydrake_kuka.svg?branch=master)](https://travis-ci.org/gizatt/pydrake_kuka)

## HOW TO RUN WITH DOCKER

1) Run `python docker_build.py`. This builds a Docker image named *pydrake_kuka*, built against (by default) the latest Drake binaries.
2) Run `python docker_run.py`. This runs the above Docker image, giving you a terminal inside of the Docker container from which you can use the code. The contents of this repo are available in the directory `/pydrake_kuka`, and it's kept in sync with the files on the host computer, so feel free to tweak things in a text editor outside of the container. Things you can try:
  - Run `cd /pydrake_kuka && ./run_tests.sh`, and point a browser to <127.0.0.1:7000/static/> to watch the robot go through an automated cutting test.
  - Break the terminal into multiple prompts with your favorite terminal multiplexer (e.g. screen, tmux), run `meshcat-server` in one (and then go to its specified URL to see the visualization), and run `python /pydrake_kuka/kuka_cutting_sim.py -N 2 -T 5.` to run the cutting sim yourself.

 To run the Kuka simulation, first run `meshcat-server` in a new terminal. It should report a web-url -- something like `127.0.0.1:7000/static/`. Open that in a browser -- this is your 3D viewer. Then run `python kuka_pydrake_sim.py` and you should see the arm spawn in the viewer before doing some movements.
