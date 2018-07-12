# Pydrake Kuka Sandbox and Utilities

[![Build Status](https://travis-ci.org/gizatt/pydrake_kuka.svg?branch=master)](https://travis-ci.org/gizatt/pydrake_kuka)

## PREREQS

```
apt-get install openscad
pip install meshcat numpy matplotlib trimesh
```

```
git clone https://github.com/RussTedrake/underactuated ~/underactuated
export PYTHONPATH=~/underactuated/src:$PYTHONPATH
```

To run the Kuka simulation, first run `meshcat-server` in a new terminal. It should report a web-url -- something like `127.0.0.1:7000/static/`. Open that in a browser -- this is your 3D viewer. Then run `python kuka_pydrake_sim.py` and you should see the arm spawn in the viewer before doing some movements.
