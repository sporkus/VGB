## What 

Instead of taking just two samples (and one more for validation) of the bed mesh for modeling, this takes repated measurements of the bed mesh during the entire duration of bed heating. 

Depending on the printer and mesh parameters, each bed mesh measurement may take several minutes. In the beginning of the measurement period, the frame temperature can change quite a bit between the first probe point to the last. This script will interpolate the intermediate temperatures for each probe point using the starting and ending temperature.

Example:
180 3x3 meshs are mesaured. For each mesh, a 3x3 grid of temperatures will be generated. For each of the 9 probe points, we have 180 pairs of Z height and temperature for modeling.

## Why

* A lot more data. The machine was already on, might as well collect data.
* Because we are interpolating temperature for each mesh, we can take our time and make high resolution meshes. 


## Scripts

* `measure_mesh_meshes.py`: this is a modification of AlchemyEngine's [measure thermal behavior](https://github.com/alchemyEngine/measure_thermal_behavior/blob/main/measure_thermal_behavior.py). It just collects data repeatedly during the entire period. User can cancel the script early and results will still be saved. By default, it will process the collected data afterwards with `generate_meshes.py`.

* `generate_meshes.py`: this does the temperature interpolation, sample fitting, and mesh generation. By default, meshes are generated in 0.1C steps ranging between 1C above/below the measurements.


## Usage

* Set global variables in `measure_mesh_meshes.py` and run it with `python3 measure_mesh_meshes.py`. Measurements will be saved to the `data` folder and generated meshes formatted for `printer.cfg` will be saved to the current folder.

## TODO

* Needs better visualization on results. Can probably use the coefficients on each point to describe deflection.
* Script for testing modeled meshes. Can't test your model with it's own samples.
* fix package requirements