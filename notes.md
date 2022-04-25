## Klipper mesh format

Use a 3x3 mesh with RRI=3 to confirm the ordering of the mesh points. Klipper laid out probe measurements as the grid in x,y, not the probing sequence.

#### Printer.cfg

* Points are laid out as the grid sampled, with (0, 0) on the top left

```
#*# [bed_mesh 3x3 example (rrl=3)]
#*# version = 1
#*# points =
#*# 	  0.017500, -0.040000, -0.022500
#*# 	  -0.032500, -0.082500, 0.000000
#*# 	  0.027500, -0.045000, 0.045000
#*# tension = 0.2
#*# min_x = 10.0
#*# algo = lagrange
#*# y_count = 3
#*# mesh_y_pps = 2
#*# min_y = 30.0
#*# x_count = 3
#*# max_y = 290.0
#*# mesh_x_pps = 2
#*# max_x = 290.0
```

#### Moonraker response

* Order is identical to printer.cfg 
```
{'points': [[0.0175, -0.04, -0.0225],
  [-0.0325, -0.0825, 0.0],
  [0.0275, -0.045, 0.045]],
 'mesh_params': {'tension': 0.2,
  'mesh_x_pps': 2,
  'algo': 'lagrange',
  'min_x': 10.0,
  'min_y': 30.0,
  'y_count': 3,
  'mesh_y_pps': 2,
  'x_count': 3,
  'max_x': 290.0,
  'max_y': 290.0}}
  ```