# Tube-transfer STL assets

These binary STL meshes reproduce the default `tubes_v2` primitives in
`mars/tubes/environment.py`, with 2 mm rack clearance. Coordinates are in
**millimetres**; use MuJoCo mesh `scale="0.001 0.001 0.001"` when importing.
The existing simulator continues to use its analytic collision primitives.

| File | Geometry | Bounds / dimensions (mm) |
|---|---|---|
| `rack.stl` | Same geometry for source and target racks, with three square holes | 60 × 160 × 60; hole width 18; pitch 45 |
| `tube_body.stl` | Solid capsule representing the simulated glass body | Diameter 16; z = −53 to 53 |
| `tube_cap.stl` | Solid cylindrical cap in tube-body coordinates | Diameter 19; z = 45 to 57 |
| `tube.stl` | Union of body and cap, with the enclosed upper hemisphere removed | Maximum diameter 19; z = −53 to 57 |

The rack origin is at the centre of its footprint on the table. Hole centres
are (0, 45), (0, 0), and (0, −45); the floor is 8 mm thick and the upper
plate occupies z = 52–60 mm. The tube origin matches the simulator's body
origin. Curved surfaces use 96 angular segments and 24 segments per hemisphere.

The body and cap intentionally overlap when loaded separately, matching the
simulation. Use `tube.stl` for a single closed surface. All four exported meshes
were checked for paired, consistently oriented edges and positive signed volume.
These are solid simulation shapes, not hollow laboratory or manufacturing models.
STL does not store the simulator's colours or material properties.
