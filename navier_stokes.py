import sys
import gmsh
import numpy as np
# Import the functions for the wing profile
from naca import generate_naca4, profile_rotation

# Gmsh initialization
gmsh.initialize()
gmsh.model.add("Wind Tunnel")

# Parameters of the rectangular domain (this big to avoid an interference wing-borders)
x_min, x_max = -5.0, 15.0
y_min, y_max = -5.0, 5.0
lc_far = 0.8   # Triangles distant from the wing
lc_near = 0.015 # Triangles very close to the wing

# Four points of the rectangle
p1 = gmsh.model.geo.addPoint(x_min, y_min, 0, lc_far)
p2 = gmsh.model.geo.addPoint(x_max, y_min, 0, lc_far)
p3 = gmsh.model.geo.addPoint(x_max, y_max, 0, lc_far)
p4 = gmsh.model.geo.addPoint(x_min, y_max, 0, lc_far)

# Four lines of the rectangle
l1 = gmsh.model.geo.addLine(p1, p2) # Bottom
l2 = gmsh.model.geo.addLine(p2, p3) # Outlet
l3 = gmsh.model.geo.addLine(p3, p4) # Top
l4 = gmsh.model.geo.addLine(p4, p1) # Inlet

x_w, y_w = generate_naca4(naca_code="2412", n_points=40)
x_w, y_w = profile_rotation(x_w, y_w, angle_deg=8)

# Load the wing points in Gmsh
wing_point_ids = []
for i in range(len(x_w)):
    p_id = gmsh.model.geo.addPoint(x_w[i], y_w[i], 0, lc_near)
    wing_point_ids.append(p_id)

# Per chiudere la curva della spline, aggiungiamo l'ID del primo punto alla fine
wing_curve_points = wing_point_ids + [wing_point_ids[0]]

# --- FASE 4: CREAZIONE DELLE CURVE E DELLE SUPERFICI ---
# Creiamo il profilo dell'ala usando una spline chiusa
wing_curve = gmsh.model.geo.addSpline(wing_curve_points)
rect_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
wing_loop = gmsh.model.geo.addCurveLoop([wing_curve])
# Superficie fluida con il "buco" dell'ala al centro
fluid_surface = gmsh.model.geo.addPlaneSurface([rect_loop, wing_loop])
# Sincronizza il motore geometrico CAD di Gmsh
gmsh.model.geo.synchronize()
# --- FASE 5: GRUPPI FISICI (Etichette per le condizioni al contorno in FEniCS) ---
gmsh.model.addPhysicalGroup(1, [l4], 1)
gmsh.model.setPhysicalName(1, 1, "Inlet")
gmsh.model.addPhysicalGroup(1, [l2], 2)
gmsh.model.setPhysicalName(1, 2, "Outlet")
gmsh.model.addPhysicalGroup(1, [l1, l3], 3)
gmsh.model.setPhysicalName(1, 3, "Walls")
gmsh.model.addPhysicalGroup(1, [wing_curve], 4)
gmsh.model.setPhysicalName(1, 4, "Airfoil")
gmsh.model.addPhysicalGroup(2, [fluid_surface], 5)
gmsh.model.setPhysicalName(2, 5, "Fluid_Domain")
# --- FASE 6: GENERAZIONE DELLA MESH E SALVATAGGIO ---
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.model.mesh.generate(2)
gmsh.write(output_file)
if 'close' not in sys.argv:
    gmsh.fltk.run()
gmsh.finalize()

if __name__ == "__main__":
    # Esecuzione standard con parametri di default
    genera_mesh_ala()