import numpy as np
import time
import meshio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

def load_mesh_data(mesh_file="prova.msh"):
    mesh = meshio.read(mesh_file)
    points = mesh.points[:, :2] 
    triangles = mesh.cells_dict["triangle"]
    num_cells = len(triangles)
    
    cell_centers = np.zeros((num_cells, 2))
    cell_volumes = np.zeros(num_cells)
    
    for i, tri in enumerate(triangles):
        p0, p1, p2 = points[tri[0]], points[tri[1]], points[tri[2]]
        cell_centers[i] = (p0 + p1 + p2) / 3.0
        cell_volumes[i] = 0.5 * np.abs(p0[0]*(p1[1] - p2[1]) + p1[0]*(p2[1] - p0[1]) + p2[0]*(p0[1] - p1[1]))

    lines = mesh.cells_dict["line"]
    line_data = mesh.cell_data_dict["gmsh:physical"]["line"]
    
    boundaries = {
        "Inlet": lines[line_data == 1],
        "Outlet": lines[line_data == 2],
        "Walls": lines[line_data == 3],
        "Airfoil": lines[line_data == 4]
    }
    return points, triangles, cell_centers, cell_volumes, boundaries

def compute_convective_flux(UL, UR, normal, area, gamma=1.4):
    # INPUT: UL e UR di forma (N, 4), normal di forma (N, 2), area di forma (N,)
    rho_L = UL[:, 0]                                                   
    u_L = UL[:, 1] / rho_L                                             
    v_L = UL[:, 2] / rho_L                                             
    E_L = UL[:, 3] / rho_L                                             
    p_L = (gamma - 1.0) * (UL[:, 3] - 0.5 * rho_L * (u_L**2 + v_L**2)) 
    vn_L = u_L * normal[:, 0] + v_L * normal[:, 1]                        
    H_L = E_L + p_L / rho_L                                         

    rho_R = UR[:, 0]
    u_R = UR[:, 1] / rho_R
    v_R = UR[:, 2] / rho_R
    E_R = UR[:, 3] / rho_R
    p_R = (gamma - 1.0) * (UR[:, 3] - 0.5 * rho_R * (u_R**2 + v_R**2))
    vn_R = u_R * normal[:, 0] + v_R * normal[:, 1]
    H_R = E_R + p_R / rho_R

    Flux_L = np.zeros_like(UL)
    Flux_L[:, 0] = rho_L * vn_L
    Flux_L[:, 1] = rho_L * u_L * vn_L + p_L * normal[:, 0]
    Flux_L[:, 2] = rho_L * v_L * vn_L + p_L * normal[:, 1]
    Flux_L[:, 3] = rho_L * H_L * vn_L

    Flux_R = np.zeros_like(UR)
    Flux_R[:, 0] = rho_R * vn_R
    Flux_R[:, 1] = rho_R * u_R * vn_R + p_R * normal[:, 0]
    Flux_R[:, 2] = rho_R * v_R * vn_R + p_R * normal[:, 1]
    Flux_R[:, 3] = rho_R * H_R * vn_R

    c_L = np.sqrt(gamma * np.maximum(p_L, 1e-5) / np.maximum(rho_L, 1e-5))
    c_R = np.sqrt(gamma * np.maximum(p_R, 1e-5) / np.maximum(rho_R, 1e-5))
    max_lambda = np.maximum(np.abs(vn_L) + c_L, np.abs(vn_R) + c_R)

    return 0.5 * (Flux_L + Flux_R) * area[:, None] - 0.5 * max_lambda[:, None] * (UR - UL) * area[:, None]

def compute_viscous_flux(UL, UR, dist, normal, area, mu=1.8e-5):
    u_L, v_L = UL[:, 1]/UL[:, 0], UL[:, 2]/UL[:, 0]
    u_R, v_R = UR[:, 1]/UR[:, 0], UR[:, 2]/UR[:, 0]
    max_dist = np.maximum(dist, 1e-6)
    du_dn = (u_R - u_L) / max_dist
    dv_dn = (v_R - v_L) / max_dist
    tau_x = mu * du_dn
    tau_y = mu * dv_dn
    
    Flux_V = np.zeros_like(UL)
    Flux_V[:, 1] = tau_x * normal[:, 0]
    Flux_V[:, 2] = tau_y * normal[:, 1]
    Flux_V[:, 3] = (tau_x * 0.5 * (u_L + u_R) + tau_y * 0.5 * (v_L + v_R))
    
    return Flux_V * area[:, None]

def solve_compressible_fvm(mesh_file="prova.msh", T=2.0, dt=0.0002, u_inf=250.0, p_inf=101325.0, rho_inf=1.225, CFL=0.5,
                          artificial_viscosity=0.0, log_interval=200):
    points, triangles, cell_centers, cell_volumes, boundaries = load_mesh_data(mesh_file)
    num_cells = len(triangles)
    gamma = 1.4
    mu = 1.8e-5 
    
    E_inf = p_inf / (gamma - 1.0) + 0.5 * rho_inf * u_inf**2
    U = np.zeros((num_cells, 4))
    U[:, 0] = rho_inf
    U[:, 1] = rho_inf * u_inf
    U[:, 2] = 0.0
    U[:, 3] = E_inf

    edge_to_cells = {}
    for i, tri in enumerate(triangles):
        edges = [(min(tri[0], tri[1]), max(tri[0], tri[1])), (min(tri[1], tri[2]), max(tri[1], tri[2])), (min(tri[2], tri[0]), max(tri[2], tri[0]))]
        for edge in edges:
            if edge not in edge_to_cells: edge_to_cells[edge] = []
            edge_to_cells[edge].append(i)
            
    internal_faces = []
    for edge, cells in edge_to_cells.items():
        if len(cells) == 2: 
            c_left, c_right = cells[0], cells[1]
            p1, p2 = points[edge[0]], points[edge[1]]
            length = np.linalg.norm(p2 - p1)
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            normal = np.array([dy, -dx])
            normal /= np.linalg.norm(normal)
            if np.dot(normal, cell_centers[c_right] - cell_centers[c_left]) < 0: normal = -normal 
            dist = np.linalg.norm(cell_centers[c_right] - cell_centers[c_left])
            internal_faces.append((c_left, c_right, normal, length, dist))

    if internal_faces:
        h_min = min(f[4] for f in internal_faces)
    else:
        h_min = 1.0

    boundary_mappings = {key: [] for key in boundaries.keys()}
    for b_name, lines in boundaries.items():
        for line in lines:
            edge = (min(line[0], line[1]), max(line[0], line[1]))
            if edge in edge_to_cells:
                c_int = edge_to_cells[edge][0]
                p1, p2 = points[line[0]], points[line[1]]
                area = np.linalg.norm(p2 - p1)
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                normal = np.array([dy, -dx])
                normal /= np.linalg.norm(normal)
                if np.dot(normal, ((p1 + p2) / 2.0) - cell_centers[c_int]) < 0: normal = -normal
                dist = np.linalg.norm(cell_centers[c_int] - ((p1 + p2) / 2.0))
                boundary_mappings[b_name].append((c_int, normal, area, dist))

    # --- MATRICIZZAZIONE DATI PER VETTORIZZAZIONE ---
    if len(internal_faces) > 0:
        if_c_L = np.array([f[0] for f in internal_faces], dtype=int)
        if_c_R = np.array([f[1] for f in internal_faces], dtype=int)
        if_normal = np.array([f[2] for f in internal_faces])
        if_area = np.array([f[3] for f in internal_faces])
        if_dist = np.array([f[4] for f in internal_faces])
    else:
        if_c_L = if_c_R = if_normal = if_area = if_dist = np.array([])

    b_arrays = {}
    for b_name, mappings in boundary_mappings.items():
        if len(mappings) > 0:
            b_arrays[b_name] = {
                "c_int": np.array([m[0] for m in mappings], dtype=int),
                "normal": np.array([m[1] for m in mappings]),
                "area": np.array([m[2] for m in mappings]),
                "dist": np.array([m[3] for m in mappings])
            }
        else:
            b_arrays[b_name] = None

    def compute_residual(U_local):
        Rloc = np.zeros_like(U_local)

        if len(if_c_L) > 0:
            F_total = compute_convective_flux(U_local[if_c_L], U_local[if_c_R], if_normal, if_area, gamma) - \
                      compute_viscous_flux(U_local[if_c_L], U_local[if_c_R], if_dist, if_normal, if_area, mu)
            
            np.add.at(Rloc, if_c_L, -F_total)
            np.add.at(Rloc, if_c_R, F_total)

            if artificial_viscosity > 0.0:
                diff = artificial_viscosity * (U_local[if_c_R] - U_local[if_c_L]) * if_area[:, None] / np.maximum(if_dist[:, None], 1e-12)
                np.add.at(Rloc, if_c_L, diff)
                np.add.at(Rloc, if_c_R, -diff)

        b_airfoil = b_arrays.get("Airfoil")
        if b_airfoil is not None:
            c_int, normal, area, dist = b_airfoil["c_int"], b_airfoil["normal"], b_airfoil["area"], b_airfoil["dist"]
            U_local_c = U_local[c_int]
            U_ghost = U_local_c.copy()
            U_ghost[:, 1] = -U_local_c[:, 1]
            U_ghost[:, 2] = -U_local_c[:, 2]
            F_b = compute_convective_flux(U_local_c, U_ghost, normal, area, gamma) - \
                  compute_viscous_flux(U_local_c, U_ghost, dist, normal, area, mu)
            np.add.at(Rloc, c_int, -F_b)

        b_inlet = b_arrays.get("Inlet")
        if b_inlet is not None:
            c_int, normal, area = b_inlet["c_int"], b_inlet["normal"], b_inlet["area"]
            U_local_c = U_local[c_int]
            U_ghost = np.tile(np.array([rho_inf, rho_inf * u_inf, 0.0, E_inf]), (len(c_int), 1))
            F_b = compute_convective_flux(U_local_c, U_ghost, normal, area, gamma)
            np.add.at(Rloc, c_int, -F_b)

        b_outlet = b_arrays.get("Outlet")
        if b_outlet is not None:
            c_int, normal, area = b_outlet["c_int"], b_outlet["normal"], b_outlet["area"]
            U_local_c = U_local[c_int]
            U_ghost = np.tile(np.array([rho_inf, rho_inf * u_inf, 0.0, E_inf]), (len(c_int), 1))
            F_b = compute_convective_flux(U_local_c, U_ghost, normal, area, gamma)
            np.add.at(Rloc, c_int, -F_b)

        b_walls = b_arrays.get("Walls")
        if b_walls is not None:
            c_int, normal, area = b_walls["c_int"], b_walls["normal"], b_walls["area"]
            U_local_c = U_local[c_int]
            U_ghost = U_local_c.copy()
            rho = U_local_c[:, 0]
            v_normal = (U_local_c[:, 1]/rho) * normal[:, 0] + (U_local_c[:, 2]/rho) * normal[:, 1]
            U_ghost[:, 1] -= 2 * rho * v_normal * normal[:, 0]
            U_ghost[:, 2] -= 2 * rho * v_normal * normal[:, 1]
            F_b = compute_convective_flux(U_local_c, U_ghost, normal, area, gamma)
            np.add.at(Rloc, c_int, -F_b)

        return Rloc

    t, step = 0.0, 0
    avg_iter_time = None
    history_data = [] 
    
    print("Inizio ciclo di simulazione (Pura computazione vettoriale)...")
    while t < T:
        iter_start = time.time()
        
        # --- VELOCIZZATO: Calcolo del dt CFL interamente vettoriale senza cicli for ---
        rho_vec = U[:, 0]
        u_vec = U[:, 1] / np.maximum(rho_vec, 1e-12)
        v_vec = U[:, 2] / np.maximum(rho_vec, 1e-12)
        p_vec = (gamma - 1.0) * (U[:, 3] - 0.5 * rho_vec * (u_vec**2 + v_vec**2))
        c_vec = np.sqrt(gamma * np.maximum(p_vec, 1e-8) / np.maximum(rho_vec, 1e-8))
        max_speed = np.max(np.sqrt(u_vec**2 + v_vec**2) + c_vec)
        max_speed = max(max_speed, 1e-12)

        dt_cfl = CFL * h_min / max_speed
        dt_used = min(dt, dt_cfl)
        t += dt_used
        step += 1
        
        # SSP RK3
        U1 = U.copy()
        R0 = compute_residual(U1)
        U1 = U + (dt_used / cell_volumes[:, None]) * R0

        R1 = compute_residual(U1)
        U2 = 0.75 * U + 0.25 * (U1 + (dt_used / cell_volumes[:, None]) * R1)

        R2 = compute_residual(U2)
        U = (1.0/3.0) * U + (2.0/3.0) * (U2 + (dt_used / cell_volumes[:, None]) * R2)

        U[:, 0] = np.maximum(U[:, 0], 1e-8)
        kinetic = 0.5 * (U[:, 1]**2 + U[:, 2]**2) / np.maximum(U[:, 0], 1e-12)
        U[:, 3] = np.maximum(U[:, 3], kinetic + 1e-8)

        iter_time = time.time() - iter_start
        if avg_iter_time is None:
            avg_iter_time = iter_time
        else:
            avg_iter_time = 0.9 * avg_iter_time + 0.1 * iter_time

        if step % log_interval == 0:
            rho = U[:, 0]
            p = (gamma - 1.0) * (U[:, 3] - 0.5 * rho * ((U[:, 1]/rho)**2 + (U[:, 2]/rho)**2))

            if np.isnan(rho).any() or np.min(rho) <= 0 or np.isnan(p).any() or np.min(p) <= 0:
                print(f"⚠️ Instabilità a t = {t:.5f}s. Interruzione.")
                break

            if avg_iter_time is not None and dt_used > 0:
                steps_remaining = max(int(np.ceil((T - t) / dt_used)), 0)
                eta_seconds = avg_iter_time * steps_remaining
                eta_str = f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
            else:
                eta_str = "--:--:--"

            percent = min(100.0, 100.0 * t / T)
            print(f"Progress: {t:.4f}/{T:.4f}s ({percent:.1f}%) — ETA {eta_str} — dt_used={dt_used:.3e}")

            # Salviamo il campo di velocità per il rendering finale rapido
            vel_mag = np.sqrt((U[:, 1]/rho)**2 + (U[:, 2]/rho)**2)
            history_data.append((t, vel_mag.copy()))

    # --- FASE DI POST-PROCESSING ULTRA-RAPIDA (No ax.clear, no ArtistAnimation) ---
    print("\nGenerazione rapida del file GIF...")
    if len(history_data) > 0:
        from PIL import Image
        fig, ax = plt.subplots(figsize=(7, 5))
        frames = []
        
        # Primo frame statico di inizializzazione
        t_init, v_init = history_data[0]
        quad_mesh = ax.tripcolor(points[:, 0], points[:, 1], triangles, facecolors=v_init, cmap='jet')
        title_text = ax.set_title(f"Velocità al tempo t = {t_init:.4f} s")
        ax.set_aspect('equal')
        fig.canvas.draw()
        
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(Image.fromarray(image.copy()))
        
        # Aggiornamento fulmineo dei frame successivi modificando solo i dati di colore
        for t_snap, v_mag in history_data[1:]:
            quad_mesh.set_array(v_mag)
            title_text.set_text(f"Velocità al tempo t = {t_snap:.4f} s")
            fig.canvas.draw()
            
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(Image.fromarray(image.copy()))
            
        if frames:
            frames[0].save("simulation.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)

        plt.close(fig)
        print("File 'simulation.gif' salvato con successo!")