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
    # Versione OTTIMIZZATA: minimizza alloc e operazioni ridondanti
    rho_L = UL[:, 0]
    rho_R = UR[:, 0]
    rho_L_safe = np.maximum(rho_L, 1e-12)
    rho_R_safe = np.maximum(rho_R, 1e-12)
    
    u_L = UL[:, 1] / rho_L_safe
    v_L = UL[:, 2] / rho_L_safe
    u_R = UR[:, 1] / rho_R_safe
    v_R = UR[:, 2] / rho_R_safe
    
    vn_L = u_L * normal[:, 0] + v_L * normal[:, 1]
    vn_R = u_R * normal[:, 0] + v_R * normal[:, 1]
    
    # Pressioni
    E_L = UL[:, 3] / rho_L_safe
    E_R = UR[:, 3] / rho_R_safe
    p_L = (gamma - 1.0) * (UL[:, 3] - 0.5 * rho_L * (u_L**2 + v_L**2))
    p_R = (gamma - 1.0) * (UR[:, 3] - 0.5 * rho_R * (u_R**2 + v_R**2))
    
    H_L = E_L + p_L / rho_L_safe
    H_R = E_R + p_R / rho_R_safe
    
    # Flussi (single allocation, then fill)
    Flux = np.empty_like(UL)
    Flux_L_data = np.empty_like(UL)
    Flux_L_data[:, 0] = rho_L * vn_L
    Flux_L_data[:, 1] = rho_L * u_L * vn_L + p_L * normal[:, 0]
    Flux_L_data[:, 2] = rho_L * v_L * vn_L + p_L * normal[:, 1]
    Flux_L_data[:, 3] = rho_L * H_L * vn_L
    
    Flux_R_data = np.empty_like(UR)
    Flux_R_data[:, 0] = rho_R * vn_R
    Flux_R_data[:, 1] = rho_R * u_R * vn_R + p_R * normal[:, 0]
    Flux_R_data[:, 2] = rho_R * v_R * vn_R + p_R * normal[:, 1]
    Flux_R_data[:, 3] = rho_R * H_R * vn_R
    
    p_L_safe = np.maximum(p_L, 1e-5)
    p_R_safe = np.maximum(p_R, 1e-5)
    c_L = np.sqrt(gamma * p_L_safe / rho_L_safe)
    c_R = np.sqrt(gamma * p_R_safe / rho_R_safe)
    max_lambda = np.maximum(np.abs(vn_L) + c_L, np.abs(vn_R) + c_R)
    
    # Flusso finale (Riemann solver centralizzato con dissipazione)
    area_scaled = area[:, None]
    central_flux = 0.5 * (Flux_L_data + Flux_R_data) * area_scaled
    dissipation = 0.5 * max_lambda[:, None] * (UR - UL) * area_scaled
    
    return central_flux - dissipation

def compute_viscous_flux(UL, UR, dist, normal, area, mu=1.8e-5):
    # Versione OTTIMIZZATA: minimizza allocazioni e divisioni
    rho_L_safe = np.maximum(UL[:, 0], 1e-12)
    rho_R_safe = np.maximum(UR[:, 0], 1e-12)
    
    u_L = UL[:, 1] / rho_L_safe
    v_L = UL[:, 2] / rho_L_safe
    u_R = UR[:, 1] / rho_R_safe
    v_R = UR[:, 2] / rho_R_safe
    
    # Gradiente delle velocità (pre-computato una sola volta)
    dist_safe = np.maximum(dist, 1e-6)
    inv_dist = 1.0 / dist_safe
    
    du_dn = (u_R - u_L) * inv_dist
    dv_dn = (v_R - v_L) * inv_dist
    
    # Stress viscoso
    tau_x = mu * du_dn
    tau_y = mu * dv_dn
    
    # Flusso viscoso (pre-allocato)
    Flux_V = np.zeros_like(UL)
    Flux_V[:, 1] = (tau_x * normal[:, 0] + tau_y * normal[:, 1]) * area
    Flux_V[:, 2] = (tau_x * normal[:, 0] + tau_y * normal[:, 1]) * area
    
    # Lavoro degli stress (energia)
    u_avg = 0.5 * (u_L + u_R)
    v_avg = 0.5 * (v_L + v_R)
    Flux_V[:, 3] = (tau_x * u_avg + tau_y * v_avg) * area
    
    return Flux_V

def solve_compressible_fvm(mesh_file="prova.msh", output_video="simulation.gif", T=2.0, dt=0.0002, u_inf=250.0, p_inf=101325.0, rho_inf=1.225, CFL=0.5, artificial_viscosity=0.0, log_interval=200):
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

    # ⚡ PRE-ALLOCAZIONE: Ghost cells e buffers per boundary conditions
    U_inlet_ghost = np.tile(np.array([rho_inf, rho_inf * u_inf, 0.0, E_inf]), (
        len(b_arrays["Inlet"]["c_int"]) if b_arrays.get("Inlet") is not None else 1, 1))
    U_outlet_ghost = np.tile(np.array([rho_inf, rho_inf * u_inf, 0.0, E_inf]), (
        len(b_arrays["Outlet"]["c_int"]) if b_arrays.get("Outlet") is not None else 1, 1))

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

        # ⚡ Boundary Airfoil: Evita copy() e fa riflessione in-place
        b_airfoil = b_arrays.get("Airfoil")
        if b_airfoil is not None:
            c_int, normal, area, dist = b_airfoil["c_int"], b_airfoil["normal"], b_airfoil["area"], b_airfoil["dist"]
            U_local_c = U_local[c_int]
            # Crea ghost direttamente senza copy
            U_ghost_airfoil = np.empty_like(U_local_c)
            U_ghost_airfoil[:, 0] = U_local_c[:, 0]  # Densità identica
            U_ghost_airfoil[:, 1] = -U_local_c[:, 1]  # Riflessione velocità x
            U_ghost_airfoil[:, 2] = -U_local_c[:, 2]  # Riflessione velocità y
            U_ghost_airfoil[:, 3] = U_local_c[:, 3]   # Energia identica
            
            F_b = compute_convective_flux(U_local_c, U_ghost_airfoil, normal, area, gamma) - \
                  compute_viscous_flux(U_local_c, U_ghost_airfoil, dist, normal, area, mu)
            np.add.at(Rloc, c_int, -F_b)

        # ⚡ Boundary Inlet: Usa ghost pre-allocato
        b_inlet = b_arrays.get("Inlet")
        if b_inlet is not None:
            c_int, normal, area = b_inlet["c_int"], b_inlet["normal"], b_inlet["area"]
            U_local_c = U_local[c_int]
            F_b = compute_convective_flux(U_local_c, U_inlet_ghost[:len(c_int)], normal, area, gamma)
            np.add.at(Rloc, c_int, -F_b)

        # ⚡ Boundary Outlet: Usa ghost pre-allocato
        b_outlet = b_arrays.get("Outlet")
        if b_outlet is not None:
            c_int, normal, area = b_outlet["c_int"], b_outlet["normal"], b_outlet["area"]
            U_local_c = U_local[c_int]
            F_b = compute_convective_flux(U_local_c, U_outlet_ghost[:len(c_int)], normal, area, gamma)
            np.add.at(Rloc, c_int, -F_b)

        # ⚡ Boundary Walls: Evita copy() con operazioni in-place
        b_walls = b_arrays.get("Walls")
        if b_walls is not None:
            c_int, normal, area = b_walls["c_int"], b_walls["normal"], b_walls["area"]
            U_local_c = U_local[c_int]
            
            # Calcola ghost senza copy()
            U_ghost_walls = np.empty_like(U_local_c)
            rho = U_local_c[:, 0]
            v_normal = (U_local_c[:, 1]/rho) * normal[:, 0] + (U_local_c[:, 2]/rho) * normal[:, 1]
            
            U_ghost_walls[:, 0] = U_local_c[:, 0]  # Densità identica
            U_ghost_walls[:, 1] = U_local_c[:, 1] - 2 * rho * v_normal * normal[:, 0]
            U_ghost_walls[:, 2] = U_local_c[:, 2] - 2 * rho * v_normal * normal[:, 1]
            U_ghost_walls[:, 3] = U_local_c[:, 3]  # Energia identica
            
            F_b = compute_convective_flux(U_local_c, U_ghost_walls, normal, area, gamma)
            np.add.at(Rloc, c_int, -F_b)

        return Rloc

    t, step = 0.0, 0
    avg_iter_time = None
    history_data = [] 
    
    # ⚡ PRE-ALLOCAZIONE: Buffers per cache di pressione e velocità
    rho_cache = np.zeros(num_cells)
    u_cache = np.zeros(num_cells)
    v_cache = np.zeros(num_cells)
    p_cache = np.zeros(num_cells)
    c_cache = np.zeros(num_cells)
    
    # ⚡ PRE-ALLOCAZIONE: Buffer per residui (evita .copy() ripetuti)
    R_buffer = np.zeros((num_cells, 4))
    U_temp1 = np.zeros((num_cells, 4))
    U_temp2 = np.zeros((num_cells, 4))
    
    print("Inizio ciclo di simulazione (Pura computazione vettoriale - OTTIMIZZATA)...")
    while t < T:
        iter_start = time.time()
        
        # --- VELOCIZZATO: Cache di quantità primitive (una sola volta per iterazione) ---
        rho_cache = U[:, 0]
        rho_safe = np.maximum(rho_cache, 1e-12)
        u_cache = U[:, 1] / rho_safe
        v_cache = U[:, 2] / rho_safe
        p_cache = (gamma - 1.0) * (U[:, 3] - 0.5 * rho_cache * (u_cache**2 + v_cache**2))
        p_safe = np.maximum(p_cache, 1e-8)
        c_cache = np.sqrt(gamma * p_safe / rho_safe)
        
        # --- VELOCIZZATO: Calcolo del dt CFL completamente vettorializzato ---
        max_speed = np.max(np.sqrt(u_cache**2 + v_cache**2) + c_cache)
        max_speed = max(max_speed, 1e-12)
        dt_cfl = CFL * h_min / max_speed
        dt_used = min(dt, dt_cfl)
        t += dt_used
        step += 1
        
        # ⚡ PRE-CALCOLO: fattore di scala per RK3 (evita 3 divisioni per volume)
        dt_over_vol = dt_used / cell_volumes
        
        # --- SSP RK3 OTTIMIZZATO ---
        R_buffer = compute_residual(U)
        
        # Stage 1: U1 = U + dt*R0/V
        np.copyto(U_temp1, U)
        U_temp1 += R_buffer * dt_over_vol[:, None]

        R_buffer = compute_residual(U_temp1)
        
        # Stage 2: U2 = 0.75*U + 0.25*(U1 + dt*R1/V)
        np.multiply(U, 0.75, out=U_temp2)
        U_temp2 += 0.25 * (U_temp1 + R_buffer * dt_over_vol[:, None])

        R_buffer = compute_residual(U_temp2)
        
        # Stage 3: U = (1/3)*U + (2/3)*(U2 + dt*R2/V)
        np.multiply(U, 1.0/3.0, out=U)
        U += (2.0/3.0) * (U_temp2 + R_buffer * dt_over_vol[:, None])

        # ⚡ Clip di sicurezza IN-PLACE (nessuna copia)
        U[:, 0] = np.maximum(U[:, 0], 1e-8)
        kinetic = 0.5 * (U[:, 1]**2 + U[:, 2]**2) / np.maximum(U[:, 0], 1e-12)
        U[:, 3] = np.maximum(U[:, 3], kinetic + 1e-8)

        iter_time = time.time() - iter_start
        if avg_iter_time is None:
            avg_iter_time = iter_time
        else:
            avg_iter_time = 0.9 * avg_iter_time + 0.1 * iter_time

        if step % log_interval == 0:
            # Riutilizza il cache di pressione/densità
            if np.isnan(rho_cache).any() or np.min(rho_cache) <= 0 or np.isnan(p_cache).any() or np.min(p_cache) <= 0:
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
            vel_mag = np.sqrt(u_cache**2 + v_cache**2)  # Riutilizza cache
            history_data.append((t, vel_mag.copy()))

    # ⚡ CALCOLO DELLA PORTANZA: Riutilizza il cache di pressione dell'ultima iterazione
    lift_force = 0.0
    b_airfoil = b_arrays.get("Airfoil")
    if b_airfoil is not None:
        c_int, normal, area = b_airfoil["c_int"], b_airfoil["normal"], b_airfoil["area"]
        # Usa p_cache già calcolato invece di ricalcolare
        lift_force = -np.sum(p_cache[c_int] * normal[:, 1] * area)

    print(f"Portanza calcolata (forza y): {lift_force:.4f} N (per unità di profondità)")

    # --- FASE DI POST-PROCESSING ULTRA-RAPIDA (No ax.clear, no ArtistAnimation) ---
    print(f"\nGenerazione GIF: {output_video}...")
    if len(history_data) > 0:
        from PIL import Image
        fig, ax = plt.subplots(figsize=(7, 5))
        frames = []

        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.5, 0.5)
        
        # Primo frame statico di inizializzazione
        t_init, v_init = history_data[0]
        quad_mesh = ax.tripcolor(points[:, 0], points[:, 1], triangles, facecolors=v_init, cmap='jet')
        cbar = fig.colorbar(quad_mesh, ax=ax)
        cbar.set_label('Velocità (m/s)')
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
            frames[0].save(output_video, save_all=True, append_images=frames[1:], duration=100, loop=0)

        plt.close(fig)
        print(f"File '{output_video}' salvato con successo!")

        return lift_force