import streamlit as st
import os

from navier_stokes import solve_compressible_fvm 
from mesh_generation import mesh_generation 

st.set_page_config(page_title="Aerodynamics Wing Simulator", layout="centered")

st.title("Aerodynamics Wing Simulator ✈️")
st.sidebar.header("Parametri di Simulazione")

# --- PARAMETRI GEOMETRICI ---
st.sidebar.subheader("Geometria Profilo Alare")
naca_codice = st.sidebar.text_input("Profilo NACA (4 cifre)", value="2412", max_chars=4)
angle_deg = st.sidebar.slider("Angolo di Attacco (gradi)", min_value=-5.0, max_value=20.0, value=8.0, step=0.5)

# --- PARAMETRI FISICI ---
st.sidebar.subheader("Proprietà del Fluido")
u_inf_knots = st.sidebar.slider("Velocità dell'aria (nodi)", min_value=40.0, max_value=550.0, value=250.0, step=5.0)
u_inf = u_inf_knots * 0.514444  

p_inf = st.sidebar.number_input("Pressione Atmosferica (Pa)", value=101325.0)
tempo_sim = st.sidebar.slider("Tempo di simulazione (s)", min_value=0.1, max_value=1.0, value=0.2, step=0.1)

CFL = st.sidebar.slider("CFL", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
art_visc = st.sidebar.number_input("Viscosità artificiale", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

mesh_path = "prova.msh"
video_path = "simulation.gif"

if st.sidebar.button("Avvia Simulazione"):
    if os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(mesh_path):
        os.remove(mesh_path)
        
    # FASE 1: Generazione Mesh
    with st.spinner(f"Generazione della griglia per il profilo NACA {naca_codice}... 📐"):
        try:
            mesh_generation(naca_code=naca_codice, n_points=100, angle_deg=angle_deg, output_file=mesh_path)
            st.sidebar.success("Nuova mesh generata con successo!")
        except Exception as e:
            st.error(f"Errore durante la generazione della mesh (Gmsh): {e}")
            st.stop()

    # FASE 2: Solutore Navier-Stokes 
    with st.spinner("Risoluzione delle equazioni di Navier-Stokes in corso... ☕"):
        dt_sicuro = 0.000005 if u_inf > 150 else 0.00002
        # CORRETTO: rimosso l'argomento inesistente progress_callback
        solve_compressible_fvm(
            mesh_file=mesh_path, 
            T=tempo_sim, 
            dt=dt_sicuro, 
            u_inf=u_inf, 
            p_inf=p_inf,
            CFL=CFL, 
            artificial_viscosity=art_visc
        )
    st.success("Simulazione completata con successo!")

# --- VISUALIZZAZIONE RISULTATI ---
st.header("Visualizzazione del Campo di Velocità")

if os.path.exists(video_path):
    # CORRETTO: sostituito use_container_width con use_column_width per compatibilità retroattiva
    st.image(video_path, caption=f"Flusso a {u_inf_knots:.1f} nodi ({u_inf:.1f} m/s) su profilo NACA {naca_codice} (AoA: {angle_deg}°)", use_column_width=True)
else:
    st.info("Configura i parametri a sinistra e clicca su 'Avvia Simulazione' per far partire il calcolo.")