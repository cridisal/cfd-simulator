import numpy as np

def generate_naca4(profile, n_points=100):
    """
    Generate the coordinates x,y of a 4-digit NACA profile.
    The wing chord is normalized at 1.0.
    """
    m = int(profile[0]) / 100.0  # Max curvature
    p = int(profile[1]) / 10.0   # Position of the max curvature
    t = int(profile[2:]) / 100.0 # Max thickness
    
    x = np.linspace(0, 1, n_points)
    
    # NACA equation of thickness
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    # Equation of the mean camber line, initialized as zeroes (for the symmetric wing)
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    # If p>0, we add the case of the non-symmetric profile
    if p > 0:

        # Here we distinguish between the two curves, one before and one after the max curvature point
        mask_front = x <= p
        mask_back = x > p
        
        # We create the profile for the front and the derivative, to then get the coordinates
        yc[mask_front] = (m / p**2) * (2 * p * x[mask_front] - x[mask_front]**2)
        dyc_dx[mask_front] = (2 * m / p**2) * (p - x[mask_front])
        
        # Then the profile for the back
        yc[mask_back] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[mask_back] - x[mask_back]**2)
        dyc_dx[mask_back] = (2 * m / (1 - p)**2) * (p - x[mask_back])

    # This theta is fundamental to get the perpendicular to the center line    
    theta = np.arctan(dyc_dx)
    
    # Calculation for the upper and the lower part of the wing
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    # We join the arrays and create the actual profile
    x_wing = np.concatenate([xu[::-1], xl[1:]])
    y_wing = np.concatenate([yu[::-1], yl[1:]])
    
    return x_wing, y_wing

def profile_rotation(x, y, angle_deg):
    """
    Rotate the coordinates of the wing profile around the axis (0,0) 
    to simulate the Angle of Attack
    """
    angle_rad = np.radians(-angle_deg) # Negative for convention
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return x_rot, y_rot