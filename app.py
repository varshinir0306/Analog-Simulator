# app.py  
"""  
Energy Band Diagram Visualization for PN Junction, BJT, and MOSFET  
Author: Semiconductor Physics Visualization  
Description: Interactive Flask application to visualize energy band diagrams  
with respect to applied voltage for semiconductor devices.  
"""  

from flask import Flask, render_template, request, jsonify  
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib  
from matplotlib.backends.backend_agg import FigureCanvasAgg  
from io import BytesIO  
import base64  

app = Flask(__name__)  
matplotlib.use('Agg')  

# ==================== CONSTANTS ====================  
k_B = 1.381e-23  # Boltzmann constant (J/K)  
q = 1.602e-19    # Elementary charge (C)  
epsilon_0 = 8.854e-12  # Permittivity of free space (F/m)  
h = 6.626e-34    # Planck constant (J·s)  

# ==================== PN JUNCTION CLASS ====================  
class PNJunction:  
    """  
    PN Junction Band Diagram Calculator  
    
    Theory:   
    Energy band diagram shows the conduction band (Ec) and valence band (Ev)  
    across the PN junction. Under applied voltage V:  
    - Depletion width: W = sqrt(2*epsilon_s*(N_d + N_a)/(q*N_d*N_a) * (V_bi - V))  
    - Built-in voltage: V_bi = (k_B*T/q) * ln(N_a*N_d/n_i^2)  
    where V_bi is forward bias (positive for reverse bias)  
    """  
    
    def __init__(self, N_a=1e16, N_d=1e16, T=300):  
        self.N_a = N_a  # Acceptor concentration (cm^-3)  
        self.N_d = N_d  # Donor concentration (cm^-3)  
        self.T = T      # Temperature (K)  
        self.n_i = 1.5e10  # Intrinsic carrier concentration at 300K (cm^-3)  
        self.epsilon_s = 11.7 * epsilon_0  # Relative permittivity of Si  
        self.E_g = 1.12  # Bandgap energy (eV) at 300K  
        
    def calculate_band_diagram(self, V_applied):  
        """  
        Calculate conduction and valence band positions  
        V_applied: Applied voltage (V) - positive for reverse bias, negative for forward  
        """  
        # Calculate built-in potential  
        V_bi = (k_B * self.T / q) * np.log((self.N_a * self.N_d) / (self.n_i**2))  
        V_bi_ev = V_bi / q  # Convert to eV  
        
        # Total voltage across junction  
        V_total = V_bi_ev - V_applied  
        
        # Depletion width  
        W = np.sqrt(2 * self.epsilon_s * (self.N_a + self.N_d) /   
                    (q * self.N_a * self.N_d) * V_total)  
        
        # Depletion region positions  
        x_n = W * self.N_a / (self.N_a + self.N_d)  # Depletion in n-region  
        x_p = W * self.N_d / (self.N_a + self.N_d)  # Depletion in p-region  
        
        # Create position array  
        x_array = np.linspace(-x_p*1e4, x_n*1e4, 1000)  # Convert to nm  
        
        # Band positions (arbitrary reference)  
        E_c = np.zeros_like(x_array)  
        E_v = np.zeros_like(x_array)  
        E_f = np.zeros_like(x_array)  
        
        # P-region (x < -x_p*1e4)  
        p_mask = x_array < -x_p*1e4  
        E_c[p_mask] = 0  
        E_v[p_mask] = -self.E_g  
        E_f[p_mask] = -0.2  
        
        # Depletion region (parabolic potential)  
        depl_mask = (x_array >= -x_p*1e4) & (x_array <= x_n*1e4)  
        x_depl_norm = (x_array[depl_mask] + x_p*1e4) / (W*1e4)  
        potential = V_total * (1 - x_depl_norm**2)  
        E_c[depl_mask] = potential  
        E_v[depl_mask] = potential - self.E_g  
        
        # N-region (x > x_n*1e4)  
        n_mask = x_array > x_n*1e4  
        E_c[n_mask] = V_total  
        E_v[n_mask] = V_total - self.E_g  
        E_f[n_mask] = -0.1  
        
        return x_array, E_c, E_v, E_f, V_bi_ev  
    
    def plot_diagram(self, V_applied):  
        """Generate band diagram plot"""  
        x, E_c, E_v, E_f, V_bi = self.calculate_band_diagram(V_applied)  
        
        fig, ax = plt.subplots(figsize=(12, 6))  
        ax.plot(x, E_c, 'b-', linewidth=2.5, label='Conduction Band (Ec)')  
        ax.plot(x, E_v, 'r-', linewidth=2.5, label='Valence Band (Ev)')  
        ax.fill_between(x, E_v, E_c, alpha=0.2, color='gray', label='Bandgap')  
        
        ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Junction')  
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)  
        
        ax.set_xlabel('Position (nm)', fontsize=12, fontweight='bold')  
        ax.set_ylabel('Energy (eV)', fontsize=12, fontweight='bold')  
        ax.set_title(f'PN Junction Energy Band Diagram\nApplied Voltage: {V_applied:.2f}V | Built-in: {V_bi:.3f}V',   
                     fontsize=13, fontweight='bold')  
        ax.legend(loc='best', fontsize=10)  
        ax.grid(True, alpha=0.3)  
        
        return fig  

# ==================== BJT CLASS ====================  
class BJT:  
    """  
    Bipolar Junction Transistor Band Diagram  
    
    Theory:  
    BJT has two junctions: Base-Emitter (BE) and Base-Collector (BC)  
    BE junction forward biased, BC junction reverse biased in active mode  
    Band bending controlled by:  
    - V_BE: Forward bias voltage  
    - V_CE: Collector-Emitter voltage  
    """  
    
    def __init__(self, N_e=1e17, N_b=1e15, N_c=1e15, T=300):  
        self.N_e = N_e  # Emitter doping (cm^-3)  
        self.N_b = N_b  # Base doping (cm^-3)  
        self.N_c = N_c  # Collector doping (cm^-3)  
        self.T = T  
        self.E_g = 1.12  
        self.epsilon_s = 11.7 * epsilon_0  
        self.n_i = 1.5e10  
        
    def calculate_bjt_diagram(self, V_BE, V_CE):  
        """  
        V_BE: Base-Emitter voltage (V)  
        V_CE: Collector-Emitter voltage (V)  
        """  
        # Position array (E-B-C regions)  
        x = np.linspace(0, 300, 1000)  
        
        # Emitter region (0-50nm)  
        E_c = np.zeros_like(x, dtype=float)  
        E_v = np.zeros_like(x, dtype=float)  
        
        # Emitter  
        e_mask = x < 50  
        E_c[e_mask] = -0.3  
        E_v[e_mask] = -1.42  
        
        # Base-Emitter junction region (50-80nm)  
        be_mask = (x >= 50) & (x < 80)  
        x_be_norm = (x[be_mask] - 50) / 30  
        E_c[be_mask] = -0.3 + 0.15 * (1 - np.exp(-10*x_be_norm))  
        E_v[be_mask] = -1.42 + 0.15 * (1 - np.exp(-10*x_be_norm))  
        
        # Base (80-120nm) - very thin  
        b_mask = (x >= 80) & (x < 120)  
        E_c[b_mask] = -0.15  
        E_v[b_mask] = -1.27  
        
        # Base-Collector junction (120-150nm)  
        bc_mask = (x >= 120) & (x < 150)  
        x_bc_norm = (x[bc_mask] - 120) / 30  
        barrier = V_CE * 0.5 * (1 + np.tanh(5*(x_bc_norm - 0.5)))  
        E_c[bc_mask] = -0.15 + barrier  
        E_v[bc_mask] = -1.27 + barrier  
        
        # Collector (150-300nm)  
        c_mask = x >= 150  
        E_c[c_mask] = V_CE * 0.3  
        E_v[c_mask] = V_CE * 0.3 - self.E_g  
        
        return x, E_c, E_v  
    
    def plot_diagram(self, V_BE, V_CE):  
        """Generate BJT band diagram"""  
        x, E_c, E_v = self.calculate_bjt_diagram(V_BE, V_CE)  
        
        fig, ax = plt.subplots(figsize=(12, 6))  
        ax.plot(x, E_c, 'b-', linewidth=2.5, label='Conduction Band (Ec)')  
        ax.plot(x, E_v, 'r-', linewidth=2.5, label='Valence Band (Ev)')  
        ax.fill_between(x, E_v, E_c, alpha=0.2, color='gray', label='Bandgap')  
        
        # Mark regions  
        ax.axvline(x=50, color='green', linestyle='--', alpha=0.5, label='E-B Junction')  
        ax.axvline(x=120, color='orange', linestyle='--', alpha=0.5, label='B-C Junction')  
        ax.text(25, 0.2, 'Emitter\n(n+)', ha='center', fontsize=9, fontweight='bold')  
        ax.text(100, 0.2, 'Base\n(p)', ha='center', fontsize=9, fontweight='bold')  
        ax.text(225, 0.2, 'Collector\n(n)', ha='center', fontsize=9, fontweight='bold')  
        
        ax.set_xlabel('Position (nm)', fontsize=12, fontweight='bold')  
        ax.set_ylabel('Energy (eV)', fontsize=12, fontweight='bold')  
        ax.set_title(f'BJT Energy Band Diagram (Active Mode)\nV_BE: {V_BE:.2f}V | V_CE: {V_CE:.2f}V',   
                     fontsize=13, fontweight='bold')  
        ax.legend(loc='best', fontsize=10)  
        ax.grid(True, alpha=0.3)  
        
        return fig  

# ==================== MOSFET CLASS ====================  
class MOSFET:  
    """  
    Metal-Oxide-Semiconductor Field-Effect Transistor Band Diagram  
    
    Theory:  
    Band bending at Si-SiO2 interface depends on gate voltage (V_GS)  
    Threshold voltage: V_T = Φ_MS + 2*Φ_F - Q_ox/(C_ox)  
    where:  
    - Φ_MS: Metal-Semiconductor work function difference  
    - Φ_F: Fermi potential = (k_B*T/q)*ln(N_a/n_i)  
    - Q_ox: Fixed oxide charge  
    - C_ox: Oxide capacitance per unit area  
    """  
    
    def __init__(self, N_a=1e15, T=300, oxide_thickness=2):  
        self.N_a = N_a  # Substrate doping (cm^-3)  
        self.T = T  
        self.n_i = 1.5e10  
        self.E_g = 1.12  
        self.epsilon_ox = 3.9 * epsilon_0  # SiO2 permittivity  
        self.oxide_thickness = oxide_thickness * 1e-7  # Convert to cm  
        self.C_ox = self.epsilon_ox / self.oxide_thickness  
        
    def calculate_mosfet_diagram(self, V_GS, V_DS=0):  
        """  
        V_GS: Gate-Source voltage (V)  
        V_DS: Drain-Source voltage (V)  
        """  
        # Fermi potential  
        phi_F = (k_B * self.T / q) * np.log(self.N_a / self.n_i)  
        phi_F_ev = phi_F / q  
        
        # Work function difference (typical Al-Si)  
        phi_MS = -0.45  # eV  
        
        # Flat-band voltage  
        V_FB = phi_MS - 0.05  # Simple approximation  
        
        # Inversion charge  
        Q_inv = self.C_ox * (V_GS - V_FB - 2*phi_F_ev)  
        
        # Position array (Gate-Oxide-Si regions)  
        x = np.linspace(0, 100, 1000)  
        
        E_c = np.zeros_like(x, dtype=float)  
        E_v = np.zeros_like(x, dtype=float)  
        phi = np.zeros_like(x, dtype=float)  
        
        # Gate region (0-10nm) - Simplified as conductor  
        gate_mask = x < 10  
        E_c[gate_mask] = 0  
        E_v[gate_mask] = -self.E_g  
        
        # Oxide region (10-22nm) - Band offset  
        oxide_mask = (x >= 10) & (x < 22)  
        x_ox_norm = (x[oxide_mask] - 10) / 12  
        E_c[oxide_mask] = (V_GS - V_FB) * x_ox_norm + 3.1  # SiO2 conduction band offset  
        E_v[oxide_mask] = E_c[oxide_mask] - 8.9  # SiO2 bandgap  
        
        # Silicon region (22-100nm)  
        si_mask = x >= 22  
        x_si_norm = (x[si_mask] - 22) / 78  
        
        # Surface potential (logarithmic for simplicity)  
        if V_GS < V_FB:  
            # Depletion  
            surface_phi = V_FB - V_GS  
        else:  
            # Accumulation/Inversion  
            surface_phi = -abs(V_GS - V_FB) * np.exp(-5*x_si_norm)  
        
        E_c[si_mask] = surface_phi + 2*phi_F_ev  
        E_v[si_mask] = E_c[si_mask] - self.E_g  
        
        return x, E_c, E_v, V_GS, V_FB  
    
    def plot_diagram(self, V_GS, V_DS=0):  
        """Generate MOSFET band diagram"""  
        x, E_c, E_v, V_GS, V_FB = self.calculate_mosfet_diagram(V_GS, V_DS)  
        
        fig, ax = plt.subplots(figsize=(12, 6))  
        ax.plot(x, E_c, 'b-', linewidth=2.5, label='Conduction Band (Ec)')  
        ax.plot(x, E_v, 'r-', linewidth=2.5, label='Valence Band (Ev)')  
        ax.fill_between(x, E_v, E_c, alpha=0.2, color='gray', label='Bandgap')  
        
        # Mark interfaces  
        ax.axvline(x=10, color='purple', linestyle='--', alpha=0.5, label='Gate-Oxide Interface')  
        ax.axvline(x=22, color='orange', linestyle='--', alpha=0.5, label='Oxide-Si Interface')  
        ax.text(5, 4, 'Gate\n(Metal)', ha='center', fontsize=9, fontweight='bold')  
        ax.text(16, 4, 'Oxide\n(SiO₂)', ha='center', fontsize=9, fontweight='bold')  
        ax.text(60, 4, 'Si Substrate\n(p-type)', ha='center', fontsize=9, fontweight='bold')  
        
        # Determine mode  
        mode = "Accumulation" if V_GS < V_FB else ("Depletion" if V_GS < V_FB + 0.5 else "Inversion")  
        
        ax.set_xlabel('Position (nm)', fontsize=12, fontweight='bold')  
        ax.set_ylabel('Energy (eV)', fontsize=12, fontweight='bold')  
        ax.set_title(f'MOSFET Energy Band Diagram\nV_GS: {V_GS:.2f}V | V_FB: {V_FB:.2f}V | Mode: {mode}',   
                     fontsize=13, fontweight='bold')  
        ax.legend(loc='best', fontsize=10)  
        ax.grid(True, alpha=0.3)  
        ax.set_ylim([-2, 5])  
        
        return fig  

# ==================== FLASK ROUTES ====================  
@app.route('/')  
def index():  
    """Main page"""  
    return '''  
    <!DOCTYPE html>  
    <html>  
    <head>  
        <title>Semiconductor Energy Band Diagrams</title>  
        <style>  
            * { margin: 0; padding: 0; box-sizing: border-box; }  
            body {  
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;  
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);  
                min-height: 100vh;  
                padding: 20px;  
            }  
            .container {  
                max-width: 1400px;  
                margin: 0 auto;  
                background: white;  
                border-radius: 15px;  
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);  
                overflow: hidden;  
            }  
            .header {  
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);  
                color: white;  
                padding: 40px;  
                text-align: center;  
            }  
            .header h1 {  
                font-size: 2.5em;  
                margin-bottom: 10px;  
            }  
            .content {  
                padding: 40px;  
            }  
            .device-section {  
                margin-bottom: 60px;  
                padding: 30px;  
                background: #f8f9fa;  
                border-radius: 10px;  
                border-left: 5px solid #667eea;  
            }  
            .device-section h2 {  
                color: #667eea;  
                margin-bottom: 20px;  
                font-size: 1.8em;  
            }  
            .control-group {  
                display: flex;  
                gap: 20px;  
                margin-bottom: 20px;  
                flex-wrap: wrap;  
                align-items: flex-end;  
            }  
            .control {  
                display: flex;  
                flex-direction: column;  
            }  
            .control label {  
                font-weight: bold;  
                margin-bottom: 8px;  
                color: #333;  
            }  
            .control input {  
                padding: 10px;  
                border: 2px solid #ddd;  
                border-radius: 5px;  
                font-size: 1em;  
                transition: border-color 0.3s;  
            }  
            .control input:focus {  
                outline: none;  
                border-color: #667eea;  
            }  
            .btn {  
                background: #667eea;  
                color: white;  
                padding: 12px 30px;  
                border: none;  
                border-radius: 5px;  
                cursor: pointer;  
                font-size: 1em;  
                font-weight: bold;  
                transition: background 0.3s;  
            }  
            .btn:hover {  
                background: #764ba2;  
            }  
            .plot-container {  
                background: white;  
                padding: 20px;  
                border-radius: 10px;  
                text-align: center;  
                margin-top: 20px;  
            }  
            .plot-container img {  
                max-width: 100%;  
                height: auto;  
                border-radius: 5px;  
            }  
            .equation-box {  
                background: white;  
                padding: 20px;  
                border-radius: 10px;  
                margin-top: 15px;  
                border: 2px solid #667eea;  
            }  
            .equation-box h4 {  
                color: #667eea;  
                margin-bottom: 10px;  
            }  
            .equation-box p {  
                font-family: 'Courier New', monospace;  
                line-height: 1.8;  
                color: #333;  
            }  
            .loading {  
                text-align: center;  
                color: #667eea;  
                font-weight: bold;  
            }  
        </style>  
    </head>  
    <body>  
        <div class="container">  
            <div class="header">  
                <h1>⚛️ Semiconductor Energy Band Diagrams</h1>  
                <p>Interactive Visualization with Applied Voltage Effects</p>  
            </div>  
            
            <div class="content">  
                <!-- PN JUNCTION -->  
                <div class="device-section">  
                    <h2>1️⃣ PN Junction</h2>  
                    <div class="control-group">  
                        <div class="control">  
                            <label for="pn_voltage">Applied Voltage (V):</label>  
                            <input type="number" id="pn_voltage" value="0" step="0.1" min="-2" max="2">  
                        </div>  
                        <button class="btn" onclick="updatePNJunction()">Plot PN Junction</button>  
                    </div>  
                    <div id="pn_plot" class="plot-container"></div>  
                    <div class="equation-box">  
                        <h4>Key Equations:</h4>  
                        <p>  
Built-in Voltage: V_bi = (k_B·T/q)·ln(N_a·N_d/n_i²)<br>
