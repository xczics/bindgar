from typing import Optional, List, overload, Tuple, Final
from dataclasses import dataclass
import math
from ..physics import Stefan_Boltzmann, G, R_gas

@dataclass(kw_only=True)
class MagmaOceanParameters:
    rho_B: float = 3500 # Bulk Density, kg m-3
    rho_M: float = 3000 # Magma Ocean Density, kg m-3
    kapa: float = 6e-7 # Thermal Diffusivity (= k/rho_MC_P), m2 s-2
    v: float = 1e2 # Magma Viscosity, Pa s
    k: float = 3 # Thermal Conductivity, W m-1 K-1
    C_p: float = 1200 # Specific Heat Capacity, J kg-1 K-1
    alpha_V: float = 2e-5 # Thermal Expansivity, K-1
    g_s: Optional[float] = None # Surface Gravity, m s-2
    r: float = 1.7e6 # Planet Radius, m
    L: float = 5e6 # Latent Heat of vapourization, J kg-1
    M_mol: float = 0.04 # Molar Mass of the Atmosphere, kg mol-1
    def __post_init__(self):
        if self.g_s is None:
            self.g_s = (4/3) * math.pi * G * self.rho_B * self.r # m s-2
        self.M_tot = (4/3) * math.pi * self.r ** 3 * self.rho_B
    
    def M_critical(self, T: float, yita: float = 1.2) -> float:
        yita_ratio = yita / (yita - 1)
        escape_ratio = R_gas * T / (G * self.M_mol)
        size_factor = (3 / (4 * math.pi * self.rho_B)) ** (1/3)
        return (yita_ratio*escape_ratio*size_factor)**(3/2)


def T_from_T_s(T_s: float, parameter: MagmaOceanParameters) -> float:
    r"""Calculate the average temperature magma ocean from the surface temperature.
    T = T_s * (1 + T_s^2 * (frac{\kapa v}{\rho_M g_s \alpha_v})^(1/4) * (2 \sigma / k)^(3/4))

    Args:
        Ts (float): Surface temperature in K.
        parameter (MagmaOceanParameters): Parameters for the magma ocean.

    Returns:
        float: Average temperature of the magma ocean in K.
    """
    A = parameter.kapa * parameter.v / (parameter.rho_M * parameter.g_s # type: ignore
                                         * parameter.alpha_V ) 
    B = 2 * Stefan_Boltzmann / parameter.k
    Result_from_Calogero = T_s * (1 + T_s ** 2 * A ** (1/4) * B ** (3/4))
    return Result_from_Calogero

def T_s_from_T (T: float, parameter: MagmaOceanParameters, threshold: float = 1e-5) -> float:
    """
    Calculate the surface temperature from the average temperature of the magma ocean.
    It will use the half interval search to find the solution from `T_from_T_s`.
    Args:
        T (float): Average temperature of the magma ocean in K.
        parameter (MagmaOceanParameters): Parameters for the magma ocean.
    Returns:
        float: Surface temperature in K.
    """
    T_s_low = 0.0
    T_s_high = T
    while T_s_high - T_s_low > threshold:
        T_s_mid = (T_s_low + T_s_high) / 2
        T_mid = T_from_T_s(T_s_mid, parameter)
        if T_mid < T:
            T_s_low = T_s_mid
        else:
            T_s_high = T_s_mid
    return (T_s_low + T_s_high) / 2

def vapour_pressure(T: float) -> float:
    r""" Calculate the vapour pressure by empirical expression obtained from thermodynamic liquid-vapour model.
    Ref to Hin et al. 2017.
    lnP = -4.0041 (lnT)^3 + 88.788 (lnT)^2 - 639.3 lnT + 1480.23
    where P is in bar and T is in K.
    Args:
        T (float): Temperature in K.
    Returns:
        float: Vapour pressure in Pa.
    """
    lnT = math.log(T)
    lnP = -4.0041 * lnT**3 + 88.788 * lnT**2 - 639.3 * lnT + 1480.23
    P_bar = math.exp(lnP)
    return P_bar * 1e5  # Convert bar to Pa

def rho_surface_from_T (T: float, M_mol: float) -> float:
    r""" Calculate the surface density of the atmosphere from the temperature.
    rho_s = P_s  / ((R_gas/M_mol) * T)
    where P_s is the vapour pressure at temperature T.
    Args:
        T (float): Temperature in K.
        M_mol (float): Molar mass of the atmosphere in kg mol-1.
    Returns:
        float: Surface density in kg m-3.
    """ 
    P_vapor = vapour_pressure(T)
    M_mol = 0.04  # kg mol-1
    rho_s = P_vapor * M_mol / ( R_gas * T )
    return rho_s

def isothermal_sound_speed (T: float, M_mol: float) -> float:
    return math.sqrt(( R_gas * T ) / M_mol)

def surface_outflow_velocity (T: float, parameter: MagmaOceanParameters, yita: float=1.2):
    r""" Calculate the surface outflow velocity.
    u_s = c_s \sqrt{ 2 * (\frac{\yita - 1}{3 \yita - 1}
                        (\frac{\yita}{\yita - 1} - \frac{GM}{c_s^2r})) ^ \frac{3 \yita - 1}{\yita -1}
                        (\frac{GM}{4c_s^2r})^ {4 \frac{\yita - 1}{3 \yita - 1}
                        (\frac{2}{\yita}) ^ {\frac{2}{\ytia - 1 }} } }
    Args:
        T (float):  Temperature in K.
        parameter (MagmaOceanParameters): Parameters, use r and M_tot
        yita (float): adiabatic index, equal to the ratio of specific heat capacities at constant pressure over constant volume.
    
    Returns:
        u_s (float) :  the surface outflow velocity.
    """
    yita_factor = (yita - 1)/(3 * yita - 1)
    c_s = isothermal_sound_speed (T, parameter.M_mol)
    c_sq2 = c_s ** 2 * parameter.r
    GM = G * parameter.M_tot
    first_part = (yita_factor*((yita / (yita - 1))-(GM / c_sq2))) ** yita_factor if (yita / (yita - 1))-(GM / c_sq2) > 0 else 0
    second_part = (GM / (4 * c_sq2)) ** (4 * yita_factor)
    third_part = (2 / yita) ** ( 2 / (yita - 1 ))
    u_s = c_s * math.sqrt(2*first_part*second_part*third_part) if first_part > 0 else 0
    return u_s

def pMpt (T: float, parameter: MagmaOceanParameters, yita: float=1.2):
    r = parameter.r
    rho_s = rho_surface_from_T (T, parameter.M_mol)
    u_s = surface_outflow_velocity (T, parameter, yita)
    return 2 * math.pi * r ** 2 * rho_s * u_s

def black_body_flux (T_s: float, r: float) -> float:
    return Stefan_Boltzmann * T_s ** 4 * (4 * math.pi * r **2)

def pTpt (T: float, M_l: float, parameter: MagmaOceanParameters, yita: float=1.2):
    """
    Calculate the cooling rate of the magma ocean/ pool.

    Args:
        T (float):  Temperature in K.
        M_l (float): The mass of the magma ocean/ pool
        parameter (MagmaOceanParameters): Magma ocean parameters.
        yita (float): adiabatic index.
    
    Return: float, the cooling rate of the magma ocean.

    """
    F_black = black_body_flux ( T_s = T_s_from_T (T, parameter),r = parameter.r)
    Heat_capacity = M_l * parameter.C_p
    GM = G * parameter.M_tot
    if parameter.M_tot > parameter.M_critical(T):
        return F_black / Heat_capacity
    else:
        return (pMpt(T,parameter,yita)*(GM / parameter.r + parameter.L) + F_black) / Heat_capacity
    
def distribution_coefficient (T: float, version: str='Hin_2017_K') -> float:
    if version == 'Hin_2017_K':
        if T <= 2500:
            return 4.5 * math.exp(3.44e-3 * (T-1800)) 
        else:
            return 50 
    else:
        raise ValueError(f"Unknown version {version} for distribution_coefficient")

def step_concentration (T: float, M_l: float, Mass_loss_step:float,version_D: str='Hin_2017_K') -> float:
    r""" Calculate the concentration step of the volatile in the magma ocean during a time step Dt.
    C_step/C =  \frac{M_l}{M_l + (\frac{\partial M_l}{\partial t}) \Delta t (D - 1)}
    where C is the current concentration, D is the distribution coefficient, pMpt is the mass loss rate, M_l is the mass of the magma ocean.
    Args:
        T (float): Temperature in K.
        M_l (float): Mass of the magma ocean in kg.
        Mass_loss_step (float): The loss of bulk mass during the step.
        version_D (str): Version of the distribution coefficient.
    Returns:
        float: Concentration step ratio.
    """
    D = distribution_coefficient (T, version_D)
    C_step_factor = M_l / (M_l + Mass_loss_step * (D - 1))
    return C_step_factor

def devoltilization (T_init: float, M_l_init: float, params: MagmaOceanParameters, dT: float=0.1, T_end: float=1200, final_only: bool=True) -> tuple:
    r""" Simulate the devolatilization process of the magma ocean from initial temperature T_init and initial mass M_l_init.
    Args:
        T_init (float): Initial temperature in K.
        M_l_init (float): Initial mass of the magma ocean in kg.
        params (MagmaOceanParameters): Parameters of the magma ocean.
        dT (float): Temperature step in K.
        T_end (float): End temperature in K.
    Returns:
        t_total (float): Total time of the devolatilization process in s.
        M_loss_total (float): Total mass loss during the devolatilization process in kg.
        C_final (float): Final concentration ratio of the volatile in the magma ocean.
    """
    T = T_init
    M_l = M_l_init
    t_total = 0.0
    M_loss_total = 0.0
    C = 1.0
    T_array = []
    t_total_array = []
    M_loss_array = []
    C_array = []
    if not final_only:
        T_array.append(T)
        t_total_array.append(float(t_total))
        M_loss_array.append(float(M_loss_total))
        C_array.append(float(C))
    while T > 1200:
        cooling_rate = float(pTpt(T, M_l, params))
        Dt = dT / cooling_rate
        t_total += Dt
        M_loss_step = float(pMpt(T,params) * Dt)
        M_loss_total += M_loss_step
        C *= float(step_concentration (T, M_l, M_loss_step))
        T -= dT
        if not final_only:
            T_array.append(T)
            t_total_array.append(t_total)
            M_loss_array.append(M_loss_total)
            C_array.append(C)
    if final_only:
        return T, t_total, M_loss_total, C
    else:
        return T_array, t_total_array, M_loss_array, C_array

if __name__ == "__main__":
    """
    Example usage and plot the parameter test.
    It will be a 2×2 subplot.
    a). The cooling process with different T_init (= 1800, 2000, 3000K). x is the time, y_left is the T evolution, y_right is the C evolution.
    b). The relationship between the T and T_s, with different parameters.
    c). The contour plot of the C in a T_int and M_l space.
    d). The contour plot of the C in a T_int and r space.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # a). The cooling process with different T_init (= 1800, 2000, 3000K). x is the time, y_left is the T evolution, y_right is the C evolution.
    T_inits = [1800, 2000, 3000]
    plt.figure(figsize=(12,10))
    ax1 = plt.subplot(2,2,1)
    default_params = MagmaOceanParameters()
    M_l = default_params.M_tot * 0.7 * 0.3
    T_arrays = []
    t_total_arrays = []
    M_loss_arrays = []
    C_arrays = []
    for T_init in T_inits:
        T_array, t_total_array, M_loss_array, C_array = devoltilization(T_init, M_l, MagmaOceanParameters(), final_only=False)
        T_arrays.append(T_array)
        t_total_arrays.append(t_total_array)
        M_loss_arrays.append(M_loss_array)
        C_arrays.append(C_array)
    for i, T_init in enumerate(T_inits):
        ax1.plot(np.array(t_total_arrays[i])/365.25/24/3600, T_arrays[i], label=r'$T_{init}$='+f'{T_init}K')
    # set x as log scale
    ax1.set_xscale('log')
    plt.xlabel('Time (year)')
    plt.ylabel('Temperature (K)')
    # 设置不要右边框线
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(right=False)

    plt.legend()
    ax2 = ax1.twinx()
    for i, T_init in enumerate(T_inits):
        ax2.plot(np.array(t_total_arrays[i])/365.25/24/3600, C_arrays[i], '--', label=r'$T_{init}$='+f'{T_init}K')
    plt.grid(color='white', linestyle='--', linewidth=0.5)
    plt.ylabel('Concentration Ratio')
    plt.legend()
    # 设置右侧框线为虚线
    # 设置右侧框线为虚线
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_linestyle('--')

    # b). The relationship between the T and T_s, with different parameters.
    plt.subplot(2,2,2)
    T_values = np.linspace(1200, 4000, 100)
    param_sets = {
        "Default": MagmaOceanParameters(),
        "Larger Radius": MagmaOceanParameters(r=2.8e6),
        "Smaller Radius": MagmaOceanParameters(r=9e5),
        "Smaller Viscosity": MagmaOceanParameters(v=1),
    }
    for label, params in param_sets.items():
        T_s_values = [T_s_from_T(T, params) for T in T_values]
        plt.plot(T_s_values, T_values, label=label)
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.xlabel(r'Surface Temperature $T_{surf}$ (K)')
    plt.ylabel('Average Temperature T (K)')
    plt.legend()

    # c). The contour plot of the C in a T_int and M_l space.
    plt.subplot(2,2,3)
    T_init_grid = np.linspace(1500, 4000, 20)
    M_l_factor = np.linspace(0.1, 0.9, 5)
    T_init_mesh, M_l_factor_mesh = np.meshgrid(T_init_grid, M_l_factor)
    C_mesh = np.zeros_like(T_init_mesh)
    for i in range(T_init_mesh.shape[0]):
        for j in range(T_init_mesh.shape[1]):
            print(i,j)
            M_l = default_params.M_tot * 0.7 * M_l_factor_mesh[i,j]
            _, _, _, C_final = devoltilization(T_init_mesh[i,j], M_l, default_params, final_only=True)
            C_mesh[i,j] = C_final
    contour = plt.contourf(T_init_mesh, M_l_factor_mesh, C_mesh, levels=20, cmap='viridis')
    plt.grid(color='white', linestyle='--', linewidth=0.5)
    plt.colorbar(contour, label='Final Concentration in Melts')
    plt.xlabel(r'Initial Temperature $T_{init}$ (K)')
    plt.ylabel('Magma Ocean Mass Fraction')
    plt.title('Final Concentration Contour')
    # d). The contour plot of the C in a T_int and r space.
    plt.subplot(2,2,4)
    # let r_values in a log sapce, from 2e5 to 1e7
    r_values = np.logspace(math.log10(2e5), math.log10(1e7), 20)
    T_init_grid = np.linspace(1500, 4000, 20)
    T_init_mesh, r_mesh = np.meshgrid(T_init_grid, r_values)
    C_mesh = np.zeros_like(T_init_mesh)
    for i in range(T_init_mesh.shape[0]):
        for j in range(T_init_mesh.shape[1]):
            print(i,j)
            params = MagmaOceanParameters(r=r_mesh[i,j])
            M_l = params.M_tot * 0.7 * 0.3
            _, _, _, C_final = devoltilization(T_init_mesh[i,j], M_l, params, final_only=True)
            C_mesh[i,j] = C_final
    contour = plt.contourf(T_init_mesh, r_mesh/1e6, C_mesh, levels=20, cmap='viridis')
    # r in log space
    plt.yscale('log')
    # add more ticks for y axis: Moon (1.7e6), Mars (3.4e6), Earth (6.4e6)
    plt.yticks([0.5, 0.8, 1.0, 1.7, 3.4, 5.0, 6.4], labels=['0.5','0.8','1.0', 'Moon', 'Mars', '5.0', 'Earth'])
    plt.grid(color='white', linestyle='--', linewidth=0.5)
    plt.colorbar(contour, label='Final Concentration in Melts')
    plt.xlabel(r'Initial Temperature $T_{init}$ (K)')
    plt.ylabel(r'Planet Radius ($10^6$ m)')
    plt.title('Final Concentration Contour')
    plt.tight_layout()
    #plt.show()
    plt.savefig('magma_ocean_devolatilization_parameter_study.png', dpi=600)
    
    