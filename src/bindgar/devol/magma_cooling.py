from typing import Optional, List, overload, Tuple, Final
from dataclasses import dataclass
import math
import numpy as np
from ..physics import Stefan_Boltzmann, G, R_gas, k_b
from ..common import format_float

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
    sb_factor: float = 1.0 # Black body radiation factor, dimensionless
    sigma: float = 3e-19 # Mean cross section of the molecular of SiO, in m2.
    def __post_init__(self):
        if self.g_s is None:
            self.g_s = (4/3) * math.pi * G * self.rho_B * self.r # m s-2
        self.M_tot = (4/3) * math.pi * self.r ** 3 * self.rho_B
    
    def M_critical(self, T: float, yita: float = 1.2) -> float:
        yita_ratio = yita / (yita - 1)
        escape_ratio = R_gas * T / (G * self.M_mol)
        size_factor = (3 / (4 * math.pi * self.rho_B)) ** (1/3)
        return (yita_ratio*escape_ratio*size_factor)**(3/2)
    
    def mantle_loss(self, mass_loss: float) -> None:
        """
        update the parameters after the mass loss of the magma ocean.
        """
        self.M_tot -= mass_loss
        self.r -= (mass_loss / self.rho_M) / (4 * math.pi * self.r ** 2)
        # recalculate the surface gravity directly from the new mass and radius.
        self.g_s = G * self.M_tot / self.r ** 2

def T_from_T_s(T_s: float, parameter: MagmaOceanParameters, diff=False) -> float:
    r"""Calculate the average temperature magma ocean from the surface temperature.
    T = T_s * (1 + T_s^2 * (frac{\kapa v}{\rho_M g_s \alpha_v})^(1/4) * (2 \sigma / k)^(3/4))

    Args:
        Ts (float): Surface temperature in K.
        parameter (MagmaOceanParameters): Parameters for the magma ocean.

    Returns:
        float: Average temperature of the magma ocean in K.
    """
    A = parameter.kapa * parameter.v / (parameter.g_s * parameter.rho_M  # type: ignore
                                         * parameter.alpha_V ) 
    B = 2 * parameter.sb_factor * Stefan_Boltzmann / parameter.k
    if not diff:
        Result_from_Calogero = T_s * (1 + T_s ** 2 * A ** (1/4) * B ** (3/4))
        return Result_from_Calogero
    else:
        dT_dT_s = 1 + 3 * T_s ** 2 * A ** (1/4) * B ** (3/4)
        return dT_dT_s

def T_s_from_T (T: float, parameter: MagmaOceanParameters, threshold: float = 1e-2, newton=True) -> float:
    """
    Calculate the surface temperature from the average temperature of the magma ocean.
    It will use the half interval search to find the solution from `T_from_T_s`.
    Args:
        T (float): Average temperature of the magma ocean in K.
        parameter (MagmaOceanParameters): Parameters for the magma ocean.
    Returns:
        float: Surface temperature in K.
    """
    if not newton:
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
    else:
        T_s_guess = T - 100
        while abs(T_from_T_s(T_s_guess, parameter) - T) > threshold:
            T_s_guess -= (T_from_T_s(T_s_guess, parameter) - T) / T_from_T_s(T_s_guess, parameter, diff=True)
        return T_s_guess

def vapour_pressure_from_Hin(T: float) -> float:
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

def vapour_pressure_from_Calogero(T: float) -> float:
    #logT = math.log10(T)
    logT = math.log(T)
    return 1e5 * math.exp(-4.0041*logT**3 + 88.7878*logT**2 - 639.302*logT + 1480.23)

def vapour_pressure(T: float) -> float:
    return vapour_pressure_from_Hin(T)

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
    rho_s = P_vapor * M_mol / ( R_gas * T )
    return rho_s # in kg m-3

def isothermal_sound_speed (T: float, M_mol: float) -> float:
    return math.sqrt(( R_gas * T ) / M_mol)

def surface_outflow_velocity (T: float, parameter: MagmaOceanParameters, yita: float=1.2, exact_solve: bool = False, original_Calogero: bool = False):
    r""" Calculate the surface outflow velocity.
    u_s = c_s \sqrt{ 2 * (\frac{\yita - 1}{3 \yita - 1}
                        (\frac{\yita}{\yita - 1} - \frac{GM}{c_s^2r})) ^ \frac{3 \yita - 1}{\yita -1}
                        (\frac{GM}{4c_s^2r})^ {4 \frac{\yita - 1}{3 \yita - 1}
                        (\frac{2}{\yita}) ^ {\frac{2}{\ytia - 1 }} } } for yita != 1.0
    u_s = c_s (r_c/r_s)^2 exp (3/2- \frac{GM}{c_s^2r_s} ) for yita = 1.0, 
        r_c = GM / (2 c_s^2), r_s = r
    Args:
        T (float):  Temperature in K.
        parameter (MagmaOceanParameters): Parameters, use r and M_tot
        yita (float): adiabatic index, equal to the ratio of specific heat capacities at constant pressure over constant volume.
    
    Returns:
        u_s (float) :  the surface outflow velocity.
    """
    c_s = isothermal_sound_speed (T, parameter.M_mol)
    c_s2r = c_s ** 2 * parameter.r
    GM = G * parameter.M_tot
    if yita != 1.0:
        if original_Calogero:
            yita_factor = (yita - 1)/(3 * yita - 1)
            first_part = (yita_factor*((yita / (yita - 1))-(GM / c_s2r))) ** (1 / yita_factor) if (yita / (yita - 1))-(GM / c_s2r) > 0 else 0
            second_part = (GM / (4 * c_s2r)) ** (4 * yita_factor)
            third_part = (2 / yita) ** ( 2 / (yita - 1 ))
            u_s = c_s * math.sqrt(2*first_part*second_part*third_part) if first_part > 0 else 0
        else:
            beta = (yita - 1) /(5 - 3 * yita)
            first_part = (yita/(yita - 1))-(GM / c_s2r)
            if first_part <= 0:
                return 0.0
            x = (beta * first_part) ** (1/beta) * ((GM / (4 * c_s2r)) ** 4) * ((2 / yita) ** ( 2 / (yita - 1 )))
            new_first_part = x + (yita/(yita - 1))-(GM / c_s2r)
            #x = (beta * new_first_part) ** (1/beta) * ((GM / (4 * c_s2r)) ** (4*beta)) * ((2 / yita) ** ( 2 / (yita - 1 )))
            x = (beta * new_first_part) ** (1/beta) * ((GM / (4 * c_s2r)) ** 4) * ((2 / yita) ** ( 2 / (yita - 1 )))
            u_s = c_s * math.sqrt(2 * x) if first_part > 0 else 0
    else:
        r_c = GM / (2 * c_s ** 2)
        r_s = parameter.r
        if not exact_solve:
            u_s = c_s * (r_c / r_s) ** 2 * math.exp(3/2 - G * parameter.M_tot / (c_s ** 2 * r_s))
        else:
            from scipy.special import lambertw
            # Solve for u_s using the Lambert W function
            Z = - (r_c / r_s)**4 * math.exp(3 - 4 * r_c / r_s)
            W = lambertw(Z)
            u_s = c_s * math.sqrt(-W.real)
    #print(u_s,c_s)
    return u_s

def Jeans_escape_pMpt_ref (T: float, parameter: MagmaOceanParameters) -> float:
    r"""
    (\frac{dM}{dt})_J = \rho * \sqrt{\frac{RT}{2\pi m}} * (1 + \lambda_J) * exp(-\lambda_J) * 4 \pi r^2 in kg s-1,
    where n is the number density at the surface, k is the Boltzmann constant, T is the temperature, m is the molar mass in kg/mol, r is the radius of the planet.
    n = rho_s / m is the number density in mol m-3. rho_s is the surface density of the atmosphere in kg m-3, m is the molar mass in kg/mol.
    lambda_J = GMm/RTr, where G is the gravitational constant, M is the mass of the planet, m is the molar mass in kg/mol, R is the gas constant, T is the temperature.

    """
    lambda_J = G * parameter.M_tot * parameter.M_mol / (R_gas * T * parameter.r) # dimensionless
    escape_fraction = (1 + lambda_J) * math.exp(-lambda_J)
    return rho_surface_from_T (T, parameter.M_mol) * parameter.M_mol * math.sqrt(R_gas * T / (2 * math.pi * parameter.M_mol)) * escape_fraction * 4 * math.pi * parameter.r ** 2 # in kg s-1

def Jeans_escape_pMpt(T: float, parameter: MagmaOceanParameters, yita: float = 1.2) -> Tuple[float,float]:
    r"""
    Calculate the mass loss rate (dM/dt) via Jeans escape based on the exobase criterion.
    
    The function accounts for atmospheric structure using two models:
    1. Polytropic model (yita > 1.0): P = C * rho^yita, where temperature decreases linearly 
       with altitude.
    2. Isothermal model (yita = 1.0): T(z) = T_surface, where density decays exponentially.
    
    Key Formulas:
    - Exobase altitude (z_ex) criterion: \int_{z_ex}^{\infty} n(z) \sigma dz = 1
        - Polytropic: z_ex = H_poly * [1 - (m / (sigma * rho_s * H_poly) * (yita/(yita-1)))^((yita-1)/yita)]
        - Isothermal: z_ex = H * ln(sigma * rho_s * H / m)
    - Jeans Parameter: \lambda_J = GMm / (k * T_ex * r_ex)
    - Mass Flux: \Phi_J = \rho_ex * \sqrt{RT_ex / 2\pi m} * (1 + \lambda_J) * exp(-\lambda_J)
    - Total Loss Rate: dM/dt = \Phi_J * 4\pi * r_ex^2
    
    Args:
        T (float): Surface temperature in Kelvin (K).
        parameter (MagmaOceanParameters): Object containing planetary constants (r, M_tot, g_s) 
            and molecular properties (M_mol, sigma).
        yita (float): Polytropic index (gamma). Defaults to 1.2 (convective/adiabatic). 
            Set to 1.0 for an isothermal atmosphere.
            
    Returns:
        float: Total mass escape rate (dM/dt) in kg s-1. Returns 0.0 if the atmosphere 
            is gravitationally bound or physically truncated.
    """
    # Physical Constants
    G = 6.67430e-11 # in m3 kg-1 s-2
    R_gas = 8.31446 # in J mol-1 K-1
    NA = 6.02214076e23 # in mol-1
    
    m_mol = parameter.M_mol            # Molar mass (kg/mol)
    m_molecule = m_mol / NA            # Mass of a single molecule (kg)
    r_s = parameter.r                  # Surface radius (m)
    M_p = parameter.M_tot              # Planet mass (kg)
    sigma = parameter.sigma            # Collision cross-section (m2)
    g_s = parameter.g_s                # Surface gravity (m/s2)
    assert g_s is not None, "This assert is written for the type checker."

    # 1. Surface density rho_s (kg/m3) from the provided helper function
    rho_s = rho_surface_from_T(T, m_mol)
    if rho_s <= 0:
        return 700.0, 0.0

    # 2. Scale height H = RT/mg at the surface (m)
    H = (R_gas * T) / (m_mol * g_s)

    # 3. Determine Exobase Altitude (z_ex) and Temperature (T_ex)
    if abs(yita - 1.0) < 1e-6:
        # --- Isothermal Case ---
        T_ex = T
        # Criterion: n_s * sigma * H * exp(-z/H) = 1
        # term represents the optical depth at the surface
        term = (rho_s * sigma * H) / m_molecule
        z_ex = H * math.log(term) if term > 1.0 else 0.0
    else:
        # --- Polytropic Case ---
        gamma_fac = yita / (yita - 1.0)
        H_poly = H * gamma_fac  # Theoretical maximum altitude where T=0
        
        # Dimensionless term for solving the integral of column density
        term = (m_molecule / (sigma * rho_s * H_poly)) * gamma_fac
        if term >= 1.0:
            # Atmosphere is so thin that the exobase is at the surface
            z_ex = 0.0
        else:
            z_ex = H_poly * (1.0 - term**((yita - 1.0) / yita))
        
        # Temperature at exobase: T_ex = T_s * (1 - z/H_poly)
        T_ex = T * (1.0 - z_ex / H_poly)

    # 4. Physical Sanity Checks
    if T_ex <= 0:
        return 700.0, 0.0
    
    r_ex = r_s + z_ex
    
    # 5. Density at Exobase (rho_ex)
    if abs(yita - 1.0) < 1e-6:
        rho_ex = rho_s * math.exp(-z_ex / H)
    else:
        rho_ex = rho_s * (T_ex / T)**(1.0 / (yita - 1.0))

    # 6. Jeans Escape Calculation
    # Escape parameter lambda_J (Dimensionless ratio of potential to thermal energy)
    lambda_J = (G * M_p * m_mol) / (R_gas * T_ex * r_ex)
    
    # Thermal velocity component (m/s)
    v_th = math.sqrt((R_gas * T_ex) / (2.0 * math.pi * m_mol))
    
    # Mass Flux Phi_J (kg m-2 s-1)
    # Using try-except to handle underflow for heavy molecules like SiO
    try:
        if lambda_J > 700: # Typical limit for math.exp()
            return lambda_J, 0.0
        Phi_J = rho_ex * v_th * (1.0 + lambda_J) * math.exp(-lambda_J)
    except OverflowError:
        # This would only happen if lambda_J was a large negative number
        return lambda_J, 0.0
    except ValueError:
        # Handles cases where T_ex or other terms might be NaN/Inf
        return lambda_J, 0.0
    
    # Total mass loss rate integrated over the exobase sphere
    dMdt = Phi_J * 4.0 * math.pi * (r_ex**2)
    
    return lambda_J, dMdt

def pMpt (T: float, parameter: MagmaOceanParameters, yita: float=1.2, Jeans_only = False, hyrodynamic_only = True, exact_gamma_1: bool = False) -> float:
    if Jeans_only:
        return Jeans_escape_pMpt (T, parameter,yita=yita)[1]
    lambda_J, pMpt_Jeans = Jeans_escape_pMpt (T, parameter, yita)
    rho_s = rho_surface_from_T (T, parameter.M_mol)
    u_s = surface_outflow_velocity (T, parameter, yita, exact_solve=exact_gamma_1)
    use_jeans = (lambda_J > 3 ) if yita ==1.0 else (u_s == 0.0)
    if use_jeans and not hyrodynamic_only:
        return pMpt_Jeans
    else:
        return rho_s * u_s * 4 * math.pi * parameter.r ** 2

def black_body_flux (T_s: float, r: float, sb_factor:float=1.0) -> float:
    return sb_factor * Stefan_Boltzmann * T_s ** 4 * (4 * math.pi * r **2)

def pTpt (T: float, M_l: float, parameter: MagmaOceanParameters,mass_loss_rate: float) -> float:
    """
    Calculate the cooling rate of the magma ocean/ pool.

    Args:
        T (float):  Temperature in K.
        M_l (float): The mass of the magma ocean/ pool
        mass_loss_rate (float): The mass loss rate in kg s-1, which can be calculated by pMpt.
    
    Return: float, the cooling rate of the magma ocean.

    """
    T_s = T_s_from_T (T, parameter)
    F_black = black_body_flux ( T_s = T_s,r = parameter.r, sb_factor=parameter.sb_factor)
    Heat_capacity = M_l * parameter.C_p
    GM = G * parameter.M_tot
    if mass_loss_rate < 1e-5:
        return F_black / Heat_capacity
    else:
        return (mass_loss_rate*(GM / parameter.r + parameter.L) + F_black) / Heat_capacity
    
def distribution_coefficient (T: float, version_D: str='Hin_2017_K',
                              shift10: float|None = None,
                              shifte: float|None = None,
                              ) -> float:
    if shift10 is not None and shifte is not None:
        raise ValueError("Only one of shift10 and shifte can be specified.")
    if shift10 is not None:
        shift_factor = 10 ** shift10
    elif shifte is not None:
        shift_factor = math.exp(shifte)
    else:
        shift_factor = 1.0
    if version_D == 'Hin_2017_K':
        if T <= 2500:
            original = 4.5 * math.exp(3.44e-3 * (T-1800)) 
        else:
            original = 50 
    else:
        raise ValueError(f"Unknown version {version_D} for distribution_coefficient")
    return original * shift_factor

def vi_Hin_2017_K():
    import matplotlib.pyplot as plt
    T_values = np.linspace(1200, 4000, 100)
    D_values = np.array([distribution_coefficient(T, version_D='Hin_2017_K') for T in T_values])
    plt.plot(T_values, D_values)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Distribution Coefficient D')
    plt.title('Distribution Coefficient from Hin et al. 2017')
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.savefig('distribution_coefficient_Hin_2017_K.pdf')

def step_concentration (T: float, M_l: float, Mass_loss_step:float,version_D: str='Hin_2017_K',**kwargs) -> float:
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
    distribution_kwargs = {key: kwargs[key] for key in ['shift10', 'shifte'] if key in kwargs}
    distribution_kwargs["version_D"] = version_D
    D = distribution_coefficient (T, **distribution_kwargs)
    C_step_factor = M_l / (M_l + Mass_loss_step * (D - 1))
    return C_step_factor

def devoltilization(T_init: float, M_l_init: float, 
                    params: MagmaOceanParameters, dT: float=5e-4, 
                    T_end: float=1200,
                    yita: float=1.2,
                    hydrodynamic_only: bool=True,
                    final_only: bool=True,
                    update_params: bool = False,
                    **kwargs) -> tuple:
    r""" Simulate the devolatilization process of the magma ocean from initial temperature T_init and initial mass M_l_init.
    Args:
        T_init (float): Initial temperature in K.
        M_l_init (float): Initial mass of the magma ocean in kg.
        params (MagmaOceanParameters): Parameters of the magma ocean.
        dT (float): Temperature steps, in the ratio of current T. If dT is 5e-4, and current T is 2000K, then the T decrease in each step will be 1.0K in the first step.
        T_end (float): End temperature in K.
    Returns:
        t_total (float): Total time of the devolatilization process in s.
        M_loss_total (float): Total mass loss during the devolatilization process in kg.
        C_final (float): Final concentration ratio of the volatile in the magma ocean.
    """
    distribution_kwargs = {key: kwargs[key] for key in ['shift10', 'shifte','version_D'] if key in kwargs}
    T = T_init
    if T_init > 10000:
        print(f"Warning: T_init {T_init}K is too high, reset it to 10000K.")
        T = 10000
    M_l = M_l_init
    if M_l_init <= 0:
        if final_only:
            return T, 0.0, 0.0, 1.0
        else:
            return [T], [0.0], [0.0], [1.0]
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
    while T > T_end and M_l > 0:
        mass_loss_rate = pMpt(T, params, yita=yita, hyrodynamic_only=hydrodynamic_only)
        cooling_rate = float(pTpt(T, M_l, params, mass_loss_rate))   
        T_decrease = dT * T
        Dt = T_decrease / cooling_rate
        t_total += Dt
        M_loss_step = float(mass_loss_rate * Dt)
        M_loss_total += M_loss_step
        C *= float(step_concentration (T, M_l, M_loss_step, **distribution_kwargs))
        M_l -= M_loss_step
        if update_params:
            params.mantle_loss(M_loss_step)
        T -= T_decrease
        if not final_only:
            T_array.append(T)
            t_total_array.append(t_total)
            M_loss_array.append(M_loss_total)
            C_array.append(C)
    if final_only:
        return T, t_total, M_loss_total, C
    else:
        return T_array, t_total_array, M_loss_array, C_array
    
def devoltilization_Ds(T_init: float, M_l_init: float, 
                    params: MagmaOceanParameters, dT: float=5e-4, 
                    T_end: float=1200,
                    yita: float=1.2,
                    hydrodynamic_only: bool=True,
                    final_only: bool=True,
                    update_params: bool = False,
                    **kwargs) -> tuple:
    """
    similar to devolatilization, but calculate a serious of results with different distribution coefficients.
    """
    distribution_kwargs = {key: kwargs[key] for key in ['version_D'] if key in kwargs}
    if 'D_shiftes' in kwargs:
        D_shiftes = kwargs['D_shiftes']
        D_shift10s = [None] * len(D_shiftes)
    elif 'D_shift10s' in kwargs:
        D_shift10s = kwargs['D_shift10s']
        D_shiftes = [None] * len(D_shift10s)
    else:
        raise ValueError("Either D_shiftes or D_shift10s must be provided in kwargs.")
    T = T_init
    if T_init > 10000:
        print(f"Warning: T_init {T_init}K is too high, reset it to 10000K.")
        T = 10000
    M_l = M_l_init
    if M_l_init <= 0:
        if final_only:
            return T, 0.0, 0.0, [1.0] * len(D_shiftes)
        else:
            return [T], [0.0], [0.0], [[1.0] * len(D_shiftes)]
    t_total = 0.0
    M_loss_total = 0.0
    Cs = [1.0] * len(D_shiftes)
    T_array = []
    t_total_array = []
    M_loss_array = []
    C_array = []
    if not final_only:
        T_array.append(T)
        t_total_array.append(float(t_total))
        M_loss_array.append(float(M_loss_total))
        C_array.append(Cs.copy())
    while T > T_end and M_l > 0:
        mass_loss_rate = pMpt(T, params, yita=yita, hyrodynamic_only=hydrodynamic_only)
        cooling_rate = float(pTpt(T, M_l, params, mass_loss_rate))   
        T_decrease = dT * T
        Dt = T_decrease / cooling_rate
        t_total += Dt
        M_loss_step = float(mass_loss_rate * Dt)
        M_loss_total += M_loss_step
        for i in range(len(Cs)):
            distribution_kwargs_i = {'shift10': D_shift10s[i],**distribution_kwargs} if D_shift10s[i] is not None else {'shifte': D_shiftes[i],**distribution_kwargs}
            Cs[i] *= float(step_concentration (T, M_l, M_loss_step, **distribution_kwargs_i))
        M_l -= M_loss_step
        if update_params:
            params.mantle_loss(M_loss_step)
        T -= T_decrease
        if not final_only:
            T_array.append(T)
            t_total_array.append(t_total)
            M_loss_array.append(M_loss_total)
            C_array.append(Cs.copy())
    if final_only:
        return T, t_total, M_loss_total, *Cs
    else:
        return T_array, t_total_array, M_loss_array, *C_array


def main():
    """
    Example usage and plot the parameter test.
    It will be a 2×2 subplot.
    a). The cooling process with different T_init (= 1800, 2000, 3000K). x is the time, y_left is the T evolution, y_right is the C evolution.
    b). The relationship between the T and T_s, with different parameters.
    c). The contour plot of the C in a T_int and M_l space.
    d). The contour plot of the C in a T_int and r space.
    """
    import matplotlib.pyplot as plt

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
        "Default": MagmaOceanParameters(r=1e6),
        "Larger Radius": MagmaOceanParameters(r=2.8e6),
        "Smaller Radius": MagmaOceanParameters(r=9e5),
        "Smaller Viscosity": MagmaOceanParameters(v=1,r=1e6),
    }
    for label, params in param_sets.items():
        T_s_values = [T_s_from_T(T, params) for T in T_values]
        plt.plot(T_s_values, T_values, label=label)
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.xlabel(r'Surface Temperature $T_{surf}$ (K)')
    plt.ylabel('Average Temperature T (K)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('magma_ocean_devolatilization_parameter_study.pdf')

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
    plt.savefig('magma_ocean_devolatilization_parameter_study.pdf')

def vi_vapour_pressure():
    import matplotlib.pyplot as plt
    T = np.linspace(1200, 4000, 100)
    P = np.array([vapour_pressure(t) for t in T])
    P_eq2 = np.array([vapour_pressure_from_Calogero(t) for t in T])
    plt.plot(T, P/1e5,label='Hin et al. 2017')
    plt.plot(T, P_eq2/1e5, '--', label ='Eq. 2')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Vapour Pressure (bar)')
    plt.savefig('vapour_pressure.pdf')

def vi_M_critical():
    import matplotlib.pyplot as plt
    from ..physics import M_EARTH
    T = np.linspace(1200, 4000, 100)
    parameter_sets = {
        "default": MagmaOceanParameters(),
        "more compressed": MagmaOceanParameters(rho_B=5000),
        "less compressed": MagmaOceanParameters(rho_B=2000),
        "lighter atmosphere": MagmaOceanParameters(M_mol=0.004),
    }
    for label, params in parameter_sets.items():
        M_critical_values = np.array([params.M_critical(t) for t in T])
        plt.plot(M_critical_values, T, label=label)
    plt.ylabel('Temperature (K)')
    plt.xlabel(r'Critical Mass (kg)')
    plt.xscale('log')
    # legend at left-top corner
    plt.legend(loc='upper left')
    plt.savefig('M_critical.pdf')

def vi_effect_blackbody():
    """
    Test the effect of blackbody by assuming the sb_factor is 1.0, 0.1, 1e-2, and 1e-3.
    Test the cooling and devol process with T_init = 2500K (in subfig a), and the T-T_s relationship (in subfig b).
    """
    import matplotlib.pyplot as plt
    T_init = 2500
    sb_factors = [1.0, 0.1, 1e-2, 1e-3]
    plt.figure(figsize=(12,5))
    ax1 = plt.subplot(1,2,1)
    ax1_twin = ax1.twinx()
    for sb_factor in sb_factors:
        params = MagmaOceanParameters(sb_factor=sb_factor)
        T_array, t_total_array, M_loss_array, C_array = devoltilization(T_init, params.M_tot * 0.7 * 0.3, params, final_only=False)
        ax1.plot(np.array(t_total_array)/365.25/24/3600, T_array, label=r'$\epsilon$='+format_float(sb_factor),alpha=0.8) # type: ignore
        ax1_twin.plot(np.array(t_total_array)/365.25/24/3600, C_array, '--', label=r'$\epsilon$='+format_float(sb_factor),alpha=0.8) # type: ignore
    ax1.set_xscale('log')
    ax1.set_xlabel('Time (year)')
    ax1.set_ylabel('Temperature (K)')
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(right=False)
    ax1.legend(loc='upper left')
    ax1.grid(color='white', linestyle='--', linewidth=0.5)
    ax1_twin.set_ylabel('Concentration Ratio')
    ax1_twin.spines['right'].set_visible(True)
    ax1_twin.spines['right'].set_linestyle('--')
    ax1_twin.legend(loc='upper right')
    plt.subplot(1,2,2)
    T_values = np.linspace(1200, 4000, 100)
    for sb_factor in sb_factors:
        params = MagmaOceanParameters(sb_factor=sb_factor)
        T_s_values = [T_s_from_T(T, params) for T in T_values]
        plt.plot(T_s_values, T_values, label=r'$\epsilon$='+format_float(sb_factor))
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.xlabel(r'Surface Temperature $T_{surf}$ (K)')
    plt.ylabel('Average Temperature T (K)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('effect_blackbody.pdf')
    """
    More test for r= 1e5, 5e5, 1e6, 1.5e6. 
    T_init set as 2000K. Each r in a subfig,
    and only the cooling and devol process. 
    No T-T_s relationship.
    """
    plt.figure(figsize=(18,10))
    r_values = [1e5, 5e5, 1e6, 1.5e6, 1.5e6, 1.5e6]
    T_init = 2000
    for i, r in enumerate(r_values):
        if i == 4:
            T_init = 1800
        if i == 5:
            T_init = 2500
        ax = plt.subplot(2,3,i+1)
        ax_twin = ax.twinx()
        for j, sb_factor in enumerate(sb_factors):
            params = MagmaOceanParameters(r=r, sb_factor=sb_factor)
            T_array, t_total_array, M_loss_array, C_array = devoltilization(T_init, params.M_tot * 0.7 * 0.3, params, final_only=False)
            ax.plot(np.array(t_total_array)/365.25/24/3600, T_array, label=r'$\epsilon$='+format_float(sb_factor),alpha=0.8) # type: ignore 
            ax_twin.plot(np.array(t_total_array)/365.25/24/3600, C_array, '--', label=r'$\epsilon$='+format_float(sb_factor),alpha=0.8) # type: ignore
        ax.set_xscale('log')
        ax.set_xlabel('Time (year)')
        ax.set_ylabel('Temperature (K)')
        ax.spines['right'].set_visible(False)
        ax.tick_params(right=False)
        ax.legend(loc='upper left')
        ax.grid(color='white', linestyle='--', linewidth=0.5)
        ax_twin.set_ylabel('Concentration Ratio')
        ax_twin.spines['right'].set_visible(True)
        ax_twin.spines['right'].set_linestyle('--')
        ax_twin.legend(loc='upper right')
        ax.set_title('r = '+format_float(r,significant_digits=2,max_decimal_places=1)+' m')
    plt.tight_layout()
    plt.savefig('effect_blackbody_radius.pdf')

def benchmark_binary_and_newton():
    # 测试T_s_from_T的二分法和牛顿法的计算速度
    import time
    T_values = np.linspace(1200, 8000, 10000)
    Results_binary = np.zeros_like(T_values)
    Results_newton = np.zeros_like(T_values)
    params = MagmaOceanParameters()
    start_time = time.time()
    for i, T in enumerate(T_values):
        Results_binary[i] = T_s_from_T(T, params, newton=False)
    binary_time = time.time() - start_time
    start_time = time.time()
    for i, T in enumerate(T_values):
        Results_newton[i] = T_s_from_T(T, params, newton=True)
    newton_time = time.time() - start_time
    print(f'Binary search time: {binary_time:.4f} seconds')
    print(f'Newton method time: {newton_time:.4f} seconds')
    # plot the results to show the difference between the two methods
    import matplotlib.pyplot as plt
    plt.plot(T_values, Results_binary - Results_newton, label='Binary - Newton')
    plt.xlabel('Average Temperature T (K)')
    plt.ylabel('Difference in Surface Temperature T_s (K)')
    plt.title('Difference between Binary Search and Newton Method for T_s_from_T')
    plt.legend()
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.savefig('benchmark_binary_newton.pdf')

def vi_pMpt():
    # 测试yita=1.4, 1.2, 1.0和仅考虑Jeans escape的pMpt随温度变化的情况对比。
    # T from 1200K to 4000K, rho_B = 3500 kg m-3, M_mol = 0.04 kg mol-1.
    # test different r in different color, different line style for different yita and Jeans escape.
    # r in Vesta size, Moon size, Mars size, and 1.5 Mars size.
    import matplotlib.pyplot as plt
    T_values = np.linspace(1000, 5000, 100)
    r_labels = {
        'Vesta': 2.6e5,
        'Moon': 1.7e6,
        'Mars': 3.4e6,
    }
    color_map = ['C0', 'C1', 'C2', 'C3']
    for index, (label, r) in enumerate(r_labels.items()):
        params = MagmaOceanParameters(r=r)
        pMpt_yita_1_4 = np.array([pMpt(T, params, yita=1.4) for T in T_values])
        pMpt_yita_1_2 = np.array([pMpt(T, params, yita=1.2) for T in T_values])
        pMpt_yita_1_0 = np.array([pMpt(T, params, yita=1.0) for T in T_values])
        pMpt_Jeans = np.array([pMpt(T, params, Jeans_only=True) for T in T_values])
        plt.plot(T_values, pMpt_yita_1_4, label=label+', '+r'$\gamma=1.4$', color=color_map[index], linestyle='-.')
        plt.plot(T_values, pMpt_yita_1_2, label=label+', '+r'$\gamma=1.2$', color=color_map[index], linestyle='-')
        plt.plot(T_values, pMpt_yita_1_0, label=label+', '+r'$\gamma=1.0$', color=color_map[index], linestyle='--')
        plt.plot(T_values, pMpt_Jeans, label=label+', Jeans', color=color_map[index], linestyle=':')
    plt.yscale('log')
    plt.ylim(1e-20, 1e20)
    plt.xlabel('Temperature (K)')
    plt.ylabel(r'Mass Loss Rate $\dot{M}$ (kg s$^{-1}$)')
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    # custom legend to show the line styles for yita and Jeans escape
    # list the lines in a table-like legend, each row is a different r, each column is a different yita or Jeans escape.
    # the table title in each column is yita=1.2, yita=1.0, Jeans escape. and in each row is Vesta, Moon, Mars, 1.5 Mars.
    from matplotlib.lines import Line2D
    line_styles = ['-.','-', '--', ':']
    column_labels = [r'$\gamma=1.4$',r'$\gamma=1.2$', r'$\gamma=1.0$', 'Jeans']
    row_labels = list(r_labels.keys())
    n_rows = len(row_labels)
    n_cols = len(line_styles)

    # 1. 构造原始矩阵 (n_rows x n_cols)
    # 每一行代表一个行星，每一列代表一种模型
    handle_matrix = []
    for r_idx, label in enumerate(row_labels):
        row_handles = []
        for ls in line_styles:
            line = Line2D([0], [0], color=color_map[r_idx], linestyle=ls)
            row_handles.append(line)
        handle_matrix.append(row_handles)

    # 2. 关键：将矩阵“展平”为 Matplotlib 纵向填充所需的顺序
    # Matplotlib 在 ncol=3 时，会先填满第 1 列的所有行，再填第 2 列...
    # 所以我们需要按列读取：[R1C1, R2C1, R3C1, R4C1, R1C2, R2C2, ...]
    flat_handles = []
    for c in range(n_cols):
        for r in range(n_rows):
            flat_handles.append(handle_matrix[r][c])

    # 3. 构造对应的标签列表（同样需要匹配纵向顺序）
    # 只有在最后一列（c=2）的时候才显示行星名字
    flat_labels = []
    for c in range(n_cols):
        for r in range(n_rows):
            if c == n_cols - 1: # 最后一列
                flat_labels.append(row_labels[r])
            else:
                flat_labels.append('')

    # 4. 绘制图例
    legend = plt.legend(flat_handles, flat_labels, 
                        ncol=n_cols, loc='lower right', frameon=False,
                        columnspacing=1.5, handletextpad=0.5)

    # 5. 获取 Handle 坐标并放置列标题
    plt.draw()
    inv = plt.gca().transData.inverted()
    
    header_indices = [0, n_rows, 2 * n_rows, 3 * n_rows]
    
    for i, idx in enumerate(header_indices):
        # 获取该 handle 的像素位置并转为数据坐标
        bbox_pixel = legend.legend_handles[idx].get_window_extent() # type: ignore
        bbox_data = inv.transform(bbox_pixel)
        
        x_center = (bbox_data[0][0] + bbox_data[1][0]) / 2
        legend_bbox = inv.transform(legend.get_window_extent())
        y_top = legend_bbox[1][1]
        
        plt.text(x_center, y_top, column_labels[i], 
                 ha='center', va='bottom')
    plt.savefig('pMpt_comparison.pdf')

def vi_pMpt_gamma():
    """
    Test the pMpt vs r with different yita, and set T = 1500 and 2500 K. r from 1e5 to 1e7 m in log scale. 
    Plot the pMpt vs r with different yita in different line styles, while different T in different colors.
    """
    import matplotlib.pyplot as plt
    r_values = np.logspace(5, 7, 100)
    yita_values = [1.2, 1.005, 1.0, 1.0]
    T_values = [1500, 2500]
    linestyles = [ '-', '--', ':', '-.']
    exact_gamma_1 = [ None, None, True, False]
    for T in T_values:
        for iyita, yita in enumerate(yita_values):
            if exact_gamma_1[iyita] is None:
                pMpt_values = np.array([pMpt(T, MagmaOceanParameters(r=r), yita=yita) for r in r_values])
                plt.plot(r_values/1e6, pMpt_values, label=f'T={T}K, '+r'$\gamma=$'+f'{yita}', linestyle=linestyles[iyita], color='C0' if T==1500 else 'C1')
            elif exact_gamma_1[iyita] is True:
                pMpt_values = np.array([pMpt(T, MagmaOceanParameters(r=r), yita=1.0,exact_gamma_1=True) for r in r_values])
                plt.plot(r_values/1e6, pMpt_values, label=f'T={T}K, '+r'$\gamma=1.0$ (exact)', linestyle=linestyles[iyita], color='C0' if T==1500 else 'C1')
            else:
                pass
                #pMpt_values = np.array([pMpt(T, MagmaOceanParameters(r=r), yita=1.0,exact_gamma_1=False) for r in r_values])
                #plt.plot(r_values/1e6, pMpt_values, label=f'T={T}K, '+r'$\gamma=1.0$ (estimated)', linestyle=linestyles[iyita], color='C0' if T==1500 else 'C1')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-20, 1e15)
    plt.xlabel(r'Planet Radius ($10^6$ m)')
    plt.ylabel(r'Mass Loss Rate $\dot{M}$ (kg s$^{-1}$)')
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig('pMpt_gamma_comparison_fix_isothermal.pdf')

def vi_isothermal_devol():
    """
    The x-axis is the r from 1e5 to 1e7 m in log scale, and the y-axis is the final concentration ratio after devolatilization with T_init = 2000K, 2500K, 3000K, and T_end = 1200K.
    """
    import matplotlib.pyplot as plt
    r_values = np.logspace(5, 7, 80)
    T_inits = [2000, 2500, 3000]
    cmap = ['C0', 'C1', 'C2']
    for i, T_init in enumerate(T_inits):
        C_final_values = []
        C_final_values_gamma_non_1 = []
        C_final_values_reaching_1 = []
        for r in r_values:
            params = MagmaOceanParameters(r=r)
            M_l = params.M_tot * 0.7 * 0.3
            _, _, _, C_final = devoltilization(T_init, M_l, params, final_only=True, yita=1.2, hydrodynamic_only=True)
            C_final_values.append(C_final)
            _, _, _, C_final_gamma_non_1 = devoltilization(T_init, M_l, params, final_only=True, yita=1.4, hydrodynamic_only=True)
            C_final_values_gamma_non_1.append(C_final_gamma_non_1)
            _, _, _, C_final_reaching_1 = devoltilization(T_init, M_l, params, final_only=True, yita=1.005, hydrodynamic_only=True)
            C_final_values_reaching_1.append(C_final_reaching_1)
        plt.plot(r_values/1e6, C_final_values, label=r'$T_{init}$='+f'{T_init}K, '+ r'$\gamma=1.2$', color=cmap[i])
        plt.plot(r_values/1e6, C_final_values_gamma_non_1, label=r'$T_{init}$='+f'{T_init}K, '+ r'$\gamma=1.4$', linestyle='--', color=cmap[i])
        plt.plot(r_values/1e6, C_final_values_reaching_1, label=r'$T_{init}$='+f'{T_init}K, '+ r'$\gamma=1.005$', linestyle=':', color=cmap[i])
    plt.xscale('log')
    plt.xlim(1e5/1e6, 5e7/1e6)
    plt.xlabel(r'Planet Radius ($10^6$ m)')
    plt.ylabel('Final C / $C_0$')
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.legend(loc = 'lower right')
    plt.savefig('isothermal_devolatilization.pdf')

def test_cooling_with_jeans():
    """
    For an r=1e6 m planet, compare the evoluations of T, C, and M_loss with time by only considering hydrodynamic or take Jeans escape into account.
    T_int = 3000K, M_l = 0.7 * M_tot, other parameters are default. Plot the evolutions of T, C, and M_loss with time in a log scale.
    T, C, M_loss/M_tot plot in different subfigures, and in each subfig, the hydrodynamic only and Jeans escape cases use different color, and yita=1.2 and 1.0 for different line styles.
    """
    import matplotlib.pyplot as plt
    T_init = 2500
    params = MagmaOceanParameters(r=1.7e6)
    plt.figure(figsize=(18,5))
    # set subplot
    ax_T = plt.subplot(1,3,1)
    ax_C = plt.subplot(1,3,2)
    ax_M_loss = plt.subplot(1,3,3)
    data_dict = {}
    for hyrodynamic_only in [True, False]:
        for yita in [1.2, 1.0]:
            T_array, t_total_array, M_loss_array, C_array = devoltilization(T_init, params.M_tot * 0.7 * 0.3, params, final_only=False, yita=yita, hydrodynamic_only=hyrodynamic_only)
            data_dict[(hyrodynamic_only, yita)] = (T_array, t_total_array, np.array(M_loss_array)/(params.M_tot * 0.7 * 0.3), C_array)
    # plot T, C, M_loss/M_tot in different subfigures
    for i, key in enumerate(data_dict.keys()):
        hyrodynamic_only, yita = key
        T_array, t_total_array, M_loss_array, C_array = data_dict[key]
        label = r'$\gamma$='+format_float(yita)
        label += '' if hyrodynamic_only else ' with Jeans escape'
        ax_T.plot(np.array(t_total_array)/365.25/24/3600, T_array, label=label, color='C0' if yita==1.2 else 'C1', linestyle ='-' if hyrodynamic_only else '--')
        ax_C.plot(np.array(t_total_array)/365.25/24/3600, C_array, label=label, color='C0' if yita==1.2 else 'C1', linestyle ='-' if hyrodynamic_only else '--')
        ax_M_loss.plot(np.array(t_total_array)/365.25/24/3600, M_loss_array, label=label, color='C0' if yita==1.2 else 'C1', linestyle ='-' if hyrodynamic_only else '--')
    ax_T.set_xscale('log')
    ax_T.set_xlabel('Time (year)')
    ax_T.set_ylabel('Temperature (K)')
    ax_T.legend()
    ax_T.grid(color='white', linestyle='--', linewidth=0.5)
    ax_C.set_xscale('log')
    ax_C.set_xlabel('Time (year)')
    ax_C.set_ylabel('Concentration Ratio')
    #ax_C.legend()
    ax_C.grid(color='white', linestyle='--', linewidth=0.5)
    ax_M_loss.set_xscale('log')
    ax_M_loss.set_xlabel('Time (year)')
    ax_M_loss.set_ylabel(r'Mass Loss / Total Mass')
    #ax_M_loss.legend()
    ax_M_loss.grid(color='white', linestyle='--', linewidth=0.5)
    # put an r = 1.7e6 m label in the second subfig
    ax_C.text(0.15, 0.15, 'r=' + format_float(params.r, significant_digits=2, max_decimal_places=1)+' m', transform=ax_C.transAxes, fontsize=12)
    plt.tight_layout()
    plt.savefig('cooling_with_jeans.pdf')

def vi_cooling_processes():
    """
    Plot the cooling processes of a magma ocean with T_init = 3000K, M_l = 0.7 * M_tot, r = 1.7e6 m.
    The x-axis is time in log scale, and the y-axis is the temperature.
    Plot different gamma in different color lines.
    """
    import matplotlib.pyplot as plt
    T_init = 3500
    r = 1e6
    params = MagmaOceanParameters(r=r)
    plt.figure(figsize=(6,5))
    for yita in [1.4, 1.2, 1.05]:
        T_array, t_total_array, M_loss_array, C_array = devoltilization(T_init, params.M_tot * 0.7, params, final_only=False, yita=yita)
        label = r'$\gamma$='+format_float(yita)
        plt.plot(np.array(t_total_array)/365.25/24/3600, T_array, label=label)
    plt.text(0.15, 0.15, 'r=' + format_float(params.r, significant_digits=2, max_decimal_places=1)+' m', transform=plt.gca().transAxes, fontsize=12)
    #plt.xscale('log')
    plt.xlabel('Time (year)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.grid(color='white', linestyle='--', linewidth=0.5)
    plt.savefig('cooling_processes.pdf')

def figure_Calogero_2025_figure2_ac():
    """
    Reproduce the figure 2a to 2c in Calogero et al. (2025).
    The x-axis is the M_total, from 10^21 - 10^23 kg in log scale.
    The y-axis is cooling time, body mass fraction loss, and K fraction lost.
    They assume the total mantle (0.7*M_total) is molten, 
        and the initial temperature is set as 1500, 2000, 2500, 3000, 3500 K.
        Their final temperature is set as 1400 K.
    """
    import matplotlib.pyplot as plt
    M_total_values = np.logspace(21, 24, 40)
    T_inits = [1500, 2000, 2500, 3000, 3500]
    cooling_times = np.zeros((len(T_inits), len(M_total_values)))
    mass_loss_fractions = np.zeros_like(cooling_times)
    K_loss_fractions = np.zeros_like(cooling_times)

    # convert M_total to radius and M_l.
    radius_values = (3 * M_total_values / (4 * math.pi * 3500)) ** (1/3)
    M_l = 0.7 * M_total_values 

    for i, radius_values in enumerate(radius_values):
        params = MagmaOceanParameters(r=radius_values)
        for j, T_init in enumerate(T_inits):
            _, cooling_time, mass_loss, C_final = devoltilization(T_init, M_l[i], params, final_only=True, T_end=1400, yita=1.2)
            cooling_times[j,i] = cooling_time / 365.25/24/3600
            mass_loss_fractions[j,i] = mass_loss / M_total_values[i]
            K_loss_fractions[j,i] = 1 - C_final
    # plot the results in three subfigures
    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1)
    for i, T_init in enumerate(T_inits):
        plt.plot(M_total_values, cooling_times[i,:], label=r'$T_{init}$='+f'{T_init}K')
    plt.xlim(1e21, 1.33e23)
    plt.ylim(0, 2000)
    plt.xscale('log')
    plt.xlabel(r'Total Mass $M_{total}$ (kg)')
    plt.ylabel('Cooling Time (year)')
    plt.legend()
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.subplot(1,3,2)
    for i, T_init in enumerate(T_inits):
        plt.plot(M_total_values, mass_loss_fractions[i,:], label=r'$T_{init}$='+f'{T_init}K')
    plt.xscale('log')
    plt.xlim(1e21, 1.33e23)
    plt.ylim(0,0.25)
    plt.xlabel(r'Total Mass $M_{total}$ (kg)')
    plt.ylabel('Mass Loss Fraction')
    plt.legend()
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.subplot(1,3,3)
    for i, T_init in enumerate(T_inits):
        plt.plot(M_total_values, K_loss_fractions[i,:], label=r'$T_{init}$='+f'{T_init}K')
    plt.xscale('log')
    plt.xlim(1e21, 1.33e23)
    plt.ylim(0,1)
    plt.xlabel(r'Total Mass $M_{total}$ (kg)')
    plt.ylabel('K Loss Fraction')
    plt.legend()
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('Calogero_2025_figure2_ac.pdf')

def simple_collision(M_total: float, # in kg
                     v_rel: float, # in v_esc
                     impactor_angle: float, # in degree, 0 for head-on collision, 90 for grazing collision
                     gamma: float, # ratio of impactor to total mass.
                     rho_B: float = 3500, # in kg m-3, the density of the bulk material.
                     rho_M: float = 3000, # in kg m-3, the density of the mantle.
                     T_int: float = 1200, # in K
                     heat_ratio: float = 1.0,
                     C_p: float = 1200, # in J kg-1 K-1, the specific heat capacity of the material.
                     mantle_fraction: float = 0.7, # the mass fraction of the mantle in the final body.
                     ) -> Tuple[float,float]:
    """
    A simple collision and melt model to estimate the result (melt fraction and peak temperature) of a collision with given total mass and relative velocity.
    I assumed the kinetic energy on the head-on direction of the collision is all (or multi a heat_ratio factor) converted to heat the final body, and the final body has a uniform temperature distribution.
    The melt fraction is estimated by comparing the final temperature with the melting temperature, which is calculated by Rubie et al., (2015) melt model, with the pressure estimated by the gravity of the final body.
    The peak temperature is estimated by the final temperature after the collision.
    """
    def _H_mantle(rho_B, rho_M, M_total, mantle_fraction) -> Tuple[float, float, float,float]:
        """
        calculate the height of the mantle by assuming the body is a sphere with bulk density rho_B, and the mantle has a density of rho_M and mass fraction of mantle_fraction.
        return:
            R_total: the radius of the final body in m.
            r_core: the radius of the core in m.
            H_mantle: the height of the mantle in m.
        """
        V_total = M_total / rho_B
        R_total = (3 * V_total / (4 * math.pi)) ** (1/3)
        M_mantle = M_total * mantle_fraction
        V_mantle = M_mantle / rho_M
        V_core = V_total - V_mantle
        r_core = (3 * V_core / (4 * math.pi)) ** (1/3)
        H_mantle = R_total - r_core
        M_core = M_total - M_mantle
        return R_total, r_core, M_core, H_mantle
    def _melt_frac(T_final, R_total, r_core, M_core, H_mantle, rho_M):
        """
        calculate the melt fraction by comparing the final temperature with the melting temperature, which is calculated by Rubie et al., (2015) melt model, with the pressure estimated by the gravity of the final body.
        The pressure is estimated by the gravity of the final body, which is calculated by the mass and radius of the final body.
        The melting temperature is calculated by Rubie et al., (2015) melt model, which is a function of pressure.
        return:
            melt_fraction: the melt fraction of the mantle after the collision.
        """
        r = np.linspace(r_core, R_total, 1000)
        dr = r[1] - r[0]
        M_r = M_core + (4/3) * math.pi * rho_M * (r**3 - r_core**3)
        g_r = G * M_r / r**2
        # calculate the pressure by integrating the gravity from the surface to the center.
        P_r = np.zeros_like(r) # in Pa
        for i in reversed(range(len(r)-1)):
            dr = r[i+1] - r[i]
            P_r[i] = P_r[i+1] + rho_M * g_r[i] * dr
        #print(P_r/1e9) # in GPa
        # calculate the melting temperature by Rubie et al., (2015) melt model, which is a function of pressure.
        T_melt_r = np.where(P_r * 1e-9 < 24, 
                         (1874.0 + 55.43 * P_r * 1e-9 - 1.74 * (P_r * 1e-9)**2.0  + 0.0193 * (P_r * 1e-9)**3.0),
                         (1249.0 + 58.28 * P_r * 1e-9 - 0.395 * (P_r * 1e-9)**2.0  + 0.011 * (P_r * 1e-9)**3.0))
        #print(T_melt_r) # in K
        melt_indexes = np.where(T_final > T_melt_r)[0]
        if len(melt_indexes) == 0:
            return 0.0
        else:
            return 4 * math.pi * np.sum(r[melt_indexes]**2 * dr) / (4/3 * math.pi * (R_total**3 - r_core**3))
    M_i = M_total * gamma
    M_t = M_total - M_i
    R_total, r_core, M_core, H_mantle = _H_mantle(rho_B, rho_M, M_total, mantle_fraction)
    v_esc = math.sqrt(2 * G * M_total / R_total)
    v_rel_m_s = v_rel * v_esc
    v_rel_m_s_head_on = v_rel_m_s * math.cos(math.radians(impactor_angle))
    kinetic_energy = 0.5 * (M_i*M_t/(M_i+M_t)) * v_rel_m_s_head_on**2
    T_final = heat_ratio * kinetic_energy / (M_total * C_p) + T_int
    melt_fraction = _melt_frac(T_final, R_total, r_core, M_core, H_mantle, rho_M)
    return melt_fraction, T_final

def contour_lower_simple():
    """
    Make a contour plot of the melt fraction in the space of total mass and relative velocity, with fixed impact angle of 0 degree and gamma of 0.5.
    The total mass is from 10^21 to 10^23 kg in log scale, and the relative velocity is from 0.9 to 10 in v_esc.
    """
    import matplotlib.pyplot as plt
    M_total_values = np.logspace(21, 23, 30)
    v_rel_values = np.linspace(0.9, 10, 30)
    melt_fraction_mesh = np.zeros((len(M_total_values), len(v_rel_values)))
    peak_temperature_mesh = np.zeros_like(melt_fraction_mesh)
    for i, M_total in enumerate(M_total_values):
        for j, v_rel in enumerate(v_rel_values):
            melt_fraction_mesh[i,j], peak_temperature_mesh[i,j] = simple_collision(M_total, v_rel, impactor_angle=0, gamma=0.5)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    contour = plt.contourf(M_total_values, v_rel_values, melt_fraction_mesh.T, levels=20, cmap='viridis')
    plt.xscale('log')
    plt.xlabel(r'Total Mass $M_{total}$ (kg)')
    plt.ylabel(r'Relative Velocity $v_{rel}$ ($v_{esc}$)')
    plt.yscale('log')
    plt.colorbar(contour, label='Melt Fraction')
    plt.title(r'0°, $\gamma$= 0.5')
    plt.subplot(1,2,2)
    # clip the peak temperature for max 10000 K to make the contour more visible.
    peak_temperature_mesh = np.clip(peak_temperature_mesh, None, 10000)
    contour = plt.contourf(M_total_values, v_rel_values, peak_temperature_mesh.T, levels=20, cmap='viridis')
    # 将colorbar 中10000K 的标签改为 '>10000K'
    cbar = plt.colorbar(contour, label='Peak Temperature (K)')
    cbar.set_ticks(list(cbar.get_ticks())[:-1] + [10000])
    cbar.set_ticklabels(list(label.get_text() for label in cbar.ax.get_yticklabels())[:-1] + ['>10000'])
    plt.xscale('log')
    plt.xlabel(r'Total Mass $M_{total}$ (kg)')
    plt.ylabel(r'Relative Velocity $v_{rel}$ ($v_{esc}$)')
    plt.yscale('log')
    
    plt.title(r'0°, $\gamma$= 0.5')
    plt.tight_layout()
    plt.savefig('simple_collision_contour.pdf')

def vi_effect_gamma():
    """
    test the effect of gamma (in atmosphere context) on the threshold of devol.
    plot x-axis as gamma, from 1.5 to 1.0, and y-axis as the largest r required to lose at leat 50% of the K, and temperatures of 2000K, 2500K, 3000K, 3500K in different lines.    
    """
    import matplotlib.pyplot as plt
    gamma_values = np.linspace(1.5, 1.0, 20)
    T_values = [2500, 3000, 3500]
    r_thresholds = np.zeros((len(T_values), len(gamma_values)))
    r_try_value_min = 1.6e5
    r_try_value_max = 5e7
    for i, gamma in enumerate(gamma_values):
        for j, T in enumerate(T_values):
            # obtain the r_threshold by binary search. The when r is smaller than r_threshold, the K loss fraction is larger than 0.5, otherwise, the K loss fraction is smaller than 0.5.
            # test if r_try_value_min can cause more than 50% K loss, if not, the r_threshold is smaller than r_try_value_min. if r_try_value_max can cause less than 50% K loss, then the r_threshold is larger than r_try_value_max. Otherwise, do binary search between r_try_value_min and r_try_value_max to find the r_threshold.
            params = MagmaOceanParameters(r=r_try_value_min)
            _, _, _, C_final = devoltilization(T, params.M_tot * 0.7 * 0.3, params, final_only=True,yita=gamma)
            if C_final > 0.5:
                r_thresholds[j,i] = np.nan
                print(f'gamma={gamma:.2f}, T={T}K, r_threshold < {r_try_value_min:.2e} m, C_final={C_final:.2f} at r={r_try_value_min:.2e} m')
                continue
            params = MagmaOceanParameters(r=r_try_value_max)
            _, _, _, C_final = devoltilization(T, params.M_tot * 0.7 * 0.3, params, final_only=True,yita=gamma)
            if C_final < 0.5:
                r_thresholds[j,i] = np.nan
                print(f'gamma={gamma:.2f}, T={T}K, r_threshold > {r_try_value_max:.2e} m, C_final={C_final:.2f} at r={r_try_value_max:.2e} m')
                continue
            # binary search
            r_max = r_try_value_max
            r_min = r_try_value_min
            while r_max - r_min > 1e3:
                r_try_value_mid = (r_max + r_min) / 2
                params = MagmaOceanParameters(r=r_try_value_mid)
                _, _, _, C_final = devoltilization(T, params.M_tot * 0.7 * 0.3, params, final_only=True,yita=gamma)
                if C_final > 0.5:
                    r_max = r_try_value_mid
                else:
                    r_min = r_try_value_mid
            r_thresholds[j,i] = (r_max + r_min) / 2
            print(f'gamma={gamma:.2f}, T={T}K, r_threshold={r_thresholds[j,i]:.2e} m')
    for j, T in enumerate(T_values):
        plt.plot(gamma_values, r_thresholds[j,:], label=f'T={T}K')
    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'largest r to lost 50% K (m)')
    plt.ylim(1e5, 1e7)
    plt.yscale('log')
    # add y-axis ticks at Earth radius, Mars radius, Moon radius, and Vesta radius, and label them.
    original_yticks = plt.yticks()[0]
    original_ylabels = plt.yticks()[1]
    new_yticks = [2.6e5, 1.7e6, 3.4e6, 6.4e6]
    new_ytick_labels = ['Vesta', 'Moon', 'Mars', 'Earth']
    plt.yticks(list(original_yticks) + new_yticks, list(original_ylabels) + new_ytick_labels) # type: ignore
    plt.ylim(1e5, 1e7)
    plt.legend()
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.savefig('effect_gamma.pdf')

def vi_effect_D():
    """
    In the above figure, add the effect of D.
    Set T = 2500 K, gamma = 1.01, 1.2, 1.4 in different line colors. and D = original * e^(shifte), where shifte ranges [-1, 2] as x-axis.
    Calculate the r_threshold for each combination of gamma and D, and plot the r_threshold vs gamma with different D in different line colors.
    """
    import matplotlib.pyplot as plt
    T = 3000
    gamma_values = [1.4, 1.2, 1.01]
    shifte_value = np.linspace(-2, 4, 20)
    r_thresholds = np.zeros((len(gamma_values), len(shifte_value)))
    r_try_value_min = 1.0e5
    r_try_value_max = 5e7
    for i, gamma in enumerate(gamma_values):
        for j, shifte in enumerate(shifte_value):
            # obtain the r_threshold by binary search. The when r is smaller than r_threshold, the K loss fraction is larger than 0.5, otherwise, the K loss fraction is smaller than 0.5.
            # test if r_try_value_min can cause more than 50% K loss, if not, the r_threshold is smaller than r_try_value_min. if r_try_value_max can cause less than 50% K loss, then the r_threshold is larger than r_try_value_max. Otherwise, do binary search between r_try_value_min and r_try_value_max to find the r_threshold.
            params = MagmaOceanParameters(r=r_try_value_min)
            _, _, _, C_final = devoltilization(T, params.M_tot * 0.7 * 0.3, params, final_only=True,yita=gamma, shifte=shifte)
            if C_final > 0.5:
                r_thresholds[i,j] = np.nan
                print(f'shifte={shifte:.2f}, gamma={gamma:.3f}, T={T}K, r_threshold < {r_try_value_min:.2e} m, C_final={C_final:.2f} at r={r_try_value_min:.2e} m')
                continue
            params = MagmaOceanParameters(r=r_try_value_max)
            _, _, _, C_final = devoltilization(T, params.M_tot * 0.7 * 0.3, params, final_only=True,yita=gamma, shifte=shifte)
            if C_final < 0.5:
                r_thresholds[i,j] = np.nan
                print(f'shifte={shifte:.2f}, gamma={gamma:.3f}, T={T}K, r_threshold > {r_try_value_max:.2e} m, C_final={C_final:.2f} at r={r_try_value_max:.2e} m')
                continue
            # binary search
            r_max = r_try_value_max
            r_min = r_try_value_min
            while r_max - r_min > 1e3:
                r_try_value_mid = (r_max + r_min) / 2
                params = MagmaOceanParameters(r=r_try_value_mid)
                _, _, _, C_final = devoltilization(T, params.M_tot * 0.7 * 0.3, params, final_only=True,yita=gamma, shifte=shifte)
                if C_final > 0.5:
                    r_max = r_try_value_mid
                else:
                    r_min = r_try_value_mid
            r_thresholds[i,j] = (r_max + r_min) / 2
            print(f'shifte={shifte:.2f}, gamma={gamma:.3f}, T={T}K, r_threshold={r_thresholds[i,j]:.2e} m')
    for i, gamma in enumerate(gamma_values):
        plt.plot(shifte_value, r_thresholds[i,:], label=r'$\gamma=$'+format_float(gamma), color='C'+str(i))
    # add a text to indicate the T_init = 3000 K for the figure.
    plt.text(0.15, 0.85, r'$T_{init}$=3000 K', transform=plt.gca().transAxes, fontsize=12)
    plt.xlabel(r'x, where D = $D_{original}$ * $e^{x}$')
    plt.ylabel(r'largest r to lost 50% K (m)')
    plt.ylim(1e5, 1e7)
    plt.yscale('log')
    # add y-axis ticks at Earth radius, Mars radius, Moon radius, and Vesta radius, and label them.
    original_yticks = plt.yticks()[0]
    original_ylabels = plt.yticks()[1]
    new_yticks = [2.6e5, 1.7e6, 3.4e6, 6.4e6]
    new_ytick_labels = ['Vesta', 'Moon', 'Mars', 'Earth']
    plt.yticks(list(original_yticks) + new_yticks, list(original_ylabels) + new_ytick_labels) # type: ignore
    plt.ylim(1e5, 1e7)
    plt.legend()
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.savefig('effect_D.pdf')

def double_check_eqn1():
    param = MagmaOceanParameters(r=1e6)
    print(T_s_from_T(1400,param))


if __name__ == "__main__":
    #main()
    #print(vapour_pressure(5000))
    #vi_vapour_pressure()
    #vi_M_critical()
    #vi_effect_blackbody()
    # For the convenience of further testing, add a command args to chose which test to run.
    # And the args can be 'main', 'vapour_pressure', 'M_critical', 'effect_blackbody', but should not
    # hardcode them, instead, any args with the same function name can be used to run the test.
    import argparse
    parser = argparse.ArgumentParser(description='Test the magma ocean devolatilization model.')
    parser.add_argument('test', type=str, help='The test to run. Can be main, vapour_pressure, M_critical, effect_blackbody, ...')
    args = parser.parse_args()
    test_name = args.test
    # find the function with the same name as test_name, and run it.
    # if the function does not exist, print an error message.
    if test_name in globals() and callable(globals()[test_name]):
        globals()[test_name]()
    else:
        print(f'Error: test {test_name} does not exist.') 