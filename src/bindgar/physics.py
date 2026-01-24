from math import acos
from typing import Tuple, Sequence, Union, Optional, Iterable
import numpy as np  # type: ignore
from dataclasses import dataclass

# physics constants
M_SUN = 1.98847e30  # kg
M_EARTH = 5.9722e24  # kg
M_Moon = 7.342e22  # kg
M_Mars = 6.4171e23  # kg
AU2M = 1.495978707e11  # meters
PI = 3.141592653589793
convert_factor_from_auptu_to_kmps = 29.8457983 # 1 au/(1/2pi year) to km/s
Stefan_Boltzmann = 5.670374419e-8  # W m-2 K-4
G = 6.67430e-11 # Gravitational Constant, m3 kg-1 s-2
R_gas = 8.314462618 # J mol-1 K-1

def DENSITY2K(density: float) -> float:
    # Convert density (in g/cm^3) to k=(3/(4*pi*rho))^(1/3), where rho is in M_sun/au^3
    density /= (M_SUN*1e3)/(AU2M*1e2)**3  # Convert g/cm^3 to M_sun/au^3
    k = (3/(4*PI*density))**(1/3)
    return k

def K2DENSITY(k: float) -> float:
    # Convert k=(3/(4*pi*rho))^(1/3), where rho is in M_sun/au^3 to density (in g/cm^3)
    density = 3/(4*PI*k**3)  # in M_sun/au^3
    density *= (M_SUN*1e3)/(AU2M*1e2)**3  # Convert M_sun/au^3 to g/cm^3
    return density

def calculate_orbital(x: float, y: float, z: float, vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
    # Calculate semi-major axis (a in au), eccentricity (e), and inclination (inc in radians) from position (x,y,z in au) and velocity (vx,vy,vz in au/day')
    # G = 1, M_sun = 1
    r = (x**2 + y**2 + z**2)**0.5
    v = (vx**2 + vy**2 + vz**2)**0.5
    a = 1/(2/r - v**2)
    h_x = y*vz - z*vy
    h_y = z*vx - x*vz
    h_z = x*vy - y*vx
    h = (h_x**2 + h_y**2 + h_z**2)**0.5
    e = (1 - h**2/a)**0.5
    inc = acos(h_z/h) 
    return a, e, inc

@dataclass
class AnalogCriteria:
    name: str
    a_lower: float
    a_upper: float
    m_lower: float
    m_upper: float
    a_mean: Optional[float] = None
    m_mean: Optional[float] = None
    def is_analog(self, a: float, m: float, m_unit: str = "earth") -> bool:
        if m_unit == "earth":
            return (self.a_lower <= a <= self.a_upper) and (self.m_lower <= m <= self.m_upper)
        elif m_unit == "sun":
            return (self.a_lower <= a <= self.a_upper) and (self.m_lower * M_EARTH / M_SUN <= m <= self.m_upper * M_EARTH / M_SUN)
        else:
            raise ValueError(f"Unknown m_unit {m_unit}.")


DEFAULT_VENUS_ANALOG = AnalogCriteria("Venus", 0.6, 0.9, 0.6, 1.2, a_mean=0.72, m_mean=0.82)
DEFAULT_EARTH_ANALOG = AnalogCriteria("Earth", 0.9, 1.4, 0.8, 1.4, a_mean=1.0, m_mean=1.0)
DEFAULT_MARS_ANALOG = AnalogCriteria("Mars", 1.4, 1.9, 0.01, 0.3, a_mean=1.52, m_mean=0.11)


def stastic_kde(data: np.ndarray,
                x: np.ndarray,
                sigma: Union[float, str] = 'auto',
                auto_sigma_factor: float = 0.5,
                ) -> np.ndarray:
    if isinstance(sigma, str):
        if sigma == 'auto':
            data_std = np.std(data)
            sigma_value = data_std * auto_sigma_factor
        else:
            raise ValueError("Unknown sigma string value")
    else:
        sigma_value = float(sigma)

    out = np.zeros_like(x)
    for d in data:
        out += np.exp(-0.5 * ((x - d) / sigma_value) ** 2) / (sigma * np.sqrt(2 * np.pi))
    out /= len(data)
    return out
