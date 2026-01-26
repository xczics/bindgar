"""
In this module, we will handle the devoltilization during a single collision event.
"""

from typing import Dict, Optional
import numpy as np
from ..physics import convert_factor_from_auptu_to_kmps, AU2M, M_EARTH, M_SUN, G, M_Mars
import math
from functools import cached_property
from .nakajima.melt_model import Model
from .magma_cooling import MagmaOceanParameters,devoltilization

class CollisionEvent():
    """
    A class to handle the single collison data.
    properties:
    target: int
        The index of the target body.
    impactor: int
        The index of the impactor body.
    total_mass: float
        The total mass of the colliding system in kg.
    total_mass_in_earth: float
        The total mass of the colliding system in Earth mass.
    gamma: float
        The mass ratio of the impactor to the total mass.
    delv: np.ndarray
        The relative velocity vector at impact in m/s [impactor - target].
    delx: np.ndarray
        The relative position vector at impact in m [impactor - target].
    impact_angle_radian: float
        The impact angle in radian.
    impact_angle: float
        The impact angle in degree.
    rho_B: float
        The bulk density of the target in kg/m3.
    radius: float
        The radius of the merged body after collision in m, will calculate from the total mass and bulk density.
    escape_velocity: float
        The escape velocity of the merged body after collision in m/s.
    vel_real: float
        The real impact velocity in m/s.
    vel_escape_ratio: float
        The ratio of the real impact velocity to the escape velocity.
    """
    target: int # The index of the target body.
    impactor: int # The index of the impactor body.
    total_mass_sun: float # The total mass of the colliding system in SUN mass.
    gamma: float # The mass ratio of the impactor to the total mass.
    delv: np.ndarray # The relative velocity vector at impact in m/s [impactor - target].
    delx: np.ndarray # The relative position vector at impact in m [impactor - target].
    rho_B: float # The bulk density of the target in kg/m3.

    def __init__(self,collision: Dict, **kwargs) -> None:
        collision_handle_kwargs = {}
        if "rho_B" in kwargs:
            collision_handle_kwargs["rho_B"] = kwargs["rho_B"]
        self._handle_collision_data(collision, **collision_handle_kwargs)
        magma_produce_keys = ["entropy0","outputfigurename","use_tex","silent"]
        self._magma_produce_kwargs = {key: kwargs[key] for key in magma_produce_keys if key in kwargs}
        cooling_keys = ["rho_B","rho_M","kapa","v" ,"k","C_p","alpha_V","g_s","r","L","M_mol"]
        self._cooling_kwargs = {key: kwargs[key] for key in cooling_keys if key in kwargs}

    def _handle_collision_data(self,collision: Dict,
                             rho_B: Optional[float] = None) -> None:
        if collision["mi"] >= collision["mj"]:
            target_key, impactor_key = "i", "j"
        else:
            target_key, impactor_key = "j", "i"
        target_index = collision[f"index{target_key}"]
        impactor_index = collision[f"index{impactor_key}"]
        target_x = np.array([collision[f"x{target_key}"], collision[f"y{target_key}"], collision[f"z{target_key}"]])
        impactor_x = np.array([collision[f"x{impactor_key}"], collision[f"y{impactor_key}"], collision[f"z{impactor_key}"]])
        target_v = np.array([collision[f"vx{target_key}"], collision[f"vy{target_key}"], collision[f"vz{target_key}"]])
        impactor_v = np.array([collision[f"vx{impactor_key}"], collision[f"vy{impactor_key}"], collision[f"vz{impactor_key}"]])
        target_mass = collision[f"m{target_key}"]
        impactor_mass = collision[f"m{impactor_key}"]
        target_radius = collision[f"r{target_key}"]
        #impactor_radius = collision[f"r{impactor_key}"]
        self.target = target_index
        self.impactor = impactor_index
        self.total_mass_sun = (target_mass + impactor_mass) 
        self.gamma = impactor_mass / self.total_mass_sun
        self.delv =( impactor_v - target_v ) * convert_factor_from_auptu_to_kmps * 1e3  # in m/s
        self.delx = ( impactor_x - target_x ) * AU2M   # in m
        if rho_B is not None:
            self.rho_B = rho_B
        else:
            # calculate bulk density in kg/m3, from the mass and volume of the target.
            volume = (4/3) * np.pi * (target_radius * AU2M)**3
            self.rho_B = target_mass * M_SUN / volume

    def _magma_init(self, **kwargs) -> Model:
        Mtotal = self.total_mass / M_Mars  # in Mars mass
        gamma = self.gamma
        vel = self.vel_escape_ratio
        # round agnle to nearest choice: 0.0, 30.0, 45.0, 60.0, 90.0
        if self.impact_angle >= 60.0 and self.impact_angle < 75.0:
            impact_angle = 60
        elif self.impact_angle >= 75.0 and self.impact_angle <= 90.0:
            impact_angle = 90
        elif self.impact_angle >=0 and self.impact_angle < 15.0:
            impact_angle = 0
        elif self.impact_angle >=15.0 and self.impact_angle < 30:
            impact_angle = 30
        else:
            impact_angle = round(self.impact_angle / 15.0) * 15
        return Model(Mtotal=Mtotal, gamma=gamma, vel=vel, impact_angle=impact_angle, **kwargs)

    def _cooling_init(self, **kwargs):
        if "rho_B" not in kwargs:
            kwargs["rho_B"] = self.rho_B
        if "r" not in kwargs:
            kwargs["r"] = self.radius
        return MagmaOceanParameters(**kwargs)

    def melt_T_increase(self, 
                        C_p:float = 1200 # J/kg/K 
                        ) -> float:
        delta_T = self.av_du_gain * 1e5 / C_p
        return delta_T
    
    def devoltilize(self,T_0: float=1200, **kwargs) -> tuple:
        cooling_params = self.cooling_params
        #print("Cooling parameters:", cooling_params)
        melt_mass = self.melt_mass
        delta_T = self.melt_T_increase(C_p=cooling_params.C_p)
        if "T_end" not in kwargs:
            kwargs["T_end"] = T_0
        result = devoltilization(T_init = T_0+delta_T, M_l_init = melt_mass, params = cooling_params, **kwargs)
        return result
    @cached_property
    def av_du_gain(self) -> float:
        return float(np.mean(self.du_gain[self.melt_label==1]))
    @cached_property
    def cooling_params(self) -> MagmaOceanParameters:
        return self._cooling_init(**self._cooling_kwargs)
    @cached_property
    def melt_model(self) -> Model:
        return self._magma_init(**self._magma_produce_kwargs)
    @cached_property
    def collision_result(self) -> Dict:
        return self.melt_model.run_model()
    @cached_property
    def melt_fraction(self) -> float:
        d = self.collision_result
        return d["melt fraction"]
    @cached_property
    def melt_mass(self) -> float:
        f_mantle = 0.7
        return f_mantle * self.melt_fraction * self.total_mass
    @cached_property
    def du_gain(self) -> np.ndarray:
        d = self.collision_result
        return d["internal energy gain"]
    @cached_property
    def melt_label(self) -> np.ndarray:
        d = self.collision_result
        return d["internal energy of the melt (considering initial temperature profile)"] 
    @cached_property
    def total_mass(self) -> float:
        """The total mass of the colliding system in kg."""
        return self.total_mass_sun * M_SUN
    @cached_property
    def total_mass_in_earth(self) -> float:
        """The total mass of the colliding system in Earth mass."""
        return self.total_mass_sun * M_SUN / M_EARTH
    @cached_property
    def impact_angle_radian(self) -> float:
        """The impact angle in radian."""
        del_v_norm = self.delv / np.linalg.norm(self.delv)
        del_x_norm = self.delx / np.linalg.norm(self.delx)
        cos_angle = np.clip( np.dot( del_v_norm, del_x_norm ), -1.0, 1.0)
        angle = np.arccos( np.abs(cos_angle) )
        return angle
    @cached_property
    def impact_angle(self) -> float:
        """The impact angle in degree."""
        return self.impact_angle_radian * 180.0 / np.pi  # in degree
    @cached_property
    def radius(self) -> float:
        """The radius of the merged body after collision in m, will calculate from the total mass and bulk density."""
        return ( self.total_mass / ( (4/3) * np.pi * self.rho_B ) )**(1/3)
    @cached_property
    def vel_real(self) -> float:
        """The real impact velocity in m/s."""
        return float(np.linalg.norm(self.delv))
    @cached_property
    def vel_escape_ratio(self) -> float:
        """The ratio of the real impact velocity to the escape velocity."""
        return self.vel_real / self.escape_velocity
    @cached_property
    def escape_velocity(self) -> float:
        """The escape velocity of the merged body after collision in m/s."""
        return math.sqrt( 2 * G * self.total_mass / self.radius )


if __name__ == "__main__":
    example_data = {
        "time":29457364.762859217823,
        "indexi":56,
        "mi":5.1525247341599930334e-08,
        "ri":5.6737661257283551178e-05 * 0.01**(1/3),
        "xi":-0.7262753531598594714,
        "yi":0.58904077113505748375,
        "zi":0.013177829409715336589,
        "vxi":-0.62028148875343291913,
        "vyi":-0.79629294321458765626,
        "vzi":-0.06786287045067085355,
        "Sxi":1.5313816917242437393e-11,
        "Syi":9.5109844617522566775e-12,
        "Szi":1.0567570577917187638e-11,
        "indexj":709,
        "mj":3.3371274299999999601e-09,
        "rj":4.9089544100000003543e-06,
        "xj":-0.72629564935738211151,
        "yj":0.58898671575534899958,
        "zj":0.013199417641283055919,
        "vxj":-0.14452254981221349106,
        "vyj":-0.66877363407722945077,
        "vzj":-0.68650049138567226237,
        "Sxj":0,
        "Syj":0,
        "Szj":0,
    }
    collision_event = CollisionEvent(example_data)
    #print(collision_event.total_mass, collision_event.gamma, collision_event.vel_escape_ratio, collision_event.impact_angle, collision_event.melt_fraction, collision_event.melt_T_increase())
    T, t_total, M_loss_total, C = collision_event.devoltilize()
    print(T, t_total/(3600*24*365), M_loss_total, C)