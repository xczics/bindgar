"""
In this module, we will handle the devoltilization during a single collision event.
"""

from typing import Dict, Optional,List, Tuple, Self, Set
import numpy as np
from ..physics import convert_factor_from_auptu_to_kmps, AU2M, M_EARTH, M_SUN, G, M_Mars, impact_angle_radians
import math
from functools import cached_property, lru_cache
from .nakajima.melt_model import Model
from .magma_cooling import MagmaOceanParameters,devoltilization
from ..output import SimulationOutput
from ..common import statstic_time
import os
import pickle

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

    @statstic_time
    def _handle_collision_data(self,collision: Dict,
                             rho_B: Optional[float] = None) -> None:
        if collision["mi"] >= collision["mj"]:
            target_key, impactor_key = "i", "j"
        else:
            target_key, impactor_key = "j", "i"
        self.time = collision["time"]
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

    @statstic_time
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
    @statstic_time
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
        return impact_angle_radians(self.delx, self.delv)
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

class SimulationMeltsEvolution():
    def __init__(self, simulation: SimulationOutput, **kwargs) -> None:
        self._event_args = kwargs
        self.simulation = simulation
        self.simulation.built_history()
        self._melt_fractions_event = None
        self._collision_events : List[CollisionEvent|None] = [None] * len(self.simulation.collisions)
        self._collision_index2particle_evolution_index : List[Tuple[int|None,int|None]] = [(None,None)] * len(self.simulation.collisions)
        self._current_point = 0
        original_path = self.simulation.path
        original_dir = os.path.dirname(original_path)
        melt_cache_file_name = ".meltfrac" + self.simulation.get_input_params("Output name").replace(" ","-") + ".npy"
        evol_cache_file_name = ".meltevol" + self.simulation.get_input_params("Output name").replace(" ","-") + ".pkl"
        self._cache_melt_frac_file_path = os.path.join(original_dir, melt_cache_file_name)
        self._cache_evol_file_path = os.path.join(original_dir, evol_cache_file_name)
        self._new_melt_frac_since_last_cache = 0
        if os.path.exists(self._cache_evol_file_path):
            self._melt_evolution_particles = pickle.load(open(self._cache_evol_file_path, "rb"))
        else:
            self._melt_evolution_particles = {}
        self._evol_cache_updated = True
        self._unexpected_melt_event = set()
    @cached_property
    def final_particles(self) -> List[int]:
        return self.simulation.survivals()
    @cached_property
    def final_particles_set(self) -> Set[int]:
        return self.simulation._survivals
    @property
    def final_particles_without_gass_giants(self) -> List[int]:
        return self.simulation.survivals_without_gas_giants
    @property
    def final_particles_without_gass_giants_set(self) -> Set[int]:
        return self.simulation.survivals_without_gas_giants_sets
    @property
    def target(self) -> int:
        current_event = self.melt_events(self._current_point)
        return current_event.target
    @property
    def impactor(self) -> int:
        current_event = self.melt_events(self._current_point)
        return current_event.impactor
    @property
    def time(self) -> float:
        current_event = self.melt_events(self._current_point)
        return current_event.time
    @property
    def new_melt_frac(self) -> float:
        current_event = self.melt_events(self._current_point)
        return current_event.melt_fraction
    @property
    def accumulated_melt_frac(self) -> float:
        index_particle, evolution_index = self._collision_index2particle_evolution_index[self._current_point]
        if index_particle is not None and evolution_index is not None:
            return self.get_particle_melt_evolution(index_particle)[evolution_index,0]
        else:
            return 0.0
    @property
    def is_beginning(self) -> bool:
        target = self.target
        history = self.simulation.particle_history(target)["history"]
        if history is None or len(history) == 0:
            return True
        if history[0][1] == self._current_point:
            return True
        return False
    @property
    def is_end(self) -> bool:
        target = self.target
        history = self.simulation.particle_history(target)["history"]
        if history is None or len(history) == 0:
            return True
        if history[-1][1] == self._current_point:
            return True
        return False
    @property
    def impator_has_chain(self) -> bool:
        impactor = self.impactor
        return self.particle_has_chain(impactor)
    
    def particle_has_chain(self, particle_index: int) -> bool:
        history = self.simulation.particle_history(particle_index)["history"]
        if history is None or len(history) == 0:
            return False
        return True

    def move_to_last(self) -> Self:
        assert not self.is_beginning
        target = self.target
        history = self.simulation.particle_history(target)["history"]
        history_chain = [history[i][1] for i in range(len(history))]
        current_in_chain = history_chain.index(self._current_point)
        self._current_point = history_chain[current_in_chain-1]
        return self

    def move_to_next(self) -> Self:
        assert not self.is_end
        target = self.target
        history = self.simulation.particle_history(target)["history"]
        history_chain = [history[i][1] for i in range(len(history))]
        current_in_chain = history_chain.index(self._current_point)
        self._current_point = history_chain[current_in_chain+1]
        return self
    
    def move_to_impactor(self) -> Self:
        assert self.impator_has_chain
        impactor = self.impactor
        history = self.simulation.particle_history(impactor)["history"]
        self._current_point = history[-1][1]
        return self

    def end_of_particle(self,i:int) -> Self:
        particle_history = self.simulation.particle_history(i)
        history = particle_history["history"]
        if history is None or len(history) == 0:
            terminal_type = particle_history["terminal_type"]
            terminal_event = particle_history["terminal_event"]
            if terminal_type in ["ejection","collision"]:
                raise ValueError(f"Particle {i} has no collision history as a target. And It finally ends by {terminal_type} with event index {terminal_event[1]}.")
            else:
                raise ValueError(f"Particle {i} has no collision history as a target. And It survived until the end of the simulation.")
        history_chain = [history[i][1] for i in range(len(history))]
        self._current_point = history_chain[-1]
        return self
    
    @statstic_time
    def melt_events(self,i:int) -> CollisionEvent:
        if self._collision_events[i] is None:
            collision = self.simulation.collisions[i]
            self._collision_events[i] = CollisionEvent(collision, **self._event_args)
        event = self._collision_events[i]
        assert event is not None
        return event
    
    @statstic_time
    def get_event_melt_fractions(self,i:int) -> float:
        if self._melt_fractions_event is None:
            #cache the melt fractions for all events in the disk.
            if os.path.exists(self._cache_melt_frac_file_path):
                self._melt_fractions_event = np.load(self._cache_melt_frac_file_path)
            else:
                len_collisions = len(self.simulation.collisions)
                self._melt_fractions_event = np.nan * np.ones(len_collisions)
        if not np.isnan(self._melt_fractions_event[i]):
            return float(self._melt_fractions_event[i])
        else:
            if i in self._unexpected_melt_event:
                return 0.0
            event = self.melt_events(i)
            try:
                melt_fraction = event.melt_fraction
            except:
                self._unexpected_melt_event.add(i)
                return 0.0
            self._melt_fractions_event[i] = melt_fraction
            self._new_melt_frac_since_last_cache += 1
            if self._new_melt_frac_since_last_cache >= 5 or np.all(~np.isnan(self._melt_fractions_event)):
                # update the cache file
                np.save(self._cache_melt_frac_file_path, self._melt_fractions_event)
                self._new_melt_frac_since_last_cache = 0
            return melt_fraction
    
    @lru_cache(maxsize=None)
    @statstic_time
    def get_particle_melt_fractions(self,i:int,dump_cache=True) -> float:
        particle_history = self.simulation.particle_history(i)["history"]
        initial_melt_frac = 0.0
        if len(particle_history) == 0:
            return initial_melt_frac
        else:
            #print(i,"history length:", len(particle_history))
            pass
        if i in self._melt_evolution_particles:
            #print(i, float(self._melt_evolution_particles[i][-1,0]))
            return float(self._melt_evolution_particles[i][-1,0])
        else:
            self._melt_evolution_particles[i] = np.zeros((len(particle_history)+1, 3))
            self._melt_evolution_particles[i][0,0] = initial_melt_frac
            self._melt_evolution_particles[i][0,1] = 0.0
            self._melt_evolution_particles[i][0,2] = 0.0
        for index_history, (event, collision_index) in  enumerate(particle_history):
            new_frac = self.get_event_melt_fractions(collision_index)
            bodyi = event["indexi"]
            bodyj = event["indexj"]
            massi:float = event["mi"]
            massj:float = event["mj"]
            time = event["time"]
            if bodyi == i:
                melti = initial_melt_frac
                meltj = self.get_particle_melt_fractions(bodyj, dump_cache=False)
                origin_mass = massi
            else:
                meltj = initial_melt_frac
                melti = self.get_particle_melt_fractions(bodyi, dump_cache=False)
                origin_mass = massj
            if index_history == 0:
                self._melt_evolution_particles[i][index_history,2] = origin_mass
            self._melt_evolution_particles[i][index_history+1,2] = massi + massj
            inherited_melt = (melti * massi + meltj * massj) / (massi + massj)
            print(i, new_frac, inherited_melt)
            initial_melt_frac = float(inherited_melt + new_frac)
            self._melt_evolution_particles[i][index_history+1,0] = initial_melt_frac
            self._melt_evolution_particles[i][index_history+1,1] = time
            self._collision_index2particle_evolution_index[collision_index] = (i, index_history+1)
            if dump_cache:
                with open(self._cache_evol_file_path, "wb") as f:
                    pickle.dump(self._melt_evolution_particles, f)
                self._evol_cache_updated = True
            else:
                self._evol_cache_updated = False
        return initial_melt_frac

    def get_particle_melt_evolution(self,i:int) -> np.ndarray:
        if i in self._melt_evolution_particles:
            return self._melt_evolution_particles[i]
        else:
            self.get_particle_melt_fractions(i)
            return self._melt_evolution_particles[i]
    # 在程序退出，或对象被GC时，保存缓存np.save(self._cache_melt_frac_file_path, self._melt_fractions_event)
    def __del__(self):
        if self._melt_fractions_event is not None and self._new_melt_frac_since_last_cache > 0:
            np.save(self._cache_melt_frac_file_path, self._melt_fractions_event)
        if hasattr(self, "_melt_evolution_particles") and not self._evol_cache_updated:
            with open(self._cache_evol_file_path, "wb") as f:
                pickle.dump(self._melt_evolution_particles, f)
    
class CollisionResult():
    def __init__(self, **kwargs) -> None:
        magma_init_keys = ['impact_angle','Mtotal','gamma','vel']
        self._magma_kwargs = {}
        for key in magma_init_keys:
            if key in kwargs:
                self._magma_kwargs[key] = kwargs[key]
        if 'Mtotal' in kwargs:
            self.Mtotal = kwargs['Mtotal']
        else:
            self.Mtotal = 1.0
    def _magma_init(self, **kwargs) -> Model:
        if 'impact_angle' not in kwargs:
            impact_angle = 0.0
        else:
            impact_angle = kwargs.pop('impact_angle')
        if 'Mtotal' not in kwargs:
            Mtotal = 1.0
        else:
            Mtotal = kwargs.pop('Mtotal')
        if 'gamma' not in kwargs:
            gamma = 0.5
        else:
            gamma = kwargs.pop('gamma')
        if 'vel' not in kwargs:
            vel = 2.0
        else:
            vel = kwargs.pop('vel')
        # round agnle to nearest choice: 0.0, 30.0, 45.0, 60.0, 90.0
        if impact_angle >= 60.0 and impact_angle < 75.0:
            impact_angle = 60
        elif impact_angle >= 75.0 and impact_angle <= 90.0:
            impact_angle = 90
        elif impact_angle >=0 and impact_angle < 15.0:
            impact_angle = 0
        elif impact_angle >=15.0 and impact_angle < 30:
            impact_angle = 30
        else:
            impact_angle = round(impact_angle / 15.0) * 15
        return Model(Mtotal=Mtotal, gamma=gamma, vel=vel, impact_angle=impact_angle, **kwargs)
    @cached_property
    def melt_model(self) -> Model:
        return self._magma_init(**self._magma_kwargs)
    @cached_property
    def av_du_gain(self) -> float:
        return float(np.mean(self.du_gain[self.melt_label==1]))
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
        return self.Mtotal * M_Mars
    @cached_property
    def peak_temperature(self) -> float:
        C_p = 1200
        T_0 = 1200
        return self.av_du_gain * 1e5 / C_p + T_0
    
def test_magma_model() -> None:
    """
    Parameter test for `CollisionResult`, and plot them.
    It will be a 3×2 subplot.
    a). Fix M_total to 1.0, gamma to 0.5, plot Melt Fraction and peak temperature vs impact angle, for different vel (= 0.5, 1.0, 2.0, 4.0).
    b). Fix M_total to 1.0, gamma to 0.1, plot Melt Fraction and peak temperature vs impact angle, for different vel (= 0.5, 1.0, 2.0, 4.0).
    c). The contour plot of Melt Fraction controlled by M_total (= 0.1 ~ 10, log scale) and gamma (= 0.1 ~ 0.5), fix the impact angle to 45, and vel to 1.0.
    d). The contour plot of peak temperature controlled by M_total and gamma, fix the impact angle to 45, and vel to 1.0. The mesh is the same as c).
    e). The contour plot of Melt Fraction controlled by M_total (= 0.1 ~ 10, log scale) and vrel (= 0.5 ~ 4.0), fix the impact angle to 45, and gamma to 0.1.
    f). The contour plot of peak temperature controlled by M_total and vrel, fix the impact angle to 45, and gamma to 0.5. The mesh is the same as e).
    """
    import matplotlib.pyplot as plt
    impact_angles = np.array([0.0, 30.0, 45.0, 60.0, 90.0])
    vels = [0.5, 1.0, 2.0, 4.0]
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    # a)
    # left y-axis is melt fraction, right y-axis is peak temperature
    ax_left = axs[0, 0]
    ax_right = ax_left.twinx()
    # left y-spines and ticks are red and solid, right y-spines and ticks are blue and dashed
    # use four red-like colors for 4 different vel to show the melt fraction
    left_colors = ["red", "orange", "brown", "magenta"]
    right_colors = ["blue", "cyan", "green", "purple"]
    for vindex, vel in enumerate(vels):
        melt_fractions = []
        peak_temperatures = []
        for angle in impact_angles:
            collision = CollisionResult(Mtotal=1.0, gamma=0.5, vel=vel, impact_angle=angle)
            melt_fractions.append(collision.melt_fraction)
            peak_temperatures.append(collision.peak_temperature)
        ax_left.plot(impact_angles, melt_fractions, color=left_colors[vindex], marker='o', label=f'Vel={vel}xVesc')
        ax_right.plot(impact_angles, peak_temperatures, color=right_colors[vindex], marker='x', linestyle='--', label=f'Vel={vel}xVesc')
    ax_right.set_yscale('log')
    ax_right.set_ylim(1000, 50000)
    ax_left.set_xlabel('Impact Angle (degree)')
    ax_left.set_ylabel('Melt Fraction', color='red')
    ax_right.set_ylabel('Peak Temperature (K)', color='blue')
    ax_left.set_title('Melt Fraction and Peak Temperature vs Impact Angle\n(Mtotal=1.0 Mars mass, gamma=0.5)')
    ax_left.legend(loc='upper left')
    ax_right.legend(loc='upper right')
    # set spines and ticks
    ax_left.spines['left'].set_color('red')
    ax_left.tick_params(axis='y', colors='red')
    ax_left.spines['right'].set_visible(False)
    ax_right.spines['right'].set_color('blue')
    ax_right.tick_params(axis='y', colors='blue')
    ax_right.spines['right'].set_linestyle('--')
    ax_right.spines['left'].set_visible(False)

    # b)
    ax_left = axs[0, 1]
    ax_right = ax_left.twinx()
    for vindex, vel in enumerate(vels):
        melt_fractions = []
        peak_temperatures = []
        for angle in impact_angles:
            collision = CollisionResult(Mtotal=1.0, gamma=0.1, vel=vel, impact_angle=angle)
            melt_fractions.append(collision.melt_fraction)
            peak_temperatures.append(collision.peak_temperature)
        ax_left.plot(impact_angles, melt_fractions, color=left_colors[vindex], marker='o', label=f'Vel={vel}xVesc')
        ax_right.plot(impact_angles, peak_temperatures, color=right_colors[vindex], marker='x', linestyle='--', label=f'Vel={vel}xVesc')
    ax_right.set_yscale('log')
    ax_right.set_ylim(1000, 50000)
    ax_left.set_xlabel('Impact Angle (degree)')
    ax_left.set_ylabel('Melt Fraction', color='red')
    ax_right.set_ylabel('Peak Temperature (K)', color='blue')
    ax_left.set_title('Melt Fraction and Peak Temperature vs Impact Angle\n(Mtotal=1.0 Mars mass, gamma=0.1)')
    ax_left.legend(loc='upper left')
    ax_right.legend(loc='upper right')
    # set spines and ticks
    ax_left.spines['left'].set_color('red')
    ax_left.tick_params(axis='y', colors='red')
    ax_left.spines['right'].set_visible(False)
    ax_right.spines['right'].set_color('blue')
    ax_right.tick_params(axis='y', colors='blue')
    ax_right.spines['right'].set_linestyle('--')
    ax_right.spines['left'].set_visible(False)

    # c) and d)
    Mtotals = np.logspace(-1, 1, 15)  # 0.1 to 10 Mars mass
    gammas = np.linspace(0.1, 0.5, 15)
    Melt_Fraction_mesh = np.zeros((len(Mtotals), len(gammas)))
    Peak_Temperature_mesh = np.zeros((len(Mtotals), len(gammas)))
    for i, Mtotal in enumerate(Mtotals):
        for j, gamma in enumerate(gammas):
            print(i,j)
            collision = CollisionResult(Mtotal=Mtotal, gamma=gamma, vel=1.0, impact_angle=45.0)
            Melt_Fraction_mesh[i, j] = collision.melt_fraction
            Peak_Temperature_mesh[i, j] = collision.peak_temperature
    ax = axs[1, 0]
    c1 = ax.contourf(gammas, Mtotals, Melt_Fraction_mesh, levels=10, cmap='viridis')
    fig.colorbar(c1, ax=ax, label='Melt Fraction')
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlabel('Gamma (Impactor Mass / Total Mass)')
    ax.set_ylabel('Total Mass (Mars mass)')
    ax.set_title('Melt Fraction Contour\n(Impact Angle=45°, Vel=1.0xVesc)')
    ax = axs[1, 1]
    # set max temperature to 10000 K for better colorbar. and set the lower bound of color bar to 1200 K
    Peak_Temperature_mesh = np.clip(Peak_Temperature_mesh, None, 10000)
    c2 = ax.contourf(gammas, Mtotals, Peak_Temperature_mesh, levels=10, cmap='coolwarm')
    fig.colorbar(c2, ax=ax, label='Peak Temperature (K)')
    c2.set_clim(1200, 10000)
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlabel('Gamma (Impactor Mass / Total Mass)')
    ax.set_ylabel('Total Mass (Mars mass)')
    ax.set_title('Peak Temperature Contour\n(Impact Angle=45°, Vel=1.0xVesc)')
    # e) and f)
    Mtotals = np.logspace(-1, 1, 15)  # 0.1 to 10 Mars mass
    vels = np.linspace(0.5, 4.0, 15)
    Melt_Fraction_mesh = np.zeros((len(Mtotals), len(vels)))
    Peak_Temperature_mesh = np.zeros((len(Mtotals), len(vels)))
    for i, Mtotal in enumerate(Mtotals):
        for j, vel in enumerate(vels):
            print(i,j)
            collision = CollisionResult(Mtotal=Mtotal, gamma=0.1, vel=vel, impact_angle=45.0)
            Melt_Fraction_mesh[i, j] = collision.melt_fraction
            Peak_Temperature_mesh[i, j] = collision.peak_temperature
    ax = axs[2, 0]
    c3 = ax.contourf(vels, Mtotals, Melt_Fraction_mesh, levels=10, cmap='viridis')
    fig.colorbar(c3, ax=ax, label='Melt Fraction')
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlabel('Velocity (x Vesc)')
    ax.set_ylabel('Total Mass (Mars mass)')
    ax.set_title('Melt Fraction Contour\n(Impact Angle=45°, Gamma=0.1)')
    ax = axs[2, 1]
    # set max temperature to 10000 K for better colorbar. 
    Peak_Temperature_mesh = np.clip(Peak_Temperature_mesh, None, 10000)
    c4 = ax.contourf(vels, Mtotals, Peak_Temperature_mesh, levels=10, cmap='coolwarm')
    c4.set_clim(1200, 10000)
    fig.colorbar(c4, ax=ax, label='Peak Temperature (K)')
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlabel('Velocity (x Vesc)')
    ax.set_ylabel('Total Mass (Mars mass)')
    ax.set_title('Peak Temperature Contour\n(Impact Angle=45°, Gamma=0.1)')

    # plot and save figure to 'collision_result_test.png'
    plt.tight_layout()
    plt.savefig('collision_result_test.pdf')


def main() -> None:
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

if __name__ == "__main__":
    main()
    # test_magma_model()