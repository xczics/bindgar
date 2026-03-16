"""
In this module, we will handle the devoltilization during a single collision event.
"""

from typing import Dict, Optional,List, Tuple, Self, Set, Callable, Any, Literal
import numpy as np
from scipy.spatial import Delaunay
from ..physics import convert_factor_from_auptu_to_kmps, AU2M, M_EARTH, M_SUN, G, M_Mars, impact_angle_radians
import math
from functools import cached_property, lru_cache
from .nakajima.melt_model import Model
from .magma_cooling import MagmaOceanParameters,devoltilization
from ..output import SimulationOutput
from ..common import statstic_time, format_float
from datetime import datetime
import os
import pickle
import atexit

magma_produce_keys = ["entropy0","outputfigurename","use_tex","silent"]
cooling_keys = ["rho_B","rho_M","kapa","v" ,"k","C_p","alpha_V","g_s","r","L","M_mol"]
combined_keys = set(magma_produce_keys) | set(cooling_keys)

class CollisionEvent():
    _time: float # The time of the collision event.
    _target: int # The index of the target body.
    _impactor: int # The index of the impactor body.
    _have_time_target_impactor: bool = False # Whether the time, target, and impactor information have been set.
    total_mass_sun: float # The total mass of the colliding system in SUN mass.
    gamma: float # The mass ratio of the impactor to the total mass.
    rho_B: float # The bulk density of the target in kg/m3.
    impact_angle_radian: float # The impact angle in radian.
    vel_real: float # The real impact velocity in m/s.

    @property
    def time(self) -> float:
        """The time of the collision event."""
        assert self._have_time_target_impactor
        return self._time
    @property
    def target(self) -> int:
        """The index of the target body."""
        assert self._have_time_target_impactor
        return self._target
    @property
    def impactor(self) -> int:
        """The index of the impactor body."""
        assert self._have_time_target_impactor
        return self._impactor

    def __init__(self,collision: Dict, **kwargs) -> None:
        collision_handle_kwargs = {}
        if "rho_B" in kwargs:
            collision_handle_kwargs["rho_B"] = kwargs["rho_B"]
        if "skip_param_convert" in kwargs and kwargs["skip_param_convert"]:
            pass
        else:
            self._record2param(collision, **collision_handle_kwargs)
        self._magma_produce_kwargs = {key: kwargs[key] for key in magma_produce_keys if key in kwargs}
        self._cooling_kwargs = {key: kwargs[key] for key in cooling_keys if key in kwargs}

    @classmethod
    def from_parameter(cls,
                       M_total: float, # The total mass in Mars mass.
                       gamma: float, # The mass ratio of the impactor to the total mass.
                       impact_angle: float, # The impact angle in degree.
                       vel_escape_ratio: float, # The ratio of the real impact velocity to the escape velocity.
                       **kwargs
                       ) -> Self:
        """
        Create a CollisionEvent object from the given parameters.
        But the event from this method will not have the time, target, and impactor information.
        The rho_B can be set by kwargs, if not provided, it will be set to 3500 kg/m^3
        """
        other_kwargs = {key: kwargs[key] for key in combined_keys if key in kwargs}
        obj = cls(collision={}, skip_param_convert=True, **other_kwargs)
        if "rho_B" in other_kwargs:
            obj.rho_B = other_kwargs["rho_B"]
        else:
            obj.rho_B = 3500 # kg/m3, a typical value for rocky planets.
        obj.total_mass_sun = M_total * M_Mars / M_SUN
        obj.gamma = gamma
        obj.impact_angle_radian = impact_angle * np.pi / 180.0
        obj.vel_real = vel_escape_ratio * obj.escape_velocity
        return obj
    
    @classmethod
    def from_collision_data(cls, collision: Dict, **kwargs) -> Self:
        """
        Create a CollisionEvent object from the collision data.
        The collision data should contain the necessary information to calculate the parameters.
        The rho_B can be set by kwargs, if not provided, it will be calculated from the mass and radius of the target.
        """
        other_kwargs = {key: kwargs[key] for key in combined_keys if key in kwargs}
        obj = cls(collision=collision, **other_kwargs)
        return obj

    @statstic_time
    def _record2param(self,collision: Dict,
                             rho_B: Optional[float] = None) -> None:
        if collision["mi"] >= collision["mj"]:
            target_key, impactor_key = "i", "j"
        else:
            target_key, impactor_key = "j", "i"
        self._time = float(collision["time"])
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
        self._target = target_index
        self._impactor = impactor_index
        self._have_time_target_impactor = True
        self.total_mass_sun = (target_mass + impactor_mass) 
        self.gamma = impactor_mass / self.total_mass_sun
        delv =( impactor_v - target_v ) * convert_factor_from_auptu_to_kmps * 1e3  # in m/s
        delx = ( impactor_x - target_x ) * AU2M   # in m
        self.impact_angle_radian = impact_angle_radians(delx, delv)
        self.vel_real = float(np.linalg.norm(delv))
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
        if 0 <= self.impact_angle < 15.0:
            impact_angle = 0
        elif 15.0 <= self.impact_angle <= 30.0:
            impact_angle = 30
        elif 60.0 <= self.impact_angle < 75.0:
            impact_angle = 60
        elif 75.0 <= self.impact_angle <= 90.0:
            impact_angle = 90
        else:
            impact_angle = round(self.impact_angle / 15.0) * 15
        return Model(Mtotal=Mtotal, gamma=gamma, vel=vel, impact_angle=impact_angle, **kwargs)

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
        if self.melt_fraction > 1e-6:
            return float(np.mean(self.du_gain[self.melt_label==1]))
        else:
            return 0.0
    @cached_property
    def cooling_params(self) -> MagmaOceanParameters:
        if "rho_B" not in self._cooling_kwargs:
            self._cooling_kwargs["rho_B"] = self.rho_B
        if "r" not in self._cooling_kwargs:
            self._cooling_kwargs["r"] = self.radius
        return MagmaOceanParameters(**self._cooling_kwargs)
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
    def impact_angle(self) -> float:
        """The impact angle in degree."""
        return self.impact_angle_radian * 180.0 / np.pi  # in degree
    @cached_property
    def radius(self) -> float:
        """The radius of the merged body after collision in m, will calculate from the total mass and bulk density."""
        return ( self.total_mass / ( (4/3) * np.pi * self.rho_B ) )**(1/3)
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
        atexit.register(self._final_save)
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
        sim_finish_time = self.simulation.datetime_finished
        self._new_melt_frac_since_last_cache = 0
        if os.path.exists(self._cache_melt_frac_file_path):
            cache_melt_frac_time = datetime.fromtimestamp(os.path.getmtime(self._cache_melt_frac_file_path))
            if cache_melt_frac_time < sim_finish_time:
                print("The melt fraction cache file is outdated. It will be updated after new melt fractions are calculated.")
                os.remove(self._cache_melt_frac_file_path)
        if os.path.exists(self._cache_evol_file_path):
            cache_evol_time = datetime.fromtimestamp(os.path.getmtime(self._cache_evol_file_path))
            if cache_evol_time < sim_finish_time:
                print("The melt evolution cache file is outdated. It will be updated after new melt fractions are calculated.")
                os.remove(self._cache_evol_file_path)
                self._melt_evolution_particles = {}
            else:
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
                print("Something wrong when calculating the melt fraction for event index", i)
                print(f"M_total: {event.total_mass_in_earth:.3f} M_earth, gamma: {event.gamma:.3f}, vel_escape_ratio: {event.vel_escape_ratio:.3f}, impact_angle: {event.impact_angle:.1f} degree.")
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
            #print(i, new_frac, inherited_melt)
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
    # 但需要注意，程序退出时，numpy可能已经被卸载了。需要设置这个操作最先执行。
    def _final_save(self):
        if self._melt_fractions_event is not None and self._new_melt_frac_since_last_cache > 0:
            np.save(self._cache_melt_frac_file_path, self._melt_fractions_event)
            self._new_melt_frac_since_last_cache
        if hasattr(self, "_melt_evolution_particles") and not self._evol_cache_updated:
            with open(self._cache_evol_file_path, "wb") as f:
                pickle.dump(self._melt_evolution_particles, f)
            self._evol_cache_updated = True
    def __del__(self):
        if self._new_melt_frac_since_last_cache > 0 or not self._evol_cache_updated:
            self._final_save()

# Only to draw the test figures, do not use anymore.
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

@lru_cache(maxsize=2)
def boundary_magma_model(draw_directly=False) -> Any:
    """
    Parameter boundary test for `CollisionResult`.
    Draw a area where the nakajima's area they covered ("Interpolation area"),
      and the area their code can run ("Extropolation area"), 
      and other area should labeld as "Out of range".
    The x-axis is the total mass (from 0.03 - 40 M_mars, in log scale), 
      the y-axis is the velocity in unit of escape velocity (0.1 - 9, log scale),
      and the z-axis is the gamma parameter, from 0.0 to 0.5 (linear scale).
    The total plot should be a 3 column and 6 rows subplot.
      In each column, the first subplot is the 3D plot, the second to sixth subplot 
      are the 2D plot of x and y with z= 0.01, 0.03 ~ 0.05, 0.09 ~ 0.11, 0.2 ~ 0.301, and 0.5.
      In the first column, the impact angle is fixed to 0 degree, 
      in the second column the impact angle is fixed to 60 degree, 
      and in the third column the impact angle is fixed to 90 degree.
    The interpolation area should read from the csv file, the csv file located at "./nakajima/sph_input.txt", the root dir is the same as this file.
    Try to run their model, and get their melt fraction. If the model cannot run, label as "Out of range", other wise, label as "Extrapolation area".
      And also label the area where melt fraction lager than 0.1.
    The calculated and test result should be cached in the disk.
    """
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    MIN_M = 0.03
    MAX_M = 40
    MIN_V = 0.3
    MAX_V = 9
    M_array = np.logspace(np.log10(MIN_M), np.log10(MAX_M), 40)
    v_array = np.logspace(np.log10(MIN_V), np.log10(MAX_V), 30)
    gamma_array = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5])
    from ..datahandle import SimulationOutputData
    csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nakajima", "sph_input.txt")
    fmt = "<< Model:s,ID,MT,gamma,theta,v_imp/v_esc,v_esc(m/s) >>"
    data = SimulationOutputData(csv_file_path, format_spec=fmt, skip_header=True, split=",")
    sph_inputs = []
    with data:
        for row in data:
            sph_inputs.append([
                row["MT"],
                row["gamma"],
                row["theta"],
                row["v_imp/v_esc"],
            ])
    sph_inputs = np.array(sph_inputs)
    def test_it():
        test_results = []
        for theta in [0, 60, 90]:
            print(f"Testing impact angle {theta} degree...")
            melt_fraction_array = np.zeros((len(gamma_array), len(M_array), len(v_array)))
            intro_area_array = np.zeros((len(gamma_array), len(M_array), len(v_array)), dtype=bool)
            canrun_area_array = np.zeros((len(gamma_array), len(M_array), len(v_array)), dtype=bool)
            related_inputs = sph_inputs[sph_inputs[:,2] == theta]
            Delaunay_area = Delaunay(related_inputs[:,[0,1,3]])
            for i, gamma in enumerate(gamma_array):
                for j, M in enumerate(M_array):
                    for k, v in enumerate(v_array):
                        print(f"...Testing point M={M:.3f} M_mars, gamma={gamma:.3f}, v={v:.3f} v_esc.")
                        collision_result = CollisionResult(Mtotal=M, gamma=gamma, vel=v, impact_angle=theta)
                        try:
                            melt_fraction = collision_result.melt_fraction
                            melt_fraction_array[i,j,k] = melt_fraction
                            canrun_area_array[i,j,k] = True
                        except:
                            canrun_area_array[i,j,k] = False
                            pass
                        if canrun_area_array[i,j,k]:
                            # check if the point is in the D3_Hull area.
                            if Delaunay_area.find_simplex([M, gamma, v]) >= 0:
                                intro_area_array[i,j,k] = True
                            else:
                                intro_area_array[i,j,k] = False
            test_results.append({
            "melt_fraction_array": melt_fraction_array,
            "intro_area_array": intro_area_array,
            "canrun_area_array": canrun_area_array,
            })
        dir_this_file = os.path.dirname(os.path.abspath(__file__))
        cache_file_name = os.path.join(dir_this_file, "magma_model_boundary_test_cache.npy")
        np.save(cache_file_name, test_results, allow_pickle=True)
        return test_results
    dir_this_file = os.path.dirname(os.path.abspath(__file__))
    cache_file_name = os.path.join(dir_this_file, "magma_model_boundary_test_cache.npy")
    if os.path.exists(cache_file_name):
        test_results = np.load(cache_file_name, allow_pickle=True)
    else:
        test_results = test_it()
    intro_area_array = test_results[0]["intro_area_array"]
    canrun_area_array = test_results[0]["canrun_area_array"]  
    def back_ground_it(
            ax_2d: Axes,
            gamma_draw: float = 0.1,
            gamma_filter: Callable[[float], bool] = lambda gamma: 0.09 <= gamma <= 0.11,
            Mass_unit: str = "M_Mars",
            register_contour: bool = False,
    ):
        if Mass_unit not in ["M_Mars", "M_Earth"]:
            raise ValueError(f"Mass_unit should be either 'M_Mars' or 'M_Earth', but got {Mass_unit}.")
        Need_convert_to_earth = False
        if Mass_unit == "M_Earth":
            Need_convert_to_earth = True
        gamma_index = np.where(gamma_array == gamma_draw)[0][0]
        intro_area_2d = intro_area_array[gamma_index,:,:]
        canrun_area_2d = canrun_area_array[gamma_index,:,:]
        colors_2d = np.zeros(intro_area_2d.shape + (4,))
        colors_2d[~canrun_area_2d] = [0.5, 0.5, 0.5, 0.3]
        # 把canrun_area_2d中为false的地方打印到终端，看看他们的分布.
        if draw_directly:
            for i in range(canrun_area_2d.shape[0]):
                for j in range(canrun_area_2d.shape[1]):
                    if not canrun_area_2d[i,j]:
                        print(f"Gamma={gamma_draw:.3f}, M={M_array[i]:.3f}, v={v_array[j]:.3f} cannot run.")
        colors_2d[intro_area_2d] = [1.0, 0.0, 0.0, 0.3]
        colors_2d[canrun_area_2d & ~intro_area_2d] = [0.0, 0.0, 1.0, 0.3]
        M_array_plot = M_array
        if Need_convert_to_earth:
            M_array_plot = M_array.copy() / M_EARTH * M_Mars
        M_grid, v_grid = np.meshgrid(M_array_plot, v_array, indexing='ij')
        plot_colors = np.transpose(colors_2d, (1, 0, 2))
        ax_2d.pcolormesh(M_array_plot, v_array, plot_colors, shading='nearest')
        counter_line_color_map = ['black', 'purple', 'orange']
        thetas = [0, 60, 90]
        for theta_index in range(3):
            melt_fraction_2d = test_results[theta_index]["melt_fraction_array"][gamma_index,:,:]
            contour = ax_2d.contour(M_grid, v_grid, melt_fraction_2d, levels=[0.05], colors=counter_line_color_map[theta_index], linewidths=1)
                # register the contour lines to the legend
            if register_contour:
                ax_2d.plot([], [], color=counter_line_color_map[theta_index], label=f'5% melts for {thetas[theta_index]}°')
        ax_2d.set_xscale('log')
        ax_2d.set_yscale('log')
        inputs_with_same_theta = sph_inputs[sph_inputs[:,2] == 60.0]
        inputs_2d = []
        for input in inputs_with_same_theta:
            if gamma_filter(input[1]):
                inputs_2d.append([input[0], input[3]])
        inputs_2d = np.array(inputs_2d)
        if Need_convert_to_earth:
            inputs_2d[:,0] = inputs_2d[:,0] / M_EARTH * M_Mars
        if inputs_2d.shape[0] > 0:
            ax_2d.scatter(inputs_2d[:,0], inputs_2d[:,1], color='red', 
                          marker='o', label='Nakajima SPH inputs', s=50,
                          zorder=10) # 设置总在最顶层
        if not Need_convert_to_earth:
            ax_2d.set_xlabel(r'Total Mass ($M_{Mars}$)')
        else:
            ax_2d.set_xlabel(r'Total Mass ($M_{Earth}$)')
        ax_2d.set_ylabel(r'Velocity (v/$v_{esc}$)')
        
    def draw_it():
        # init page and subplots
        SUB_PLOT_ROWS = 3
        SUB_PLOT_COLS = 2
        WIDTH_SUBPLOT = 4
        HEIGHT_SUBPLOT = 4
        fig = plt.figure(figsize=(WIDTH_SUBPLOT*SUB_PLOT_COLS, HEIGHT_SUBPLOT*SUB_PLOT_ROWS))
        fig_grid = fig.add_gridspec(SUB_PLOT_ROWS, SUB_PLOT_COLS, wspace=0.3, hspace=0.3)
        SUB_PLOT_GAMMAS = [0.01, 0.03, 0.1, 0.2, 0.5]
        gamma_filters = [
            lambda _: False,
            lambda gamma: 0.02 <= gamma <= 0.05,
            lambda gamma: 0.09 <= gamma <= 0.11,
            lambda gamma: 0.2 <= gamma <= 0.302,
            lambda gamma: gamma >= 0.45,
        ]
        # 使用 add_axes 手动创建3D轴, 第一列，第一行
        left = 0.05
        bottom = 0.7
        width = 0.4
        height = 0.25
        ax_3d = fig.add_axes([left, bottom, width, height], projection='3d') #type: ignore
        # 填充内插区域维红色，外插但可运行区域为蓝色，其他区域为灰色。高亮 melt fraction 大于 0.1 的区域。并投上输入散点。
        colors = np.zeros(intro_area_array.shape + (4,))
        colors[~canrun_area_array] = [0.5, 0.5, 0.5, 0.0]
        colors[intro_area_array] = [0.8, 0.0, 0.0, 0.8]
        colors[canrun_area_array & ~intro_area_array] = [0.0, 0.0, 1.0, 0.0]
        gamma_grid, M_grid, v_grid = np.meshgrid(gamma_array, M_array, v_array)
        log_M_grid = np.log10(M_grid)
        log_v_grid = np.log10(v_grid)
        ax_3d.scatter(log_M_grid.flatten(), log_v_grid.flatten(), gamma_grid.flatten(), 
                  color=colors.reshape(-1,4), marker='x', s=1) # type: ignore
        log_sph_inputs_x = np.log10(sph_inputs[:,0])
        log_sph_inputs_y = np.log10(sph_inputs[:,3])
        ax_3d.scatter(log_sph_inputs_x, log_sph_inputs_y, sph_inputs[:,1], color='red', marker='o', label='Nakajima SPH inputs', s=10) # type: ignore
        # 添加y=0 平面 和z=0.1 平面
        # y=0 平面
        ax_3d.plot_surface(np.log10(np.array([[MIN_M,MAX_M], [MIN_M, MAX_M]])),#type: ignore
                               np.log10(np.array([[1.0, 1.0], [1.0, 1.0]])),
                               np.array([[0.0,0.0], [0.5, 0.5]]), 
                                color='grey', alpha=0.25)
        # z=0.1 平面
        ax_3d.plot_surface(np.log10(np.array([[MIN_M, MAX_M], [MIN_M, MAX_M]])),#type: ignore
                               np.log10(np.array([[MIN_V, MIN_V], [MAX_V, MAX_V]])),
                                 np.array([[0.1,0.1], [0.1, 0.1]]), 
                                  color='grey', alpha=0.25)
            
        ax_3d.set_xlim(np.log10(MIN_M), np.log10(MAX_M))
        ax_3d.set_ylim(np.log10(MIN_V), np.log10(MAX_V))
        ax_3d.set_zlim(0.0, 0.5) #type: ignore
        ax_3d.set_xlabel(r'Total Mass (log10($M_{Mars}$))')
        ax_3d.set_ylabel(r'Velocity (log10(v/$v_{esc}$))')
        ax_3d.set_zlabel('Gamma') #type: ignore
        ax_3d.legend()
        # 画二维图，x轴是总质量，y轴是速度，颜色是 melt fraction。每个子图对应一个 gamma 值。从上到下，从左到右
        for sub_index in range(5):
            row = (sub_index + 1) % SUB_PLOT_ROWS
            col = (sub_index + 1) // SUB_PLOT_ROWS
            ax_2d = fig.add_subplot(fig_grid[row, col])
            ax_2d.set_xlim(MIN_M, MAX_M)
            ax_2d.set_ylim(0.7, MAX_V)
            gamma = SUB_PLOT_GAMMAS[sub_index]
            ax_2d.set_title(f'Gamma = {gamma:.2f}')
            back_ground_it(ax_2d, gamma_draw=gamma, gamma_filter=gamma_filters[sub_index], register_contour=sub_index==0)
            if sub_index == 0:
                ax_2d.legend()
        # 调整整体标题和布局
        # save the figure
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "magma_model_boundary_test.pdf"))

    if draw_directly:
        draw_it()
    else:
        return back_ground_it

def f_T_C_m_contour(draw_directly: bool=False,
                            gamma: float = 0.1,
                            impact_angle: int= 60,
                            gamma_angle_list = None,
                            ) -> Any:
    """
    A parameter test for `CollisionResult`. It will generate the data
    in an parameter space of x=M_total=0.03 ~ 40 M_mars (log scale), y=v_rel=0.3 ~ 9 v_esc (log scale) mesh.
    The gamma can be choice of [0.01, 0.03, 0.1, 0.2, 0.5], and the impact angle can be choice of [0, 30, 45,60,90] degree.
    If draw_directly is True, it will directly draw the contour plot of melt fraction and peak temperature in the parameter space, and save the figure.
    If draw_directly is False, it will return a function that can be used to draw the contour plot of melt fraction and peak temperature in the parameter space.
    If gamma_angle_list is None, it will use the gamma and impact_angle in the arguments.
    If gamma_angle_list is not None, the gamma and impact_angle will be ignored, but 
        use the (gamma, impact_angle) pairs in the gamma_angle_list to draw the contour plot.
        In the meanwhile, if draw_directly is false, the returned function should also accept an index parameter to specify which (gamma, impact_angle) pair in the gamma_angle_list to draw.
    The gamma_angle_list should be a list of tuples, each tuple is (gamma, impact_angle).
    The function is expensive, so when you want to call with draw_directly=False and different gamma, impact_angle value, 
        use gamma_angle_list to specify all the (gamma, impact_angle) pairs ahead, and call the return function with different index.
    """
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.contour import QuadContourSet
    MIN_M = 0.03
    MAX_M = 40
    MIN_V = 0.3
    MAX_V = 9
    M_array = np.logspace(np.log10(MIN_M), np.log10(MAX_M), 80)
    v_array = np.logspace(np.log10(MIN_V), np.log10(MAX_V), 80)
    M_grid, v_grid = np.meshgrid(M_array, v_array, indexing='ij')
    contor_titles = {
        "melt_fraction": r"$f_{melt}$",
        "peak_temperature": r"$T_{peak}$ (K)",
        "C": r"$C/C_{0}$",
        "M_loss": r"$M_{loss}/M_{melt}$",
    }
    def calculate_contour(gamma_value: float, impact_angle_value: int) -> Dict[str, np.ndarray]:
        melt_fraction_array = np.zeros(M_grid.shape)
        peak_temperature_array = np.zeros(M_grid.shape)
        C_array = np.zeros(M_grid.shape)
        M_loss_array = np.zeros(M_grid.shape)
        for i in range(M_grid.shape[0]):
            for j in range(M_grid.shape[1]):
                #print(i,j)
                M = M_grid[i,j]
                v = v_grid[i,j]
                collision_result = CollisionEvent.from_parameter(M_total=M, gamma=gamma_value, vel_escape_ratio=v, impact_angle=impact_angle_value)
                try:
                    melt_fraction_array[i,j] = collision_result.melt_fraction
                    peak_temperature_array[i,j] = collision_result.melt_T_increase() + 1200
                    devol_result = collision_result.devoltilize(final_only=True)
                    C_array[i,j] = devol_result[3]
                    M_loss_array[i,j] = devol_result[2] / collision_result.melt_mass
                except Exception as e:
                    melt_fraction_array[i,j] = np.nan
                    peak_temperature_array[i,j] = np.nan
                    C_array[i,j] = np.nan
                    M_loss_array[i,j] = np.nan
                    # in test mode, raise the error to see where the problem is.
                    raise e
        return {
            "melt_fraction": melt_fraction_array,
            "peak_temperature": peak_temperature_array,
            "C": C_array,
            "M_loss": M_loss_array,
        }
    if gamma_angle_list is None:
        gamma_angle_list = [(gamma, impact_angle)]
        is_one_pair = True
    else:
        is_one_pair = False
    MT_Datas = []
    dir_this_file = os.path.dirname(os.path.abspath(__file__))
    for index, (gamma_value, impact_angle_value) in enumerate(gamma_angle_list):
        if not any(np.isclose(gamma_value, valid_gamma, atol=1e-6) for valid_gamma in [0.01, 0.03, 0.1, 0.2, 0.5]):
            raise ValueError(f"Gamma should be one of [0.01, 0.03, 0.1, 0.2, 0.5], but got {gamma}.")
        if impact_angle_value not in [0, 30, 45, 60, 90]:
            raise ValueError(f"Impact angle should be one of [0, 30, 45, 60, 90] degree, but got {impact_angle}.")
        cache_file_name = os.path.join(dir_this_file, f"./.cache/magma_model_MT_contour_cache_gamma{gamma_value:.2f}_angle{impact_angle_value:02d}.npy")
        if os.path.exists(cache_file_name):
            MT_Data = np.load(cache_file_name, allow_pickle=True).item()
        else:
            MT_Data = calculate_contour(gamma_value, impact_angle_value)
            np.save(cache_file_name, MT_Data, allow_pickle=True) # type: ignore
        MT_Datas.append(MT_Data)
    def draw_it(ax: Axes,
                index: int = 0,
                contour: str = "melt_fraction",
                Mass_unit: str = "M_Mars",
                **contor_kwargs
                ) -> QuadContourSet:
        assert contour in ["melt_fraction", "peak_temperature", "C", "M_loss"], f"Contor should be one of ['melt_fraction', 'peak_temperature', 'C', 'M_loss'], but got {contour}."
        assert Mass_unit in ["M_Mars", "M_Earth"], f"Mass_unit should be either 'M_Mars' or 'M_Earth', but got {Mass_unit}."
        related_Data = MT_Datas[index]
        contor_data = related_Data[contour]
        if "levels" in contor_kwargs:
            levels = contor_kwargs.pop("levels")
        else:
            levels = 10
        if "cmap" in contor_kwargs:
            cmap = contor_kwargs.pop("cmap")
        else:
            cmap = "viridis"
        if Mass_unit == "M_Earth":
            need_convert_to_earth = True
            M_grid_plot = M_grid.copy() / M_EARTH * M_Mars
        else:
            need_convert_to_earth = False
            M_grid_plot = M_grid
        # 如果是peak_temperature, 
        if contour == "peak_temperature":
            # 将np.nan的地方设置为1200K, 也就是没有升温的地方。
            contor_data = np.where(np.isnan(contor_data), 1200, contor_data)
            # 将超过10 000K 的地方设置为 10 000K, 也就是模型的最高温度。
            contor_data = np.where(contor_data > 10000, 10000, contor_data)
        contor_plot = ax.contourf(M_grid_plot, v_grid, contor_data, levels=levels, cmap=cmap, **contor_kwargs)
        if contour != "peak_temperature":
            #添加colorbar
            plt.colorbar(contor_plot, ax=ax)
        else:
            # 对于peak_temperature, 如果最高温度超过10000K, 就在colorbar上标注一个特殊的标签，表示超过模型最高温度。
            cbar = plt.colorbar(contor_plot, ax=ax)
            ticks = cbar.get_ticks()
            # 去除ticks的最高点，如果它超过了10000
            ticks = ticks[ticks <= 10000]
            # 去除ticks中小于1200的点，因为1200K是没有升温的地方，不需要标注。
            ticks = ticks[ticks >= 1200]
            cbar.set_ticks(list(ticks) + [10000])
            cbar.set_ticklabels([*map(str, ticks), '>10000K'])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(0.9, MAX_V)
        ax.set_yticks([1, 3, 9])
        ax.set_yticks([2, 4, 5, 6, 7, 8], minor=True) 
        # 设置yticks的label为整数，而非科学计数法
        ax.set_yticklabels([f'{int(tick)}' for tick in ax.get_yticks()])
        # 忽略minor ticks的标签
        ax.tick_params(which='minor', labelbottom=False, labelleft=False)
        if not need_convert_to_earth:
            ax.set_xlabel(r'Total Mass ($M_{Mars}$)')
        else:
            ax.set_xlabel(r'Total Mass ($M_{Earth}$)')
        ax.set_ylabel(r'Velocity (v/$v_{esc}$)')
        ax.set_title(contor_titles[contour] +
                     " (Gamma = " + format_float(gamma_angle_list[index][0],1,2) +
                     f", Impact Angle = {gamma_angle_list[index][1]}°)")
        return contor_plot
    def draw_first(
            ax: Axes,
            contor: str = "melt_fraction",
            Mass_unit: str = "M_Mars",
            **contor_kwargs
            ) -> QuadContourSet:
        return draw_it(ax, index=0, contor=contor, Mass_unit=Mass_unit, **contor_kwargs)
    if not draw_directly:
        if is_one_pair:
            return draw_first
        else:
            return draw_it
    H_SUBPLOT = 5
    W_SUBPLOT = 6
    if len(gamma_angle_list) <=8:
        SUBPLOT_COLS = 4
        SUBPLOT_ROWS = len(gamma_angle_list)
    else:
        # 如果有超过8对参数组合，行数就设置为sqrt(gamma_angle_list*4), 向上取整
        SUBPLOT_ROWS = math.ceil(math.sqrt(len(gamma_angle_list)*4))
        # 列数设置为 4*SUBPLOT_ROWS/len(gamma_angle_list)，向上取整
        SUBPLOT_COLS = math.ceil(4*SUBPLOT_ROWS/len(gamma_angle_list))
    fig = plt.figure(figsize=(W_SUBPLOT*SUBPLOT_COLS, H_SUBPLOT*SUBPLOT_ROWS))
    # for each (gamma, impact_angle) pair, draw the contour of melt fraction, peak temperature, C, and M_loss in a row.
    for index in range(len(gamma_angle_list)):
        logical_col_index = index % SUBPLOT_ROWS
        logical_row_index = index // SUBPLOT_ROWS
        ax_melt_fraction = fig.add_subplot(SUBPLOT_ROWS, SUBPLOT_COLS, logical_row_index*SUBPLOT_COLS+logical_col_index*4+1)
        draw_it(ax_melt_fraction, index=index, contour="melt_fraction", levels=np.linspace(0, 1, 11))
        ax_peak_temperature = fig.add_subplot(SUBPLOT_ROWS, SUBPLOT_COLS, logical_row_index*SUBPLOT_COLS+logical_col_index*4+2)
        draw_it(ax_peak_temperature, index=index, contour="peak_temperature", levels=10)
        ax_C = fig.add_subplot(SUBPLOT_ROWS, SUBPLOT_COLS, logical_row_index*SUBPLOT_COLS+logical_col_index*4+3)
        draw_it(ax_C, index=index, contour="C", levels=10)
        ax_M_loss = fig.add_subplot(SUBPLOT_ROWS, SUBPLOT_COLS, logical_row_index*SUBPLOT_COLS+logical_col_index*4+4)
        draw_it(ax_M_loss, index=index, contour="M_loss", levels=10)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "magma_model_MT_contour_test.pdf"))


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
    # main()
    # test_magma_model()
    # boundary_magma_model(draw_directly=True)
    f_T_C_m_contour(draw_directly=True,
                    gamma_angle_list=[
                                      (0.01, 30), 
                                      (0.01, 90),
                                      (0.01, 45),
                                      (0.03, 30), 
                                      (0.03, 45),
                                      (0.03, 90), 
                                      (0.2, 30),
                                      (0.2, 90),
                                      (0.5, 30),
                                      (0.5, 90),
                    ])