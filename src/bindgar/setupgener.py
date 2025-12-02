import rebound # type: ignore
import numpy as np # type: ignore
#import pandas as pd
import os 
from typing import List, Tuple, Union, Literal
from ctypes import c_uint32
from .physics import DENSITY2K, M_EARTH, M_SUN
from .datahandle import SimulationOutputData
from .cli import register_command


# import dataclass
from dataclasses import dataclass

@dataclass
class ParticleGroupParams:
    N: int
    mass: float
    a_range: Tuple[float, float]
    e: float
    inc: float
    r: float
    e_type: Literal["uniform", "rayleigh", "fixed"] = "rayleigh"
    inc_type: Literal["uniform", "rayleigh", "fixed", "e_factor"] = "rayleigh"
    a_type: Literal["uniform", "powerlaw"] = "uniform"
    e2i_factor: float = 0.5
    a_factor: float = -1.5

def rand_powerlaw(min_v: float, max_v: float, factor: float, N: int) -> float:
    a = np.random.uniform(0, 1, N)
    if factor > 0:
        raise ValueError("Powerlaw factor must be negative.")
    if factor == -2.0:
        value = min_v * (max_v / min_v) ** a
    else:
        power = factor + 2.0
        value = ((max_v**power - min_v**power) * a + min_v**power) ** (1.0 / power)
    return value

def rand_uniform(minimum: float, maximum: float, N: int) -> float:
    return np.random.uniform(0, 1, N) * (maximum - minimum) + minimum

def rand_rayleigh(sigma: float, N: int) -> float:
    u = np.random.uniform(0, 1, N)
    return sigma * np.sqrt(-2.0 * np.log(1.0 - u))

def add_star(sim: rebound.Simulation, mass: float=1.0) -> None:
    hash_val = c_uint32(0)
    sim.add(m=mass, hash=hash_val)

def add_gas_giants(sim: rebound.Simulation, gas_giants_list: List[Tuple[float, float, float, float]]) -> None:
    exits_particles = len(sim.particles)
    #angles_e = np.random.uniform(0, 2*np.pi, len(gas_giants_list))
    #angles_inc = np.random.uniform(0, 2*np.pi, len(gas_giants_list))
    f = np.random.uniform(-np.pi, np.pi, len(gas_giants_list))
    for i, (mass, a, e, inc) in enumerate(gas_giants_list):
        sim.add(m=mass, a=a, e=e, inc=inc, f=f[i], r=DENSITY2K(1.33)*(mass)**(1/3), hash=c_uint32(exits_particles + i))

def add_particle_groups(sim: rebound.Simulation, embryo_params: ParticleGroupParams) -> None:
    exits_particles = len(sim.particles)
    N = embryo_params.N

    # set a
    if embryo_params.a_type == "uniform":
        a_values = rand_uniform(embryo_params.a_range[0], embryo_params.a_range[1], N)
    elif embryo_params.a_type == "powerlaw":
        a_values = rand_powerlaw(embryo_params.a_range[0], embryo_params.a_range[1], embryo_params.a_factor, N)
    else:
        raise ValueError(f"Unknown a_type: {embryo_params.a_type}")
    # set e
    if embryo_params.e_type == "uniform":
        e_values = rand_uniform(0.0, embryo_params.e, N)
    elif embryo_params.e_type == "rayleigh":
        e_values = rand_rayleigh(embryo_params.e, N)
    elif embryo_params.e_type == "fixed":
        e_values = np.full(N, embryo_params.e)
    else:
        raise ValueError(f"Unknown e_type: {embryo_params.e_type}")
    # set inc
    if embryo_params.inc_type == "uniform":
        inc_values = rand_uniform(0.0, embryo_params.inc, N)
    elif embryo_params.inc_type == "rayleigh":
        inc_values = rand_rayleigh(embryo_params.inc, N)
    elif embryo_params.inc_type == "e_factor":
        inc_values = e_values * embryo_params.e2i_factor
    elif embryo_params.inc_type == "fixed":
        inc_values = np.full(N, embryo_params.inc)
    else:
        raise ValueError(f"Unknown inc_type: {embryo_params.inc_type}")
    # set f, omega, Omega
    f_values = np.random.uniform(-np.pi, np.pi, N)
    omega_values = np.random.uniform(0.0, 2*np.pi, N)
    Omega_values = np.random.uniform(0.0, 2*np.pi, N)
    while sim.N < (exits_particles + embryo_params.N):
        index = sim.N - exits_particles
        sim.add(
            m=embryo_params.mass,
            a=a_values[index],
            e=e_values[index],
            inc=inc_values[index],
            f=f_values[index],
            omega=omega_values[index],
            Omega=Omega_values[index],
            r=embryo_params.r,
            hash=c_uint32(sim.N)
        )
def OutPutSim(sim: rebound.Simulation, output_file: str, format_str: str) -> None:
    outdata = SimulationOutputData(output_file, format_str, mode="w")
    with outdata:
        for i in range(1, sim.N):
            p = sim.particles[i]
            sun = sim.particles[0]
            data_dict = {
                "x": p.x - sun.x,
                "y": p.y - sun.y,
                "z": p.z - sun.z,
                "vx": p.vx - sun.vx,
                "vy": p.vy - sun.vy,
                "vz": p.vz - sun.vz,
                "m": p.m,
                "r": p.r,
            }
            # double check if format_str need more parameters
            attr_names = outdata.get_names()
            for name in attr_names:
                if name not in data_dict:
                    data_dict[name] = getattr(p, name)
            outdata.write_data(data_dict)

@register_command("setup-generate",help_msg="Generate initial conditions.")
def main():
    from .input import InputLoader
    DEFAULT_PARAMS = {
    "N_emb": {
        "default": 10,
        "help": "Number of embryos in the simulation",
        "type": int,
    },
    "N_pl": {
        "default": 100,
        "help": "Number of planetesimals in the simulation",
        "type": int,
    },
    "Mtot_disk": {
        "default": 10.0,
        "help": "Total mass of the disk in Earth mass",
        "type": float,
    },
    "Mtot_emb": {
        "default": None,
        "help": "Total mass of embryos in Earth mass (overrides m_emb if set)",
        "type": float,
    },
    "Mtot_pl": {
        "default": None,
        "help": "Total mass of planetesimals in Earth mass (overrides m_pl if set)",
        "type": float,
    },
    "m_emb": {
        "default": None,
        "help": "Mass of each embryo in solar mass",
        "type": float,
    },
    "m_pl": {
        "default": None,
        "help": "Mass of each planetesimal in solar mass",
        "type": float,
    },
    "r_emb": {
        "default": None,
        "help": "Radius of each embryo in AU",
        "type": float,
    },
    "r_pl": {
        "default": None,
        "help": "Radius of each planetesimal in AU",
        "type": float,
    },
    "rho": {
        "default": 4.0,
        "help": "Density of particles in g/cm^3",
        "type": float,
    },
    "emb_a_range": {
        "default": [0.8, 1.2],
        "help": "Semi-major axis range of embryos in AU",
        "type": List,
    },
    "pl_a_range": {
        "default": [0.9, 1.1],
        "help": "Semi-major axis range of planetesimals in AU",
        "type": List,
    },
    "emb_e_sigma": {
        "default": 0.01,
        "help": "Rayleigh sigma of embryos' eccentricity",
        "type": float,
    },
    "emb_inc_sigma": {
        "default": 0.005,
        "help": "Rayleigh sigma of embryos' inclination",
        "type": float,
    },
    "pl_e_sigma": {
        "default": 0.01,
        "help": "Rayleigh sigma of planetesimals' eccentricity",
        "type": float,
    },
    "pl_inc_sigma": {
        "default": 0.005,
        "help": "Rayleigh sigma of planetesimals' inclination",
        "type": float,
    },
    "gass_giants_list": {
        "default": [(9.5e-4, 5.2, 0.0489, 0.022)],
        "help": "List of gas giants to add (mass, a, e, inc)",
        "type": List,
    },
    "dat_filename": {
        "default": "initial_conditions.dat",
        "help": "Output filename for initial conditions",
        "type": str,
    },
    "format_str": {
        "default": "<< x:.9f y:.9f z:.9e m vx vy vz >>",
        "help": "Output format string",
        "type": str,
    },
    "random_seed": {
        "default": 42,
        "help": "Random seed for reproducibility",
        "type": int,
    },
    "a_type": {
        "default": "uniform",
        "help": "Type of semi-major axis distribution ('uniform' or 'powerlaw')",
        "type": str,
        "choices": ["uniform", "powerlaw"],
    },
    "a_powerlaw_factor": {
        "default": -1.5,
        "help": "Powerlaw factor for semi-major axis distribution (only if a_type is 'powerlaw')",
        "type": float,
    },
    "e_type": {
        "default": "rayleigh",
        "help": "Type of eccentricity distribution ('uniform', 'rayleigh', or 'fixed')",
        "type": str,
        "choices": ["uniform", "rayleigh", "fixed"],
    },
    "inc_type": {
        "default": "rayleigh",
        "help": "Type of inclination distribution ('uniform', 'rayleigh', 'fixed', or 'e_factor')",
        "type": str,
        "choices": ["uniform", "rayleigh", "fixed", "e_factor"],
    },
    "e2i_factor": {
        "default": 0.5,
        "help": "Factor to convert eccentricity to inclination if inc_type is 'e_factor'",
        "type": float,
    },
    }
    me = M_EARTH / M_SUN
    input_param = InputLoader(DEFAULT_PARAMS).load()
    N_emb = input_param["N_emb"]
    N_pl = input_param["N_pl"]
    Mtot_emb = input_param["Mtot_emb"] if input_param["Mtot_emb"] is not None else input_param["Mtot_disk"] / 2.0
    Mtot_pl = input_param["Mtot_pl"] if input_param["Mtot_pl"] is not None else input_param["Mtot_disk"] / 2.0
    m_emb = input_param["m_emb"] if input_param["m_emb"] is not None else Mtot_emb * me / float(N_emb)
    m_pl = input_param["m_pl"] if input_param["m_pl"] is not None else Mtot_pl * me / float(N_pl)
    r_emb = input_param["r_emb"]
    r_pl = input_param["r_pl"]
    if r_emb is None or r_pl is None:
        k = DENSITY2K(input_param["rho"])
        if r_emb is None:
            r_emb = k * (m_emb)**(1./3.)
        if r_pl is None:
            r_pl = k * (m_pl)**(1./3.)
    sim = rebound.Simulation()
    np.random.seed(input_param["random_seed"])
    add_star(sim)
    # Add embryos
    embryo_params = ParticleGroupParams(N=N_emb,
                                        mass=m_emb,
                                        a_range=tuple(input_param["emb_a_range"]),
                                        e=input_param["emb_e_sigma"],
                                        inc=input_param["emb_inc_sigma"],
                                        r=r_emb,
                                        e_type=input_param["e_type"],
                                        inc_type=input_param["inc_type"],
                                        a_type=input_param["a_type"],
                                        e2i_factor=input_param["e2i_factor"],
                                        a_factor=input_param["a_powerlaw_factor"])
    add_particle_groups(sim, embryo_params)
    # Add planetesimals
    planetesimal_params = ParticleGroupParams(N=N_pl,
                                              mass=m_pl,
                                              a_range=tuple(input_param["pl_a_range"]),
                                              e=input_param["pl_e_sigma"],
                                              inc=input_param["pl_inc_sigma"],
                                              r=r_pl,
                                              e_type=input_param["e_type"],
                                              inc_type=input_param["inc_type"],
                                              a_type=input_param["a_type"],
                                              e2i_factor=input_param["e2i_factor"],
                                              a_factor=input_param["a_powerlaw_factor"])
    add_particle_groups(sim, planetesimal_params)

    # add gas giants
    add_gas_giants(sim, gas_giants_list=input_param["gass_giants_list"])
    sim.move_to_com()
    # output to dat file
    output_file = input_param["dat_filename"]
    format_str = input_param["format_str"]
    OutPutSim(sim, output_file, format_str)
    print(f"Initial conditions written to {output_file}")

if __name__ == "__main__":
    main()
    
