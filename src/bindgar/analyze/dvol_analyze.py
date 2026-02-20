from ..input import InputLoader,InputAcceptable
from ..cli import register_command
from ..output import SimulationOutput
from ..devol.collision_event import CollisionEvent


@register_command("devol-calc")
def main():
    DEFAULT_PARAMS: InputAcceptable = {
        "simulations_lists" : {
            "default": None,
            "help": "list of simulation paths",
            "short": "l",
            "type": list,
        },
    }
    input_params = InputLoader(DEFAULT_PARAMS).load()
    simulations_lists = input_params["simulations_lists"]
    for simulation in simulations_lists:
        print(f"Processing simulation: {simulation}")
        simobj = SimulationOutput(simulation)
        devol_out = simobj.open_write_pip("devol.out",fmt="<< m_melt:.6e c:9.6f t:.2f m_loss:.6e T_increase:9.2f >>")
        collisions = simobj.collisions
        with collisions:
            for index, collision in enumerate(collisions):
                print(f"Processing collision {index}")
                col_event = CollisionEvent(collision)
                m_melt = col_event.melt_mass
                _, t, m_loss, C = col_event.devoltilize()
                devol_out.write_data({
                    "m_melt": m_melt,
                    "c": C,
                    "t": t/(24*3600*365.25), 
                    "m_loss": m_loss,
                    "T_increase": col_event.melt_T_increase(),
                })
                print(f"  Melted mass: {m_melt:.6e}, Peak Temperature: {(col_event.melt_T_increase()+1200.0):9.2f} Devolatilization loss: {m_loss:.6e}, C: {C:.6f}, Time: {t/(24*3600*365.25):.2f} years")
        devol_out.close()
                 
if __name__ == "__main__":
    main()