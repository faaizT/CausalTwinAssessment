from observational_model.Model import Model


class RealSimulator(Model):
    pass


def get_simulator(simulator_name):
    if simulator_name == "real":
        return RealSimulator()
    else:
        raise Exception("Simulator name incorrect. Must be one of [real]")
