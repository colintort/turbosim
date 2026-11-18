from thermo_engine import State
from CoolProp.CoolProp import PropsSI


class Turbine:
    def __init__(self, inlet_state: State, outlet_pressure, efficiency=1.0):
        self.inlet = inlet_state
        self.P_out = outlet_pressure
        self.eta = efficiency
        self.solve()

    def solve(self):
        h_in = self.inlet.h
        s_in = self.inlet.s
        fluid = self.inlet.fluid

        # Isentropic outlet enthalpy at the outlet pressure. (P, S) is valid in
        # every phase region, so there is no fallback and no clamping: the old
        # clamp to the saturation range forced superheated or subcooled outlets
        # onto the saturation line.
        h_out_s = PropsSI("H", "P", self.P_out, "S", s_in, fluid)
        h_out = h_in - self.eta * (h_in - h_out_s)

        self.outlet = State(fluid, P=self.P_out, H=h_out)
        self.w = h_in - self.outlet.h