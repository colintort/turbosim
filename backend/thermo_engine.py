from CoolProp.CoolProp import PropsSI

# Accept lowercase / friendly names too, so State(fluid, P=..., h=...) works
# the same as State(fluid, P=..., H=...). CoolProp itself wants P, T, H, S, U, D, Q.
_ALIASES = {"p": "P", "t": "T", "h": "H", "s": "S", "u": "U", "d": "D", "q": "Q", "x": "Q"}


class State:
    def __init__(self, fluid, **kwargs):
        self.fluid = fluid
        self.inputs = {_ALIASES.get(k, k): v for k, v in kwargs.items()}
        self.solve_state()

    def solve_state(self):
        keys = list(self.inputs.keys())
        if len(keys) != 2:
            raise ValueError("State must be defined by exactly two properties")

        k1, k2 = keys
        v1, v2 = self.inputs[k1], self.inputs[k2]

        def prop(name):
            if name in self.inputs:
                return self.inputs[name]
            return PropsSI(name, k1, v1, k2, v2, self.fluid)

        self.P = prop("P")
        self.T = prop("T")
        self.h = prop("H")
        self.s = prop("S")
        self.u = prop("U")
        self.v = 1 / prop("D")

    def to_dict(self):
        return {
            "P": self.P,
            "T": self.T,
            "h": self.h,
            "s": self.s,
            "u": self.u,
            "v": self.v
        }