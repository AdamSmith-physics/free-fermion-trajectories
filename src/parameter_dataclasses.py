from dataclasses import dataclass

@dataclass
class SimulationParameters:
    steps: int
    Nx: int
    Ny: int
    p: float
    bonds: list
    site_in: int
    site_out: int
    drive_type: str = "current"  # "current", "dephasing"
    corner_dephasing: bool = False
    initial_state: str = "random"  # "checkerboard", "empty", "random", "custom"

    def to_dict(self):
        return {
            "steps": self.steps,
            "Nx": self.Nx,
            "Ny": self.Ny,
            "p": self.p,
            "bonds": self.bonds,
            "site_in": self.site_in,
            "site_out": self.site_out,
            "drive_type": self.drive_type,
            "corner_dephasing": self.corner_dephasing,
            "initial_state": self.initial_state
        }
    
    def from_dict(cls, data):
        return cls(
            steps=data["steps"],
            Nx=data["Nx"],
            Ny=data["Ny"],
            p=data["p"],
            bonds=data["bonds"],
            site_in=data["site_in"],
            site_out=data["site_out"],
            drive_type=data.get("drive_type", "current"),
            corner_dephasing=data.get("corner_dephasing", False),
            initial_state=data.get("initial_state", "random")
        )
