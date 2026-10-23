"""Container for one event's distance-aligned qualifying laps."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Map renamed teams onto one lineage so team effects pool across seasons.
# Edit freely: this is a modelling decision, not a fact about the sport.
TEAM_ALIASES = {
    "Red Bull Racing": "Red Bull",
    "Alfa Romeo": "Sauber", "Audi": "Sauber", "Kick Sauber": "Sauber", "Alfa Romeo Racing": "Sauber",
    "AlphaTauri": "RB", "Racing Bulls": "RB", "Visa Cash App RB": "RB",
    "Haas F1 Team": "Haas", "Alpine F1 Team": "Alpine",
}


def canon_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


@dataclass
class EventData:
    year: int
    event: str
    distance: np.ndarray    # (g,) metres, uniform grid starting at 0
    drivers: np.ndarray     # (n,) three-letter codes
    teams: np.ndarray       # (n,)
    lap_times: np.ndarray   # (n,) seconds
    speed: np.ndarray       # (n, g) km/h on the distance grid
    time: np.ndarray        # (n, g) seconds elapsed since lap start at each grid point

    @property
    def key(self) -> str:
        return f"{self.year} {self.event}"

    def save(self, folder: str | Path) -> Path:
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        safe = self.key.replace(" ", "_").replace("/", "-")
        path = folder / f"{safe}.npz"
        np.savez_compressed(
            path, year=self.year, event=self.event, distance=self.distance,
            drivers=self.drivers.astype(str), teams=self.teams.astype(str),
            lap_times=self.lap_times, speed=self.speed, time=self.time)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EventData":
        z = np.load(path, allow_pickle=False)
        return cls(int(z["year"]), str(z["event"]), z["distance"], z["drivers"],
                   z["teams"], z["lap_times"], z["speed"], z["time"])


def load_all(folder: str | Path) -> list[EventData]:
    return [EventData.load(p) for p in sorted(Path(folder).glob("*.npz"))]
