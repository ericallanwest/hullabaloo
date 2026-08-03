"""Project-wide configuration: paths, CRS, race parameters, model defaults.

Every tunable in the pipeline is declared here so the marimo notebooks can import a
single object and override individual fields without reaching into module internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"

GPX_DIR = RAW / "gpx"
DEM_PATH = RAW / "dem_1m.tif"
DEM_TILE_PATH = RAW / "dem_tile_1m.tif"
DEM_FALLBACK_PATH = RAW / "dem_10m.tif"

#: USGS 3DEP 1 m lidar tile covering the entire Pandapas Pond study area.
#: Tile x54y413 spans 540000-550000 E, 4120000-4130000 N in UTM 17N, which contains the
#: whole trail network. Downloading the source tile is markedly more reliable than the
#: dynamic elevation service, whose async client fails intermittently on Windows.
DEM_TILE_URL = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/"
    "VA_FEMA-NRCS_SouthCentral_2017_D17/TIFF/"
    "USGS_1M_17_x54y413_VA_FEMA-NRCS_SouthCentral_2017_D17.tif"
)

TRAILS_RAW = INTERIM / "trails_raw.parquet"
TRAIL_TABLE = INTERIM / "trail_table.csv"
EDGES = PROCESSED / "network_edges.parquet"
NODES = PROCESSED / "network_nodes.parquet"
EDGES_TIMED = PROCESSED / "network_edges_timed.parquet"
CONNECTORS = PROCESSED / "connectors.parquet"
GRAPH_EDGES = PROCESSED / "graph_edges.parquet"
GRAPH_NODES = PROCESSED / "graph_nodes.parquet"

for _d in (RAW, INTERIM, PROCESSED, OUTPUTS, GPX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Site
# --------------------------------------------------------------------------------------

BASE_URL = "https://thetrailnottaken.com"
LOGIN_URL = f"{BASE_URL}/login"
TRAIL_LIST_URL = f"{BASE_URL}/displayTrailList"
GPX_URL = f"{BASE_URL}/downloadTrailGPX?trailid={{trail_id}}"

# --------------------------------------------------------------------------------------
# Spatial reference
# --------------------------------------------------------------------------------------

CRS_GEOGRAPHIC = "EPSG:4326"
#: NAD83(2011) / UTM zone 17N, metres. Matches the 3DEP 1 m lidar tiling for this area
#: (tile x54y413 => 540000 E, 4130000 N in UTM 17N).
CRS_PROJECTED = "EPSG:6346"

#: Race start/finish, as given by the organizer (WGS84 lat/lon).
START_LAT = 37.24503461196985
START_LON = -80.45980444290193

M_PER_MILE = 1609.344
FT_PER_M = 3.280839895


# --------------------------------------------------------------------------------------
# Model parameters
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ToblerParams:
    """Parameterized Tobler hiking function.

        W(S) = base * exp(-k * |S + s0|)     [km/h],  S = dz/dx  (rise over run)

    ``s0`` shifts the peak to a slight downhill, which is what makes the function
    asymmetric and therefore makes direction of travel matter.
    """

    base_kmh: float = 6.0
    k: float = 3.5
    s0: float = 0.05
    #: Multiplier applied to on-trail speed when travelling off-trail (bushwhacking).
    off_trail_factor: float = 0.60
    #: Global pace multiplier — a place to encode fatigue, pack weight, or personal
    #: fitness once calibrated. 1.0 == textbook Tobler.
    #:
    #: Set to 1.35 for this racer: Tobler's constants describe unhurried walking, and a
    #: fit competitor moving with purpose over seven hours is meaningfully quicker. This
    #: puts peak speed at 8.1 km/h (5.03 mph) on a gentle downhill and 6.80 km/h
    #: (4.23 mph, ~14.2 min/mile) on the flat. It scales every speed linearly, so the
    #: shape of the function — and which direction round a loop is faster — is unchanged.
    pace_factor: float = 1.35
    #: Speed floor so that pathological slopes cannot produce ~infinite traversal times.
    min_speed_kmh: float = 0.15


@dataclass(frozen=True)
class TopologyParams:
    #: Vertex-cluster / snap tolerance in metres. Drives junction detection.
    #:
    #: Chosen from the sweep in ``topology.tolerance_sweep``: component count falls
    #: 14 -> 6 -> 4 -> 3 as tolerance rises and then plateaus at 3 from 18 m all the way
    #: out to 50 m. That plateau is the structural floor — the three remaining components
    #: are separated by genuine 285-700 m gaps that no snapping will ever close, which is
    #: exactly why Phase 4 exists. Every junction created in the 10-18 m band was reviewed
    #: individually (see ``topology.junctions_in_band``) and each is a real trail
    #: intersection whose endpoints simply disagree by plausible under-canopy GPS error.
    snap_tol_m: float = 18.0
    #: Douglas-Peucker tolerance applied to shed GPS jitter, metres.
    simplify_tol_m: float = 0.5
    #: Edges shorter than this after splitting are collapsed into their neighbour.
    min_edge_len_m: float = 1.0


@dataclass(frozen=True)
class ElevationParams:
    #: Gaussian smoothing radius applied before slope computation, in **metres**.
    #:
    #: Specified in ground units rather than pixels on purpose: the same value then means
    #: the same physical smoothing whether we are on the 1 m lidar or the 10 m fallback.
    #: Expressing it in pixels silently applied 30 m of smoothing to the 10 m DEM and
    #: flattened genuinely steep trails.
    #:
    #: 5 m chosen from ``elevation.smoothing_sensitivity``: it drives the bias against the
    #: 80 published gain values to ~0 ft. Reassuringly the choice is not a big lever —
    #: total network traversal time moves only 3.4% across sigma 0-12 m — so the model is
    #: not hostage to it.
    smooth_sigma_m: float = 5.0
    #: Along-line resampling step for elevation profiles, metres.
    sample_step_m: float = 10.0
    #: Buffer around the network bbox when fetching the DEM, metres.
    dem_buffer_m: float = 600.0
    dem_resolution_m: int = 1


@dataclass(frozen=True)
class BushwhackParams:
    #: Maximum straight-line distance between two nodes to be considered as a candidate
    #: off-trail connector, metres.
    max_connector_dist_m: float = 900.0
    #: Within a component, only keep a connector if the on-network route is at least this
    #: many times slower than the bushwhack.
    min_detour_ratio: float = 2.5
    #: Hard cap on the number of connectors admitted to the graph.
    max_connectors: int = 80
    #: Slopes steeper than this (rise/run) are treated as impassable off-trail.
    max_offtrail_slope: float = 1.0
    #: Treat built-up NLCD classes as impassable off-trail.
    #:
    #: Not optional in practice. Without it the optimizer routed its longest bushwhack
    #: 848 m straight through a residential neighbourhood, because a bare-earth DEM sees
    #: only gentle slope where the houses are.
    avoid_developed: bool = True
    #: Which NLCD classes count as built-up.
    #:
    #: 22/23/24 are low/medium/high-intensity development — houses, driveways, parking,
    #: commercial. Those are genuinely off-limits.
    #:
    #: 21 ("Developed, Open Space", <20% impervious) is deliberately **excluded** by
    #: default. In a forested study area it is dominated by road right-of-ways and their
    #: verges, and blocking it forbids ever crossing a road — which cost 18 of the 44
    #: cross-component connectors and ~3 points, for no legitimate reason. Add 21 here if
    #: you want to keep clear of large-lot residential lawns as well.
    developed_classes: tuple[int, ...] = (22, 23, 24)
    #: Downsample factor for the cost surface. 1 m cells over the full bbox is a very
    #: large grid; 3 m is ample for connector routing and ~9x cheaper.
    cost_surface_res_m: float = 3.0


@dataclass(frozen=True)
class RaceParams:
    #: Total time budget, seconds. 7 hours.
    time_budget_s: float = 7 * 3600
    #: Points per fully-completed trail.
    points_per_trail: float = 1.0
    #: Points per unique mile of trail traversed.
    points_per_mile: float = 1.0
    max_trail_points: float = 40.0
    max_mile_points: float = 40.0


@dataclass(frozen=True)
class Config:
    tobler: ToblerParams = field(default_factory=ToblerParams)
    topology: TopologyParams = field(default_factory=TopologyParams)
    elevation: ElevationParams = field(default_factory=ElevationParams)
    bushwhack: BushwhackParams = field(default_factory=BushwhackParams)
    race: RaceParams = field(default_factory=RaceParams)

    def with_tobler(self, **kw) -> "Config":
        return replace(self, tobler=replace(self.tobler, **kw))

    def with_topology(self, **kw) -> "Config":
        return replace(self, topology=replace(self.topology, **kw))

    def with_race(self, **kw) -> "Config":
        return replace(self, race=replace(self.race, **kw))


CONFIG = Config()
