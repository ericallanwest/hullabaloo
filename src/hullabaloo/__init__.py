"""Hullabaloo — route optimization for a 7-hour trail rogaine at Pandapas Pond, VA.

Pipeline
--------
``ingest``    scrape the official trail list and GPX tracks
``topology``  planarize 40 overlapping polylines into a noded, routable network
``elevation`` sample 3DEP 1 m lidar and price every edge with a Tobler hiking function
``bushwhack`` least-cost off-trail connectors that join the disconnected components
``graph``     assemble the directed time-weighted graph and compute baselines
``optimize_alns`` / ``optimize_milp``  solve the prize-collecting arc routing problem
``export``    GeoPackage, GPX, cue sheet, maps
"""

__version__ = "0.1.0"

from .config import CONFIG, Config  # noqa: F401
