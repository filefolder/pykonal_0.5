#!/usr/bin/env python3
"""
pykonal_relocate.py

Batch hypocenter relocation utility for PyKonal & SeisComP.
Accepts a JSON configuration file and processes inputs from:
  1. QuakeML XML catalog files (single or multi-event)
  2. Simple whitespace-delimited text files
"""

import argparse
import datetime
import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import pykonal
from pykonal.locate import EQLocator
from pykonal.transformations import geo2sph, sph2geo


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class PhasePick:
    network: str
    station: str
    phase: str
    arrival_time: float  # Unix timestamp in seconds
    uncertainty: float = 0.1
    channel: str = ""
    location_code: str = ""
    pick_id: str = ""


@dataclass
class EventHypocenter:
    event_id: str
    latitude: float
    longitude: float
    depth_km: float
    origin_time: float  # Unix timestamp in seconds
    picks: List[PhasePick] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def parse_time(time_str: str) -> float:
    """Parse float Unix epoch or ISO 8601 string to Unix timestamp in seconds."""
    s = str(time_str).strip()
    try:
        return float(s)
    except ValueError:
        s_clean = s.replace("Z", "+00:00")
        return datetime.datetime.fromisoformat(s_clean).timestamp()


def format_iso_time(timestamp: float) -> str:
    """Convert float epoch seconds to ISO 8601 UTC string."""
    dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def parse_simple_text(filepath: str) -> List[EventHypocenter]:
    """
    Parses simple text format:
      hypocenter_lat hypocenter_lon hypocenter_depth hypocenter_time num_phases
      NET STA PHASE arrival_time [uncertainty]
    """
    events: List[EventHypocenter] = []
    current_event: Optional[EventHypocenter] = None
    remaining_phases = 0
    event_counter = 1

    with open(filepath, "r") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if remaining_phases == 0:
                if len(parts) < 5:
                    raise ValueError(
                        f"Line {line_num}: Header requires at least 5 fields "
                        f"(lat lon depth time num_phases), got: '{line}'"
                    )
                lat = float(parts[0])
                lon = float(parts[1])
                depth = float(parts[2])
                otime = parse_time(parts[3])
                remaining_phases = int(parts[4])

                current_event = EventHypocenter(
                    event_id=f"event_{event_counter:04d}",
                    latitude=lat,
                    longitude=lon,
                    depth_km=depth,
                    origin_time=otime,
                    picks=[],
                )
                events.append(current_event)
                event_counter += 1
            else:
                if len(parts) < 4:
                    raise ValueError(
                        f"Line {line_num}: Phase entry requires (net sta phase arrival_time [unc]), got: '{line}'"
                    )
                net = parts[0]
                sta = parts[1]
                pha = parts[2].upper()
                arr_t = parse_time(parts[3])
                unc = float(parts[4]) if len(parts) >= 5 else 0.1

                current_event.picks.append(
                    PhasePick(
                        network=net,
                        station=sta,
                        phase=pha,
                        arrival_time=arr_t,
                        uncertainty=unc,
                    )
                )
                remaining_phases -= 1

    return events


def parse_quakeml(filepath: str) -> List[EventHypocenter]:
    """Parses QuakeML XML catalog file into EventHypocenter structures."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # Index all picks in document
    pick_dict: Dict[str, PhasePick] = {}
    for p in root.iter(f"{ns}pick"):
        pid = p.attrib.get("publicID", "")
        t_elem = p.find(f"{ns}time/{ns}value")
        if t_elem is None or not t_elem.text:
            continue
        arr_time = parse_time(t_elem.text)

        unc_elem = p.find(f"{ns}time/{ns}uncertainty")
        unc = float(unc_elem.text) if (unc_elem is not None and unc_elem.text) else 0.1

        wf = p.find(f"{ns}waveformID")
        net = wf.attrib.get("networkCode", "") if wf is not None else ""
        sta = wf.attrib.get("stationCode", "") if wf is not None else ""
        cha = wf.attrib.get("channelCode", "") if wf is not None else ""
        loc = wf.attrib.get("locationCode", "") if wf is not None else ""

        ph_elem = p.find(f"{ns}phaseHint")
        phase = ph_elem.text.upper() if (ph_elem is not None and ph_elem.text) else "P"

        pick_dict[pid] = PhasePick(
            network=net,
            station=sta,
            phase=phase,
            arrival_time=arr_time,
            uncertainty=unc,
            channel=cha,
            location_code=loc,
            pick_id=pid,
        )

    # Process events
    events: List[EventHypocenter] = []
    event_elems = list(root.iter(f"{ns}event"))
    if not event_elems:
        event_elems = [root]

    for ev_elem in event_elems:
        ev_id = ev_elem.attrib.get("publicID", f"event_{len(events)+1}")

        # Choose preferred origin or first origin found
        target_orig = None
        pref_id = ev_elem.find(f"{ns}preferredOriginID")
        if pref_id is not None and pref_id.text:
            for o in ev_elem.iter(f"{ns}origin"):
                if o.attrib.get("publicID") == pref_id.text:
                    target_orig = o
                    break
        if target_orig is None:
            target_orig = ev_elem.find(f"{ns}origin")
        if target_orig is None:
            continue

        lat_e = target_orig.find(f"{ns}latitude/{ns}value")
        lon_e = target_orig.find(f"{ns}longitude/{ns}value")
        dep_e = target_orig.find(f"{ns}depth/{ns}value")
        t_e = target_orig.find(f"{ns}time/{ns}value")

        if any(x is None or not x.text for x in [lat_e, lon_e, dep_e, t_e]):
            continue

        lat = float(lat_e.text)
        lon = float(lon_e.text)
        # QuakeML depths are in meters; PyKonal expects km
        depth_km = float(dep_e.text) / 1000.0
        otime = parse_time(t_e.text)

        ev_obj = EventHypocenter(
            event_id=ev_id,
            latitude=lat,
            longitude=lon,
            depth_km=depth_km,
            origin_time=otime,
            picks=[],
        )

        for arrival in target_orig.iter(f"{ns}arrival"):
            pk_ref = arrival.find(f"{ns}pickID")
            ph_val = arrival.find(f"{ns}phase")
            if pk_ref is not None and pk_ref.text in pick_dict:
                p = pick_dict[pk_ref.text]
                if ph_val is not None and ph_val.text:
                    p.phase = ph_val.text.upper()
                ev_obj.picks.append(p)

        events.append(ev_obj)

    return events


# ---------------------------------------------------------------------------
# Batch Relocator Engine
# ---------------------------------------------------------------------------
class BatchRelocator:
    def __init__(self, config_data: dict):
        self.cfg = config_data
        self.coord_sys = self.cfg.get("coord_sys", "spherical")
        self.inventory_path = self.cfg["inventory_path"]
        self.method = self.cfg.get("method", "edt").lower()
        self.alpha = float(self.cfg.get("alpha", 0.05))
        self.min_picks = int(self.cfg.get("min_picks", 4))
        self.default_pick_error = float(self.cfg.get("default_pick_error", 0.1))

        # Inversion step sizes: [dr/dx, dtheta/dy, dphi/dz, dt]
        if "delta" in self.cfg:
            self.delta = np.array(self.cfg["delta"], dtype=np.float64)
        else:
            if self.coord_sys == "spherical":
                self.delta = np.array([2.0, np.radians(0.02), np.radians(0.02), 0.5])
            else:
                self.delta = np.array([2.0, 2.0, 2.0, 0.5])

        # Load inventory
        self.inventory = pykonal.inventory.TraveltimeInventory(
            self.inventory_path,
            mode="r",
        )

    def _geo_to_internal(self, lat: float, lon: float, depth_km: float) -> np.ndarray:
        if self.coord_sys == "spherical":
            return geo2sph(np.array([lat, lon, depth_km]))
        else:
            # Cartesian convention
            ref_lat = self.cfg.get("ref_lat", 0.0)
            ref_lon = self.cfg.get("ref_lon", 0.0)
            from geographiclib.geodesic import Geodesic
            geod = Geodesic.WGS84.Inverse(ref_lat, ref_lon, lat, lon)
            s12 = geod["s12"] / 1000.0  # km
            azi = np.radians(geod["azi1"])
            x = s12 * np.sin(azi)
            y = s12 * np.cos(azi)
            z = depth_km
            return np.array([x, y, z])

    def _internal_to_geo(self, coords: np.ndarray) -> Tuple[float, float, float]:
        if self.coord_sys == "spherical":
            res = sph2geo(coords[:3])
            return float(res[0]), float(res[1]), float(res[2])
        else:
            ref_lat = self.cfg.get("ref_lat", 0.0)
            ref_lon = self.cfg.get("ref_lon", 0.0)
            from geographiclib.geodesic import Geodesic
            x, y, z = coords[0], coords[1], coords[2]
            s12 = np.sqrt(x**2 + y**2) * 1000.0
            azi = np.degrees(np.arctan2(x, y))
            direct = Geodesic.WGS84.Direct(ref_lat, ref_lon, azi, s12)
            return float(direct["lat2"]), float(direct["lon2"]), float(z)

    def relocate_event(self, ev: EventHypocenter) -> Optional[dict]:
        if len(ev.picks) < self.min_picks:
            logging.warning(f"{ev.event_id}: insufficient picks ({len(ev.picks)} < {self.min_picks})")
            return None

        locator = EQLocator(
            coord_sys=self.coord_sys,
            alpha=self.alpha,
            norm=1 if self.method == "edt" else 2,
        )

        # Set EDT parameters if present
        if "edt_ot_wt" in self.cfg:
            locator.edt_ot_wt = float(self.cfg["edt_ot_wt"])
        if "edt_exponent" in self.cfg:
            locator.edt_exponent = float(self.cfg["edt_exponent"])

        used_picks = 0
        for p in ev.picks:
            key = f"{p.network}_{p.station}_{p.phase}"
            if key not in self.inventory:
                # Try station_phase fallback
                key = f"{p.station}_{p.phase}"
                if key not in self.inventory:
                    continue

            unc = p.uncertainty if p.uncertainty > 0 else self.default_pick_error
            locator.add_arrival(self.inventory[key], p.arrival_time, uncertainty=unc)
            used_picks += 1

        if used_picks < self.min_picks:
            logging.warning(f"{ev.event_id}: only {used_picks} picks available in inventory.")
            return None

        init_spatial = self._geo_to_internal(ev.latitude, ev.longitude, ev.depth_km)
        init_4vec = np.array(
            [init_spatial[0], init_spatial[1], init_spatial[2], ev.origin_time],
            dtype=np.float64,
        )

        try:
            sol_4vec = locator.locate(
                initial=init_4vec,
                delta=self.delta,
                method=self.method,
            )
        except Exception as e:
            logging.error(f"{ev.event_id}: Inversion failed: {e}")
            return None

        reloc_lat, reloc_lon, reloc_depth = self._internal_to_geo(sol_4vec[:3])
        reloc_time = sol_4vec[3]
        residuals = locator.get_residuals(sol_4vec)
        rms = np.sqrt(np.mean(residuals**2))

        return {
            "event_id": ev.event_id,
            "orig_lat": ev.latitude,
            "orig_lon": ev.longitude,
            "orig_depth": ev.depth_km,
            "orig_time": format_iso_time(ev.origin_time),
            "reloc_lat": reloc_lat,
            "reloc_lon": reloc_lon,
            "reloc_depth": reloc_depth,
            "reloc_time": format_iso_time(reloc_time),
            "rms_sec": float(rms),
            "num_phases": used_picks,
        }

    def close(self):
        if hasattr(self, "inventory") and self.inventory:
            self.inventory.close()


# ---------------------------------------------------------------------------
# Output Formatter
# ---------------------------------------------------------------------------
def write_results_text(results: List[dict], out_path: str):
    with open(out_path, "w") as f:
        f.write("# event_id reloc_lat reloc_lon reloc_depth_km reloc_time rms_sec num_phases\n")
        for r in results:
            f.write(
                f"{r['event_id']} {r['reloc_lat']:.5f} {r['reloc_lon']:.5f} "
                f"{r['reloc_depth']:.3f} {r['reloc_time']} "
                f"{r['rms_sec']:.4f} {r['num_phases']}\n"
            )


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Batch hypocenter relocator for QuakeML and Simple Text files using PyKonal."
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to JSON configuration file (e.g. pykonal_locext.json).",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input file (QuakeML .xml/.qml or simple text .txt/.dat).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="relocated_catalog.txt",
        help="Path to output text file (default: relocated_catalog.txt).",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "quakeml", "text"],
        default="auto",
        help="Input file format (default: auto).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    # Load JSON Config
    if not os.path.isfile(args.config):
        logging.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config_data = json.load(f)

    # Detect Format
    fmt = args.format
    if fmt == "auto":
        ext = os.path.splitext(args.input)[1].lower()
        fmt = "quakeml" if ext in [".xml", ".qml", ".quakeml"] else "text"

    # Ingest Hypocenters
    logging.info(f"Parsing input events from '{args.input}' using format='{fmt}'...")
    if fmt == "quakeml":
        events = parse_quakeml(args.input)
    else:
        events = parse_simple_text(args.input)

    logging.info(f"Loaded {len(events)} event(s) to relocate.")
    if not events:
        logging.warning("No events found to process.")
        return

    # Execute Relocation
    relocator = BatchRelocator(config_data)
    results = []

    try:
        for idx, ev in enumerate(events, start=1):
            logging.info(f"[{idx}/{len(events)}] Relocating {ev.event_id} ({len(ev.picks)} picks)...")
            res = relocator.relocate_event(ev)
            if res:
                results.append(res)
                logging.info(
                    f" -> {ev.event_id} => Lat: {res['reloc_lat']:.4f}, Lon: {res['reloc_lon']:.4f}, "
                    f"Depth: {res['reloc_depth']:.2f} km | RMS: {res['rms_sec']:.3f} s"
                )
    finally:
        relocator.close()

    # Save Output
    write_results_text(results, args.output)
    logging.info(f"Relocation complete. {len(results)}/{len(events)} events written to {args.output}")


if __name__ == "__main__":
    main()