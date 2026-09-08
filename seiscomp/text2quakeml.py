#!/usr/bin/env python3
"""
text2quakeml.py

Converts simple text hypocenter/pick files into valid QuakeML (v1.2) XML documents.

Supported Text Format:
  hypocenter_lat hypocenter_lon hypocenter_depth_km hypocenter_time num_phases
  NET STA PHASE arrival_time [uncertainty]
"""

import argparse
import datetime
import os
import sys
import uuid
import xml.dom.minidom
import xml.etree.ElementTree as ET
from typing import List, Optional


def parse_timestamp(time_str: str) -> Tuple[datetime.datetime, str]:
    """Parse float Unix timestamp or ISO string into a UTC datetime and formatted ISO string."""
    time_str = str(time_str).strip()
    try:
        val = float(time_str)
        dt = datetime.datetime.fromtimestamp(val, tz=datetime.timezone.utc)
    except ValueError:
        clean = time_str.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
    
    iso_formatted = dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return dt, iso_formatted


def text_to_quakeml(
    input_text_path: str,
    output_xml_path: str,
    agency_id: str = "pykonal",
    author: str = "pykonal_batch",
    uri_prefix: str = "smi:org.pykonal",
) -> None:
    """
    Parses simple text format and serializes a fully valid QuakeML 1.2 XML tree.
    """
    QML_NS = "http://quakeml.org/xmlns/quakeml/1.2"
    BED_NS = "http://quakeml.org/xmlns/bed/1.2"

    ET.register_namespace("", BED_NS)
    ET.register_namespace("q", QML_NS)

    qml_root = ET.Element(
        f"{{{QML_NS}}}quakeml",
        {
            "xmlns": BED_NS,
            "xmlns:q": QML_NS,
        },
    )

    ev_params_id = f"{uri_prefix}/eventParameters/{uuid.uuid4()}"
    ev_params = ET.SubElement(
        qml_root,
        f"{{{BED_NS}}}eventParameters",
        {"publicID": ev_params_id},
    )

    creation_time = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )

    remaining_phases = 0
    event_idx = 0
    current_event_elem = None
    current_origin_elem = None
    current_event_picks = []

    with open(input_text_path, "r") as f:
        lines = f.readlines()

    for line_num, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()

        # -------------------------------------------------------------
        # 1. Event / Origin Header
        # -------------------------------------------------------------
        if remaining_phases == 0:
            if len(parts) < 5:
                raise ValueError(
                    f"Line {line_num}: Event header must have 5 tokens "
                    f"(lat lon depth time num_phases), found: '{line}'"
                )

            event_idx += 1
            lat = float(parts[0])
            lon = float(parts[1])
            depth_m = float(parts[2]) * 1000.0  # QuakeML uses SI meters
            _, time_iso = parse_timestamp(parts[3])
            remaining_phases = int(parts[4])

            # Generate unique IDs
            ev_uuid = f"ev_{datetime.datetime.now().strftime('%Y%m%d')}_{event_idx:04d}_{uuid.uuid4().hex[:6]}"
            orig_uuid = f"orig_{ev_uuid}"

            event_id = f"{uri_prefix}/event/{ev_uuid}"
            origin_id = f"{uri_prefix}/origin/{orig_uuid}"

            # Create <event>
            current_event_elem = ET.SubElement(
                ev_params,
                f"{{{BED_NS}}}event",
                {"publicID": event_id},
            )

            # Preferred origin reference
            pref_orig = ET.SubElement(current_event_elem, f"{{{BED_NS}}}preferredOriginID")
            pref_orig.text = origin_id

            # Event type
            ev_type = ET.SubElement(current_event_elem, f"{{{BED_NS}}}type")
            ev_type.text = "earthquake"

            # Create <origin>
            current_origin_elem = ET.SubElement(
                current_event_elem,
                f"{{{BED_NS}}}origin",
                {"publicID": origin_id},
            )

            # Origin Time
            t_elem = ET.SubElement(current_origin_elem, f"{{{BED_NS}}}time")
            ET.SubElement(t_elem, f"{{{BED_NS}}}value").text = time_iso

            # Latitude
            lat_elem = ET.SubElement(current_origin_elem, f"{{{BED_NS}}}latitude")
            ET.SubElement(lat_elem, f"{{{BED_NS}}}value").text = f"{lat:.6f}"

            # Longitude
            lon_elem = ET.SubElement(current_origin_elem, f"{{{BED_NS}}}longitude")
            ET.SubElement(lon_elem, f"{{{BED_NS}}}value").text = f"{lon:.6f}"

            # Depth
            depth_elem = ET.SubElement(current_origin_elem, f"{{{BED_NS}}}depth")
            ET.SubElement(depth_elem, f"{{{BED_NS}}}value").text = f"{depth_m:.1f}"

            # Creation Info
            ci = ET.SubElement(current_origin_elem, f"{{{BED_NS}}}creationInfo")
            ET.SubElement(ci, f"{{{BED_NS}}}agencyID").text = agency_id
            ET.SubElement(ci, f"{{{BED_NS}}}author").text = author
            ET.SubElement(ci, f"{{{BED_NS}}}creationTime").text = creation_time

        # -------------------------------------------------------------
        # 2. Phase Picks & Arrivals
        # -------------------------------------------------------------
        else:
            if len(parts) < 4:
                raise ValueError(
                    f"Line {line_num}: Phase pick requires at least 4 fields "
                    f"(NET STA PHASE arrival_time [uncertainty]), found: '{line}'"
                )

            net = parts[0]
            sta = parts[1]
            phase = parts[2].upper()
            _, arr_time_iso = parse_timestamp(parts[3])
            uncertainty = float(parts[4]) if len(parts) >= 5 else 0.1

            pick_uid = f"pick_{net}_{sta}_{phase}_{uuid.uuid4().hex[:8]}"
            pick_id = f"{uri_prefix}/pick/{pick_uid}"
            arrival_id = f"{uri_prefix}/arrival/{uuid.uuid4().hex[:8]}"

            # 2a. Add <pick> directly under <eventParameters> (standard QuakeML practice)
            pick_elem = ET.SubElement(
                ev_params,
                f"{{{BED_NS}}}pick",
                {"publicID": pick_id},
            )

            p_time = ET.SubElement(pick_elem, f"{{{BED_NS}}}time")
            ET.SubElement(p_time, f"{{{BED_NS}}}value").text = arr_time_iso
            ET.SubElement(p_time, f"{{{BED_NS}}}uncertainty").text = f"{uncertainty:.4f}"

            ET.SubElement(
                pick_elem,
                f"{{{BED_NS}}}waveformID",
                {
                    "networkCode": net,
                    "stationCode": sta,
                    "channelCode": "HHZ" if phase.startswith("P") else "HHE",
                    "locationCode": "",
                },
            )
            ET.SubElement(pick_elem, f"{{{BED_NS}}}phaseHint").text = phase
            
            p_ci = ET.SubElement(pick_elem, f"{{{BED_NS}}}creationInfo")
            ET.SubElement(p_ci, f"{{{BED_NS}}}agencyID").text = agency_id
            ET.SubElement(p_ci, f"{{{BED_NS}}}creationTime").text = creation_time

            # 2b. Add <arrival> linking the pick to the origin
            arrival_elem = ET.SubElement(
                current_origin_elem,
                f"{{{BED_NS}}}arrival",
                {"publicID": arrival_id},
            )
            ET.SubElement(arrival_elem, f"{{{BED_NS}}}pickID").text = pick_id
            ET.SubElement(arrival_elem, f"{{{BED_NS}}}phase").text = phase
            ET.SubElement(arrival_elem, f"{{{BED_NS}}}timeWeight").text = "1.0"

            remaining_phases -= 1

    # Pretty-print and save
    xml_str = ET.tostring(qml_root, encoding="utf-8")
    parsed_dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml = parsed_dom.toprettyxml(indent="  ", encoding="utf-8")

    with open(output_xml_path, "wb") as f:
        f.write(pretty_xml)


def main():
    parser = argparse.ArgumentParser(
        description="Convert simple text hypocenter format into valid QuakeML XML."
    )
    parser.add_argument("input", help="Path to input simple text file")
    parser.add_argument("output", help="Path to write QuakeML output (.xml or .qml)")
    parser.add_argument("--agency", default="pykonal", help="Agency ID (default: pykonal)")
    parser.add_argument("--author", default="pykonal_batch", help="Author string")

    args = parser.parse_args()
    text_to_quakeml(args.input, args.output, agency_id=args.agency, author=args.author)
    print(f"Successfully converted '{args.input}' -> '{args.output}'")


if __name__ == "__main__":
    main()