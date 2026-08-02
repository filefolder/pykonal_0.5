#!/usr/bin/env python3
"""
Bulk relocation driver for the pykonal LocExt plugin.

Runs the SAME code path as the SeisComP plugin (pykonal_locext.main), but
over many events in a single process, so you pay Python/pykonal import and
config/transform setup ONCE instead of per event. Each input is a SeisComP
EventParameters XML document (as produced by scxmldump); each output is the
relocated bare <Origin> the plugin emits, ready to feed to scdispatch.

Usage:
    pykonal_batch.py --config pykonal_locext.json \
        --in-dir  /path/to/exported_events/ \
        --out-dir /path/to/relocated_origins/ \
        [--pattern '*.xml'] [--continue-on-error] [--merged FILE]

Each <in-dir>/<name>.xml -> <out-dir>/<name>.origin.xml (bare Origin).
Per-event diagnostics (residuals, ess, dropped arrivals) go to
<out-dir>/<name>.log. A --merged file, if given, collects every relocated
Origin into one EventParameters document for a single scdispatch call.

Re-import with scdispatch, e.g. per file:
    scdispatch -O merge -H host -i reloc.origin.xml
or one shot with --merged (origins share the original picks already in the
database, referenced by publicID).

NOTE: the plugin's stdout contract is a BARE <Origin>, not EventParameters,
and it carries NO picks -- only arrivals referencing existing pick
publicIDs. The relocated origin is therefore only meaningful re-imported
against a database (or catalog) that still holds those picks. Bulk
relocation replaces the hypocenter and echoes arrival weights/timeUsed; it
does not re-emit picks.
"""
import argparse
import glob
import io
import os
import sys
import traceback

# import the plugin module (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pykonal_locext as plx


def relocate_bytes(input_xml_bytes):
    """
    Relocate one event given its EventParameters XML as bytes; return the
    relocated Origin as XML bytes. Runs the plugin's main() with stdin and
    stdout swapped for in-memory buffers so no code path diverges from the
    production plugin.
    """
    # feed the input via a fake stdin the plugin's read_input() will read
    fake_stdin = io.BytesIO(input_xml_bytes)

    captured = {}

    def _fake_read_input():
        return plx.read_input_from_bytes(input_xml_bytes)

    def _fake_write_output(origin):
        captured["bytes"] = plx.origin_to_bytes(origin)

    # swap the two I/O functions; main() does everything in between
    real_read, real_write = plx.read_input, plx.write_output
    plx.read_input = _fake_read_input
    plx.write_output = _fake_write_output
    # main() dups fd1->fd2 for stdout safety; harmless here but do it once
    # by giving it a real _REAL_STDOUT it can write to (unused, we capture).
    try:
        plx.main()
    finally:
        plx.read_input, plx.write_output = real_read, real_write

    if "bytes" not in captured:
        raise RuntimeError("relocation produced no origin")
    return captured["bytes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pattern", default="*.xml")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--merged", default=None,
                    help="also write all origins into one EventParameters doc")
    # pass-through plugin flags
    ap.add_argument("--max-dist", type=float, default=None)
    ap.add_argument("--fixed-depth", type=float, default=None)
    ap.add_argument("--ignore-initial-location", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # the plugin reads its flags from sys.argv via argparse in main(); set
    # them once so every event sees the same options.
    plugin_argv = ["pykonal_locext.py", "--config", args.config]
    if args.max_dist is not None:
        plugin_argv += ["--max-dist", str(args.max_dist)]
    if args.fixed_depth is not None:
        plugin_argv += ["--fixed-depth", str(args.fixed_depth)]
    if args.ignore_initial_location:
        plugin_argv += ["--ignore-initial-location"]

    files = sorted(glob.glob(os.path.join(args.in_dir, args.pattern)))
    if not files:
        print(f"no files matching {args.pattern} in {args.in_dir}",
              file=sys.stderr)
        return 1
    print(f"relocating {len(files)} event(s)", file=sys.stderr)

    merged_origins = []
    ok = 0
    fail = 0
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.out_dir, f"{name}.origin.xml")
        log_path = os.path.join(args.out_dir, f"{name}.log")
        with open(path, "rb") as f:
            data = f.read()

        # per-event: capture the plugin's stderr diagnostics to a log file
        real_stderr_fd = os.dup(2)
        logf = open(log_path, "wb")
        os.dup2(logf.fileno(), 2)
        try:
            sys.argv = list(plugin_argv)
            origin_bytes = relocate_bytes(data)
            with open(out_path, "wb") as of:
                of.write(origin_bytes)
            merged_origins.append(origin_bytes)
            ok += 1
            status = "OK"
        except Exception:
            fail += 1
            status = "FAIL"
            traceback.print_exc()      # goes to the per-event log
            if not args.continue_on_error:
                os.dup2(real_stderr_fd, 2)
                os.close(real_stderr_fd)
                logf.close()
                print(f"{name}: FAILED (see {log_path})", file=sys.stderr)
                return 1
        finally:
            os.dup2(real_stderr_fd, 2)
            os.close(real_stderr_fd)
            logf.close()
        print(f"  {name}: {status}", file=sys.stderr)

    if args.merged and merged_origins:
        _write_merged(args.merged, merged_origins)
        print(f"merged {len(merged_origins)} origins -> {args.merged}",
              file=sys.stderr)

    print(f"done: {ok} ok, {fail} failed", file=sys.stderr)
    return 0 if fail == 0 else 2


def _write_merged(path, origin_bytes_list):
    """
    Wrap all relocated origins into a single EventParameters document for
    one scdispatch call. Origins reference existing pick publicIDs, so the
    target database must already hold the picks.
    """
    import seiscomp.datamodel as dm
    import seiscomp.io as sio
    import tempfile
    ep = dm.EventParameters()
    for b in origin_bytes_list:
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tf:
            tf.write(b)
            tp = tf.name
        ar = sio.XMLArchive()
        ar.open(tp)
        obj = ar.readObject()
        ar.close()
        os.unlink(tp)
        origin = dm.Origin.Cast(obj)
        if origin is not None:
            ep.add(origin)
    ar = sio.XMLArchive()
    ar.setFormattedOutput(True)
    ar.create(path)
    ar.writeObject(ep)
    ar.close()


if __name__ == "__main__":
    sys.exit(main())
