"""
scrape_laws.py — One-time setup script to ingest tenant-rights statutes
for all 50 US states into ChromaDB.

Usage:
    cd lease-lens
    source .venv/bin/activate
    python data/scrape_laws.py              # all states (2s delay)
    python data/scrape_laws.py CA NY TX     # specific states only
"""

import sys
import pathlib
import datetime

# Make sure app/ is importable when run from the project root or data/
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from app.ingest import ingest_all_states, ingest_state_law, STATE_URLS


def main():
    log_dir = pathlib.Path(__file__).parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"scrape_log_{timestamp}.txt"

    # Redirect print output to both stdout and a log file
    import io

    class Tee(io.TextIOBase):
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data):
            for s in self._streams:
                s.write(data)
                s.flush()
            return len(data)

    with open(log_path, "w") as log_file:
        tee = Tee(sys.stdout, log_file)
        sys.stdout = tee

        try:
            states_arg = [s.upper() for s in sys.argv[1:]]

            if states_arg:
                unknown = [s for s in states_arg if s not in STATE_URLS]
                if unknown:
                    print(f"Unknown state(s): {', '.join(unknown)}")
                    print(f"Valid states: {', '.join(sorted(STATE_URLS.keys()))}")
                    sys.exit(1)

                results = {}
                for i, state in enumerate(states_arg, start=1):
                    print(f"\n[{i}/{len(states_arg)}] Ingesting {state} ...")
                    try:
                        results[state] = ingest_state_law(state)
                    except Exception as exc:
                        print(f"ERROR — {exc}")
                        results[state] = 0
            else:
                results = ingest_all_states(delay=2.0)

            print("\n── Summary ──────────────────────────────")
            for state, count in sorted(results.items()):
                status = f"{count} chunks" if count > 0 else "FAILED"
                print(f"  {state}: {status}")

        finally:
            sys.stdout = sys.__stdout__

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
