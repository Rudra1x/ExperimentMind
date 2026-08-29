import os
import sys
import time
import threading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from agents.validator import validate_config

WATCH_DIR = os.getenv("WATCH_DIR", "experiments/queue")


class ConfigFileHandler(FileSystemEventHandler):
    """Watchdog handler — fires when a new YAML appears in the queue folder."""

    def __init__(self):
        super().__init__()
        self._processing = set()  # track files being processed
        self._lock = threading.Lock()

    def on_created(self, event):
        if event.is_directory:
            return

        path = event.src_path
        if not (path.endswith('.yaml') or path.endswith('.yml')):
            return

        # Debounce — avoid double-firing on some editors
        with self._lock:
            if path in self._processing:
                return
            self._processing.add(path)

        print(f"\n[Watcher] New config detected: {path}")

        # Small delay to ensure file is fully written before reading
        time.sleep(0.5)

        try:
            validate_config(path)
        except Exception as e:
            print(f"[Watcher] Error processing {path}: {e}")
        finally:
            with self._lock:
                self._processing.discard(path)


class ConfigWatcherAgent:
    """
    Config Watcher Agent — monitors the queue folder for new experiment configs
    and triggers the Validator Agent automatically.
    """

    def __init__(self, watch_dir: str = WATCH_DIR):
        self.watch_dir = watch_dir
        self.observer = None
        os.makedirs(watch_dir, exist_ok=True)
        print(f"[Watcher] Initialized. Watching: {os.path.abspath(watch_dir)}")

    def start(self, block: bool = True):
        """Start watching the queue folder."""
        handler = ConfigFileHandler()

        # PollingObserver works reliably on Windows + network drives
        self.observer = PollingObserver()
        self.observer.schedule(handler, self.watch_dir, recursive=False)
        self.observer.start()

        print(f"[Watcher] Started. Drop a .yaml config into:")
        print(f"          {os.path.abspath(self.watch_dir)}")
        print(f"[Watcher] Press Ctrl+C to stop.\n")

        if block:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()

    def stop(self):
        """Stop the watcher."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        print("[Watcher] Stopped.")

    def process_existing(self):
        """Process any configs already sitting in the queue on startup."""
        existing = [
            os.path.join(self.watch_dir, f)
            for f in os.listdir(self.watch_dir)
            if f.endswith('.yaml') or f.endswith('.yml')
        ]
        if existing:
            print(f"[Watcher] Found {len(existing)} existing config(s) in queue.")
            for config_path in existing:
                validate_config(config_path)
        return existing


if __name__ == "__main__":
    watcher = ConfigWatcherAgent()
    watcher.process_existing()
    watcher.start(block=True)