#!/usr/bin/env python3
"""Read-only V100 benchmark telemetry, flushed to disk to investigate unexpected resets.

No power, clock, driver, kernel or reboot settings are changed. Sampling cannot preserve an
event that was never delivered by the kernel or that happened after the last disk sync.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


GPU_FIELDS = (
    "timestamp,index,pci.bus_id,temperature.gpu,temperature.memory,power.draw,power.limit,"
    "utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.mem,pstate,ecc.mode.current,"
    "ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total,"
    "ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,"
    "clocks_event_reasons.active,clocks_event_reasons.sw_power_cap,"
    "clocks_event_reasons.hw_slowdown,clocks_event_reasons.hw_thermal_slowdown,"
    "clocks_event_reasons.hw_power_brake_slowdown,power.draw.instant,"
    "pcie.link.gen.current,pcie.link.width.current"
)


def durable_json(path, value):
    """Keep either the previous or the complete new report across an interrupted write."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class HealthLog:
    def __init__(self, directory, interval=0.2):
        self.directory = Path(directory)
        self.interval = interval
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.processes = []
        self.threads = []
        self.files = []
        self.boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def open_log(self, name):
        output = (self.directory / name).open("x")
        self.files.append(output)
        return output

    def write(self, output, line):
        with self.lock:
            output.write(line.rstrip("\n") + "\n")
            output.flush()
            os.fsync(output.fileno())

    def event(self, event, **fields):
        self.write(self.events, json.dumps({
            "time": datetime.now(timezone.utc).isoformat(), "monotonic_ns": time.monotonic_ns(),
            "boot_id": self.boot_id, "event": event, **fields}, ensure_ascii=False))

    def snapshot(self, name):
        result = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, timeout=15)
        with self.open_log(name) as output:
            self.write(output, result.stdout + result.stderr)
        self.event("gpu_snapshot", file=name, returncode=result.returncode)

    def pipe(self, name, command):
        output = self.open_log(name)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        self.processes.append(process)

        def collect():
            try:
                for line in process.stdout:
                    self.write(output, line)
                code = process.wait()
                if not self.stop.is_set():
                    self.event("collector_exited", file=name, returncode=code)
            except Exception as error:
                self.event("collector_error", file=name, error=repr(error))

        thread = threading.Thread(target=collect, daemon=True)
        self.threads.append(thread)
        thread.start()

    def follow_file(self, path, name, *, from_start=True):
        # Preserve existing application logs without changing product logging or CUDA execution.
        self.pipe(name, ["tail", "-n", "+1" if from_start else "0", "-F", "-s", "0.1", "--", str(path)])

    def attach_pid(self, pid):
        """Sample only the selected workload, not other jobs in the user's session cgroup."""
        base = Path("/proc") / str(pid)
        # starttime identifies this process even if a PID is subsequently reused.
        identity = (base / "stat").read_text().rsplit(")", 1)[1].split()[19]
        output = self.open_log(f"workload-{pid}.jsonl")
        self.event("process_attached", pid=pid, starttime_ticks=identity)

        def collect():
            while not self.stop.is_set():
                try:
                    stat = (base / "stat").read_text().strip()
                    if stat.rsplit(")", 1)[1].split()[19] != identity:
                        self.event("process_pid_reused", pid=pid)
                        return
                    status = {key: value.strip() for key, value in
                        (line.split(":", 1) for line in (base / "status").read_text().splitlines())
                        if key in ("Name", "State", "VmRSS", "VmHWM", "VmSize", "VmLck", "VmSwap",
                                   "Threads", "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches")}
                    sample = {"time": datetime.now(timezone.utc).isoformat(),
                              "monotonic_ns": time.monotonic_ns(), "boot_id": self.boot_id,
                              "pid": pid, "stat": stat, "status": status, "threads": []}
                    for name in ("io", "schedstat", "wchan", "limits", "cgroup"):
                        try:
                            sample[name] = (base / name).read_text().strip()
                        except OSError:
                            pass
                    for task in (base / "task").iterdir():
                        try:
                            sample["threads"].append({"tid": int(task.name),
                                "name": (task / "comm").read_text().strip(),
                                "stat": (task / "stat").read_text().strip(),
                                "wchan": (task / "wchan").read_text().strip(),
                                "schedstat": (task / "schedstat").read_text().strip()})
                        except OSError:
                            pass
                    self.write(output, json.dumps(sample))
                except FileNotFoundError:
                    self.event("process_exited", pid=pid)
                    return
                except Exception as error:
                    self.event("process_sample_error", pid=pid, error=repr(error))
                self.stop.wait(1.0)

        thread = threading.Thread(target=collect, daemon=True)
        self.threads.append(thread)
        thread.start()

    def attach_cgroup(self, path):
        path = Path(path)
        output = self.open_log("workload.jsonl")
        self.event("cgroup_attached", path=str(path))

        def collect():
            while not self.stop.is_set():
                try:
                    counters = {}
                    for name in ("memory.current", "memory.peak", "memory.events", "memory.stat",
                                 "memory.max", "memory.swap.max", "cpu.stat", "cpu.pressure",
                                 "memory.pressure", "io.pressure", "io.stat", "cgroup.events"):
                        item = path / name
                        if item.exists():
                            counters[name] = item.read_text().strip()
                    processes = []
                    for pid in (path / "cgroup.procs").read_text().split():
                        base = Path("/proc") / pid
                        try:
                            status = {key: value.strip() for key, value in
                                      (line.split(":", 1) for line in (base / "status").read_text().splitlines())
                                      if key in ("Name", "State", "VmRSS", "VmHWM", "VmSize", "VmLck",
                                                 "Threads", "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches")}
                            threads = []
                            for task in (base / "task").iterdir():
                                try:
                                    threads.append({"tid": int(task.name),
                                        "name": (task / "comm").read_text().strip(),
                                        "stat": (task / "stat").read_text().strip(),
                                        "wchan": (task / "wchan").read_text().strip(),
                                        "schedstat": (task / "schedstat").read_text().strip()})
                                except OSError:
                                    pass
                            process = {"pid": int(pid), "status": status, "threads": threads}
                            try:
                                process["io"] = (base / "io").read_text().strip()
                            except OSError:
                                pass
                            processes.append(process)
                        except OSError:
                            pass
                    self.write(output, json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                        "monotonic_ns": time.monotonic_ns(), "boot_id": self.boot_id,
                        "cgroup": counters, "processes": processes}))
                except FileNotFoundError:
                    self.event("cgroup_removed", path=str(path))
                    return
                except Exception as error:
                    self.event("workload_sample_error", error=repr(error))
                self.stop.wait(1.0)

        thread = threading.Thread(target=collect, daemon=True)
        self.threads.append(thread)
        thread.start()

    def host_samples(self):
        output = self.open_log("host.jsonl")
        sensors = sorted(Path("/sys/class/hwmon").glob("hwmon*/temp*_input"))
        pci_devices = set()
        for device in ("0000:03:00.0", "0000:04:00.0"):
            resolved = Path("/sys/bus/pci/devices", device).resolve()
            pci_devices.update(parent for parent in (resolved, *resolved.parents)
                               if (parent / "vendor").exists())
        pci_counters = [path for device in sorted(pci_devices) for path in device.glob("aer_*")]
        edac = [path for controller in Path("/sys/devices/system/edac/mc").glob("mc[0-9]*")
                for pattern in ("ce_count", "ue_count", "ce_noinfo_count", "ue_noinfo_count",
                                "dimm*/dimm_ce_count", "dimm*/dimm_ue_count")
                for path in controller.glob(pattern)]
        hwmon = [path for chip in Path("/sys/class/hwmon").glob("hwmon*")
                 for pattern in ("temp*_crit_alarm", "in*_input", "curr*_input", "power*_input", "fan*_input")
                 for path in chip.glob(pattern)]
        previous = None
        while not self.stop.is_set():
            try:
                cpu = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
                total, idle = sum(cpu), cpu[3] + cpu[4]
                busy = None if previous is None else (
                    100 * (1 - (idle - previous[1]) / max(1, total - previous[0])))
                previous = (total, idle)
                memory = {key: int(value.split()[0]) * 1024 for key, value in
                          (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                          if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")}
                temperatures, aer = {}, {}
                for path in sensors:
                    try:
                        label = path.with_name(path.name.replace("_input", "_label"))
                        chip = (path.parent / "name").read_text().strip()
                        name = label.read_text().strip() if label.exists() else path.stem
                        temperatures[f"{path.parent.name}/{chip}/{name}"] = int(path.read_text()) / 1000
                    except OSError:
                        pass
                for path in pci_counters:
                    try:
                        aer[f"{path.parent.name}/{path.name}"] = path.read_text().strip()
                    except OSError:
                        pass
                extra = {}
                for path in (*edac, *hwmon):
                    try:
                        extra[str(path)] = path.read_text().strip()
                    except OSError:
                        pass
                pressures = {name: Path("/proc/pressure", name).read_text().strip()
                             for name in ("cpu", "memory", "io")}
                self.write(output, json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                    "monotonic_ns": time.monotonic_ns(), "boot_id": self.boot_id,
                    "cpu_busy_percent": busy, "memory_bytes": memory,
                    "temperature_c": temperatures, "pci_aer": aer,
                    "edac_hwmon_raw": extra, "pressure": pressures}))
            except Exception as error:
                self.event("host_sample_error", error=repr(error))
            self.stop.wait(1.0)

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events = self.open_log("events.jsonl")
        self.event("health_start", gpu_interval_seconds=self.interval, host_interval_seconds=1.0,
                   kernel_command_line=Path("/proc/cmdline").read_text().strip())
        try:
            self.snapshot("gpu-start.txt")
            topology = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                                      text=True, timeout=15)
            with self.open_log("gpu-topology.txt") as output:
                self.write(output, topology.stdout + topology.stderr)
            self.event("gpu_topology", returncode=topology.returncode)
            self.pipe("gpu.csv", ["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv",
                                  "-lms", str(round(self.interval * 1000))])
            self.pipe("gpu-dmon.log", ["stdbuf", "-oL", "-eL", "nvidia-smi", "dmon", "-s", "pucvmet",
                                       "-o", "DT", "-d", "1"])
            self.pipe("kernel.jsonl", ["journalctl", "-b", "-k", "-f", "-n", "50", "-o", "json",
                                       "--no-pager"])
            self.pipe("system.jsonl", ["journalctl", "-b", "-f", "-n", "50", "-o", "json", "--no-pager",
                                       "_PID=1", "+", "_SYSTEMD_UNIT=systemd-logind.service", "+",
                                       "_SYSTEMD_UNIT=systemd-oomd.service", "+",
                                       "_SYSTEMD_UNIT=systemd-journald.service"])
            thread = threading.Thread(target=self.host_samples, daemon=True)
            self.threads.append(thread)
            thread.start()
            descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        self.stop.set()
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for thread in self.threads:
            thread.join(timeout=5)
        try:
            self.snapshot("gpu-end.txt")
            self.event("health_stop")
        finally:
            for output in self.files:
                if not output.closed:
                    output.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.2, help="GPU sampling seconds; host sampling is 1 second")
    parser.add_argument("--pid", type=int, help="also sample this workload's process and threads")
    args = parser.parse_args()
    if args.interval < 0.1:
        parser.error("interval must be at least 0.1 seconds")
    with HealthLog(args.output, args.interval) as health:
        if args.pid is not None:
            health.attach_pid(args.pid)
        for number in (signal.SIGINT, signal.SIGTERM):
            signal.signal(number, lambda *_: health.stop.set())
        while not health.stop.wait(1):
            pass


if __name__ == "__main__":
    main()
