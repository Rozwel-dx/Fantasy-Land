#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from vllm.logger import logger

ALLOWED_CPUS_PATH = "/proc/self/status"


def execute_command(cmd: List[str]) -> Tuple[str, int]:
    with subprocess.Popen(
            cmd,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
    ) as p:
        out, _ = p.communicate(timeout=1000)
    return out.decode(), p.returncode


class DeviceInfo:
    def __init__(self):
        self.npu_map_info: Dict[str, Dict[str, str]] = self.get_npu_map_info()
        self.allowed_cpus: List[int] = self.parse_allowed_cpus()
        self.running_npu_list: List[int] = self.get_running_npus()
        self.npu_affinity: Dict[int, List[int]] = self.parse_topo_affinity()

    @staticmethod
    def expand_cpu_list(allowed_list_str: str) -> List[int]:
        allowed_cpus_list = []
        for per_range in allowed_list_str.split(","):
            if "-" in per_range:
                start_cpu, end_cpu = map(int, per_range.split("-"))
                allowed_cpus_list.extend(range(start_cpu, end_cpu + 1))
            else:
                allowed_cpus_list.append(int(per_range))
        return allowed_cpus_list

    @staticmethod
    def get_npu_map_info() -> Dict[str, Dict[str, str]]:
        npu_map_info = {}
        npu_info, _ = execute_command(["npu-info", "info", "-m"])
        npu_map = npu_info.strip().split("\n")[1:]
        for line in npu_map:
            npu_id, chip_id, chip_logic_id = line.strip().split()[:3]
            if not chip_logic_id.isdigit():
                continue
            if npu_id not in npu_map_info:
                npu_map_info[npu_id] = {}
            npu_map_info[npu_id][chip_id] = chip_logic_id
        return npu_map_info

    def get_running_npus(self) -> List[int]:
        npu_message, _ = execute_command(["npu-message", "info"])
        in_proc_section = False
        running_npu_set = set()
        for line in npu_message.splitlines():
            line = line.strip()
            if line.startswith("| NPU") and "Process id" in line:
                in_proc_section = True
                continue
            if not in_proc_section:
                continue
            if line.startswith("| "):
                parts = [p.strip() for p in line.strip("|").split("|")]
                if len(parts) >= 2:
                    npu_id = parts[0].split()[0]
                    chip_id = parts[0].split()[1]
                    if not npu_id.isdigit() or not chip_id.isdigit():
                        continue
                    chip_logic_id = self.npu_map_info.get(npu_id, {}).get(chip_id)
                    running_npu_set.add(int(chip_logic_id))
        if not running_npu_set:
            raise RuntimeError("Can not get running npu info, you can use BIND_CPU=0 to skip.")
        return sorted(running_npu_set)

    def parse_allowed_cpus(self) -> List[int]:
        if not os.path.exists(ALLOWED_CPUS_PATH):
            return []
        with open(ALLOWED_CPUS_PATH) as f:
            for line in f:
                if line.startswith("Cpus_allowed_list"):
                    return self.expand_cpu_list(line.split()[1])
        raise RuntimeError("Can not found specific 'Cpus_allowed_list' in the '/proc/self/status' file.")

    def parse_topo_affinity(self) -> Dict[int, List[int]]:
        affinity = {}
        affinity_message, _ = execute_command(["npu-smi", "info", "-t", "topo"])
        for line in affinity_message.splitlines():
            if line.startswith("NPU"):
                parts = line.split()
                index = int(parts[0][3:])
                last_part = parts[-1]
                if last_part != "Affinity":
                    affinity[index] = self.expand_cpu_list(last_part)
        return affinity


class CpuAlloc:
    def __init__(self):
        self.device_info: DeviceInfo = DeviceInfo()
        self.cpu_node: Dict[int, int] = {}
        self.numa_to_cpu_map: Dict[int, List[int]] = {}
        self.npu_cpu_pool: Dict[int, List[int]] = {}
        self.assign_main: Dict[int, List[int]] = {}
        self.assign_acl: Dict[int, List[int]] = {}
        self.assign_rel: Dict[int, Optional[int]] = {}

    @staticmethod
    def get_acl_main_threads() -> List[int]:
        thread_message, _ = execute_command(["ps", "-Te"])
        pids = []
        acl_threads_set = set()
        for line in thread_message.splitlines():
            if "acl_thread" in line:
                pid = line.split()[0]
                if pid not in acl_threads_set:
                    acl_threads_set.add(pid)
                    pids.append(int(pid))
        return pids

    @staticmethod
    def find_thread(pid: int, name: str) -> int:
        thread_message, _ = execute_command(["ps", "-Te"])
        for line in thread_message.splitlines():
            if name in line and str(pid) in line:
                return int(line.split()[1])
        return 0

    @staticmethod
    def bind(pid: int, cpus: List[int], bind_sub_thread: bool) -> None:
        if cpus:
            cpu_list = ",".join(map(str, cpus))
            if bind_sub_thread:
                bind_result, return_code = execute_command(["taskset", "-acp", cpu_list, str(pid)])
            else:
                bind_result, return_code = execute_command(["taskset", "-cp", cpu_list, str(pid)])
            if return_code != 0:
                raise RuntimeError(f"Failed to bind {pid} to CPU {cpu_list}.")

    def average_distribute(self, groups: Dict[str, List[int]]) -> Dict[int, List[int]]:
        result = {}
        for key, npu_list in groups.items():
            cpu_list = sorted(self.npu_cpu_pool[npu_list[0]])
            cpu_num_per_npu = len(cpu_list) // len(npu_list)
            for i, npu in enumerate(npu_list):
                start_index = i * cpu_num_per_npu
                end_index = (i + 1) * cpu_num_per_npu if i < len(npu_list) - 1 else len(npu_list)
                result[npu] = cpu_list[start_index:end_index]
        return result

    def extend_numa(self, cpu_list: List[int]) -> List[int]:
        if not cpu_list:
            return []
        nodes = {self.cpu_node[c] for c in cpu_list}
        if len(nodes) != 1:
            return cpu_list
        node = list(nodes)[0]
        next_node = (node + 1) % len(self.numa_to_cpu_map)
        extended = cpu_list[:]
        for cpu in self.numa_to_cpu_map[next_node]:
            if cpu in self.device_info.allowed_cpus:
                extended.append(cpu)
        return sorted(set(extended))

    def build_cpu_node_map(self) -> None:
        cpu_numa_map, _ = execute_command(["lscpu", "-p=CPU,NODE"])
        for line in cpu_numa_map.splitlines():
            if not line.startswith("#"):
                cpu, node = line.split(",")
                cpu = int(cpu)
                node = int(node)
                self.cpu_node[cpu] = node
                self.numa_to_cpu_map[node].append(cpu)
        if len(self.numa_to_cpu_map) == 0:
            raise RuntimeError("lscpu command output error, no NUMA node available. Please check!")

    def handle_no_affinity(self) -> None:
        num_running_npu = len(self.device_info.running_npu_list)
        num_numa_node = len(self.numa_to_cpu_map)
        if num_numa_node == 0 or num_running_npu == 0:
            return
        if num_running_npu % num_numa_node != 0:
            npu_num_per_node = num_running_npu // num_numa_node + 1
        else:
            npu_num_per_node = num_running_npu // num_numa_node
        index = 0
        for node in sorted(self.numa_to_cpu_map):
            # 该 NUMA 上可用的 CPU（受 allowed_cpus 约束）
            cpus = [c for c in self.numa_to_cpu_map[node] if c in self.device_info.allowed_cpus]
            if not cpus:
                continue
            # 这个 NUMA 上实际要分配的 NPU 数量
            npu_num_this_node = min(npu_num_per_node, num_running_npu - index)
            if npu_num_this_node <= 0:
                break
            # 把这个 NUMA 的 CPU 再均分给 npu_num_this_node 张 NPU
            total_cpu_num = len(cpus)
            base_cpu_num = total_cpu_num // npu_num_this_node
            extra_cpu_num = total_cpu_num % npu_num_this_node
            start_index = 0
            for i in range(npu_num_this_node):
                take_cpu_num = base_cpu_num + (1 if i < extra_cpu_num else 0)
                end_index = start_index + take_cpu_num
                select_cpus_list = cpus[start_index:end_index]
                if index < num_running_npu:
                    npu = self.device_info.running_npu_list[index]
                    self.npu_cpu_pool[npu] = select_cpus_list
                    index += 1
                start_index = end_index

    def build_cpu_pools(self) -> None:
        self.build_cpu_node_map()
        if not self.device_info.npu_affinity:
            self.handle_no_affinity()
            return
        for npu in self.device_info.running_npu_list:
            base_cpu_list = [cpu for cpu in self.device_info.npu_affinity.get(npu, [])
                             if cpu in self.device_info.allowed_cpus]
            extra_cpu_list = self.extend_numa(base_cpu_list)
            self.npu_cpu_pool[npu] = extra_cpu_list
        groups = defaultdict(list)
        for npu, cpus in self.npu_cpu_pool.items():
            groups[str(cpus)].append(npu)
        final = {}
        for key, npu_list in groups.items():
            if len(npu_list) == 1:
                final[npu_list[0]] = self.npu_cpu_pool[npu_list[0]]
            else:
                final.update(self.average_distribute({key: npu_list}))
        self.npu_cpu_pool = final

    def allocate(self) -> None:
        for npu, pool in self.npu_cpu_pool.items():
            if len(pool) >= 3:
                main = pool[:-2]
                acl = pool[-2]
                rel = pool[-1]
            else:
                raise RuntimeError("The number of CPUs is insufficient to bind to the NPUs. "
                                   "Each NPU requires at least 3 CPUs.")
            self.assign_main[npu] = main
            self.assign_acl[acl] = main
            self.assign_rel[rel] = acl

    def print_plan(self) -> None:
        logger.info("The CPU allocation plan is as follows:")
        for npu in sorted(self.device_info.running_npu_list):
            main = " ".join(map(str, self.assign_main[npu]))
            acl = " ".join(map(str, self.assign_acl[npu]))
            rel = str(self.assign_rel[npu]) if self.assign_rel[npu] else ""
            logger.info(f"NPU{npu}: main=[{main}]  acl=[{acl}]  release=[{rel}]")

    def bind_threads(self) -> None:
        threads = self.get_acl_main_threads()
        for npu, pid in zip(self.device_info.running_npu_list, threads):
            self.bind(pid, self.assign_main[npu], True)
            acl = self.find_thread(pid, "acl_thread")
            if acl:
                self.bind(acl, self.assign_acl[npu], False)
            rel = self.find_thread(pid, "relation_thread")
            if rel and self.assign_rel[npu]:
                self.bind(rel, [self.assign_rel[npu]], False)

    def run_all(self) -> None:
        self.build_cpu_pools()
        self.allocate()
        self.print_plan()
        self.bind_threads()


def bind_cpus(self) -> None:
    binder = CpuAlloc()
    binder.run_all()
