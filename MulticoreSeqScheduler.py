"""
DAG Task Scheduler Benchmark
============================

AUTHOR: Adel MOUSSAOUI et al.

PAPER: Performance Analysis of Dependency Aware Scheduling Algorithms
       for Parallel Nonlinear System Solving on Multicore Architectures

Description:
------------
Benchmarks different scheduling algorithms for parallel execution of tasks
with precedence constraints (Directed Acyclic Graph). Tasks represent
decomposed nonlinear systems solved using scipy.optimize.fsolve.

Features:
- Multiple scheduling algorithms (HEFT, ETF, CP-Priority, CPOP, etc.)
- Configurable DAG structures (balanced, serial, mixed)
- Performance metrics (makespan, utilization, speedup, efficiency)
- Statistical analysis across multiple trials

Main Components:
- PrecedenceGraph: Manages task dependencies and parallel execution
- Schedulers: 10 different task scheduling algorithms
- Benchmark: Compares performance across scenarios and core counts

Usage:
- Configure test scenarios in the main block
- Set run_mode to select which scenarios to test
- Adjust num_graphs for number of trials
- Results written to comparison_tables.txt and precedence_graphs.txt
"""

import multiprocessing as mp
import time
import os
import numpy as np
from scipy.optimize import fsolve
from collections import deque
import random
import networkx as nx
import math
import threading

# Set seeds globally for reproducibility
random.seed(42)
np.random.seed(42)

class PrecedenceGraph:
    def __init__(self, max_parallel, scheduler_mode='cp_priority'):
        self.processes = {}  # Dict of process info: dependencies, size, completion status
        self.max_parallel = max_parallel  # Max concurrent processes
        self.scheduled = set()  # Track scheduled processes
        self.scheduler_mode = scheduler_mode  # Scheduling algorithm
        self.b_levels = {}  # Bottom-level: longest path to exit
        self.t_levels = {}  # Top-level: longest path from entry
        self.cached_cp = None  # Cached critical path
        self.critical_path_calculated = False
        
    def add_process(self, name, system_size, dependencies=None):
        dependencies = dependencies or []
        # Validate dependencies exist
        for dep in dependencies:
            if dep not in self.processes and dep != name:
                raise ValueError(f"Dependency {dep} not found")
        self.processes[name] = {
            'name': name,
            'dependencies': dependencies,
            'system_size': system_size,
            'completed': False,
            'success': False,
            'duration': 0.0,
            'start_time': None,
            'end_time': None
        }

    def compute_b_levels(self):
        """Bottom levels: length of longest path from node to exit"""
        start_time = time.time()
        self.b_levels = {name: 0.0 for name in self.processes}
        
        def topological_sort():
            # Kahn's algorithm for topological ordering
            in_degree = {name: len(info['dependencies']) for name, info in self.processes.items()}
            queue = deque([name for name, degree in in_degree.items() if degree == 0])
            topo_order = []
            while queue:
                node = queue.popleft()
                topo_order.append(node)
                # Decrease in-degree of successors
                for next_node, info in self.processes.items():
                    if node in info['dependencies']:
                        in_degree[next_node] -= 1
                        if in_degree[next_node] == 0:
                            queue.append(next_node)
            if len(topo_order) != len(self.processes):
                raise ValueError("Graph contains a cycle")
            return topo_order

        topo_order = topological_sort()
        
        # Compute b-level in reverse topological order
        for node in reversed(topo_order):
            max_child_b_level = 0.0
            # Find max b-level of children
            for next_node, info in self.processes.items():
                if node in info['dependencies']:
                    max_child_b_level = max(max_child_b_level, self.b_levels[next_node])
            duration = self._estimate_duration(node)
            self.b_levels[node] = duration + max_child_b_level
        
        return time.time() - start_time

    def compute_t_levels(self):
        """Top levels: length of longest path from entry to node"""
        self.t_levels = {name: 0.0 for name in self.processes}
        
        def topological_sort():
            in_degree = {name: len(info['dependencies']) for name, info in self.processes.items()}
            queue = deque([name for name, degree in in_degree.items() if degree == 0])
            topo_order = []
            while queue:
                node = queue.popleft()
                topo_order.append(node)
                for next_node, info in self.processes.items():
                    if node in info['dependencies']:
                        in_degree[next_node] -= 1
                        if in_degree[next_node] == 0:
                            queue.append(next_node)
            return topo_order
        
        topo_order = topological_sort()
        
        # Compute t-level in forward topological order
        for node in topo_order:
            max_parent_t_level = 0.0
            # Find max t-level among parents
            for dep in self.processes[node]['dependencies']:
                parent_t = self.t_levels[dep] + self._estimate_duration(dep)
                max_parent_t_level = max(max_parent_t_level, parent_t)
            self.t_levels[node] = max_parent_t_level

    def _estimate_duration(self, node_name):
        """Consistent duration estimation using the given formula"""
        system_size = self.processes[node_name]['system_size']
        return 1e-9 * system_size ** 3  # Cubic complexity

    def get_critical_path_tasks(self):
        # Return cached result if available
        if self.critical_path_calculated and self.cached_cp:
            return self.cached_cp
            
        start_time = time.time()
        
        # Build weighted DAG
        G = nx.DiGraph()
        for name, info in self.processes.items():
            G.add_node(name, weight=self._estimate_duration(name))
            for dep in info['dependencies']:
                G.add_edge(dep, name)
        
        try:
            # Find longest path (critical path)
            longest_path = nx.dag_longest_path(G, weight='weight')
            path_duration = sum(G.nodes[node]['weight'] for node in longest_path)
            
            # Cache result
            self.cached_cp = (set(longest_path), path_duration, time.time() - start_time)
            self.critical_path_calculated = True
            return self.cached_cp
            
        except nx.NetworkXUnfeasible:
            raise ValueError("Graph contains a cycle - cannot compute critical path")

    def critical_path(self):
        critical_path_nodes, critical_path_duration, scheduling_time = self.get_critical_path_tasks()
        # Rebuild graph for path extraction
        G = nx.DiGraph()
        for name, info in self.processes.items():
            G.add_node(name, weight=self._estimate_duration(name))
            for dep in info['dependencies']:
                G.add_edge(dep, name)
        longest_path = nx.dag_longest_path(G, weight='weight')
        
        # Print critical path details
        print(f"\nCritical Path for {self.scheduler_mode}:")
        print(" -> ".join(longest_path))
        print(f"Total Duration: {critical_path_duration:.4f} seconds")
        print("Task Durations (Estimated):")
        for node in longest_path:
            duration = self._estimate_duration(node)
            print(f"  Node {node}: {duration:.4f}s (system_size={self.processes[node]['system_size']})")
        return longest_path, critical_path_duration, scheduling_time

    def verify_critical_path(self):
        # Verify critical path using NetworkX
        G = nx.DiGraph()
        for name, info in self.processes.items():
            G.add_node(name, weight=self._estimate_duration(name))
            for dep in info['dependencies']:
                G.add_edge(dep, name)
        longest_path = nx.dag_longest_path(G, weight='weight')
        path_duration = sum(G.nodes[node]['weight'] for node in longest_path)
        print(f"NetworkX CP: {' -> '.join(longest_path)}, Duration: {path_duration:.4f}s")
        return longest_path, path_duration

    def run(self, start_time):
        print(f"Starting equation solving processes (max parallel: {self.max_parallel}, mode: {self.scheduler_mode})...")
        
        scheduling_time = 0.0
        
        # Compute scheduling metrics based on mode
        if self.scheduler_mode in ['heft', 'etf', 'most_successors_first']:
            scheduling_time += self.compute_b_levels()
            if self.scheduler_mode == 'etf':
                self.compute_t_levels()
        
        critical_tasks = set()
        if self.scheduler_mode in ['cp_priority', 'cpop']:
            crit_start = time.time()
            critical_tasks, _, crit_time = self.get_critical_path_tasks()
            scheduling_time += crit_time
        
        total_processes = len(self.processes)
        completed_count = 0
        running_count = 0
        total_duration = 0.0
        
        # Check if workload is fully parallel (no dependencies)
        has_dependencies = any(len(self.processes[name]['dependencies']) > 0 for name in self.processes)
        if not has_dependencies:
            print(f"Fully parallel workload detected")
        
        pool_start = time.time()
        first_submit_time = None
        last_complete_time = None
        submission_count = 0
        
        # Event-driven scheduling with callbacks
        lock = threading.Lock()
        ready_queue = []  # Tasks ready to execute
        
        def get_ready_tasks():
            """Find all tasks whose dependencies are satisfied"""
            ready = []
            for name, proc_info in self.processes.items():
                if not proc_info['completed'] and name not in self.scheduled:
                    # Check if all dependencies completed
                    deps_completed = all(
                        self.processes[dep]['completed'] for dep in proc_info['dependencies']
                    )
                    if deps_completed or not proc_info['dependencies']:
                        ready.append(name)
            return ready
        
        def sort_tasks(tasks):
            """Sort tasks according to scheduler policy"""
            if self.scheduler_mode == 'cp_priority':
                # Prioritize critical path tasks, then by size
                if not has_dependencies:
                    tasks.sort(key=lambda x: (-self.processes[x]['system_size'], x))
                else:
                    tasks.sort(key=lambda x: (
                        0 if x in critical_tasks else 1,
                        -self.processes[x]['system_size'],
                        x
                    ))
            elif self.scheduler_mode == 'heft':
                # Highest level first (HEFT algorithm)
                tasks.sort(key=lambda x: (-self.b_levels[x], x))
            elif self.scheduler_mode == 'etf':
                # Earliest time first
                tasks.sort(key=lambda x: (self.t_levels[x], -self.b_levels[x], x))
            elif self.scheduler_mode == 'level_by_level':
                # By dependency level, then size
                tasks.sort(key=lambda x: (
                    len(self.processes[x]['dependencies']),
                    -self.processes[x]['system_size'],
                    x
                ))
            elif self.scheduler_mode == 'largest_first':
                # Largest tasks first
                tasks.sort(key=lambda x: (-self.processes[x]['system_size'], x))
            elif self.scheduler_mode == 'smallest_first':
                # Smallest tasks first
                tasks.sort(key=lambda x: (self.processes[x]['system_size'], x))
            elif self.scheduler_mode == 'most_successors_first':
                # Tasks with most dependents first
                successor_count = {}
                for node in self.processes:
                    successor_count[node] = sum(
                        1 for n, info in self.processes.items() 
                        if node in info['dependencies']
                    )
                tasks.sort(key=lambda x: (-successor_count.get(x, 0), -self.processes[x]['system_size'], x))
            elif self.scheduler_mode == 'cpop':
                # Critical path on processors
                if critical_tasks:
                    cp_tasks_ready = [t for t in tasks if t in critical_tasks]
                    non_cp_ready = [t for t in tasks if t not in critical_tasks]
                    non_cp_ready.sort(key=lambda x: (-self.processes[x]['system_size'], x))
                    tasks = cp_tasks_ready + non_cp_ready
                else:
                    tasks.sort(key=lambda x: (-self.processes[x]['system_size'], x))
            elif self.scheduler_mode == 'fifo':
                # First-in-first-out
                tasks.sort(key=lambda x: x)
            elif self.scheduler_mode == 'random':
                # Random order
                random.shuffle(tasks)
            else:
                tasks.sort(key=lambda x: x)
            return tasks
        
        def on_task_complete(result):
            """Callback fired when task completes"""
            nonlocal completed_count, running_count, total_duration, last_complete_time
            
            name, duration, success = result
            
            with lock:
                # Update task info
                self.processes[name]['completed'] = True
                self.processes[name]['success'] = success
                self.processes[name]['duration'] = duration
                self.processes[name]['end_time'] = time.time()
                last_complete_time = time.time()
                total_duration += duration
                completed_count += 1
                running_count -= 1
                
                print(f"COMPLETE: {name}, duration: {duration:.4f}s, running: {running_count}")
                
                # Add newly ready tasks to queue
                newly_ready = get_ready_tasks()
                if newly_ready:
                    ready_queue.extend(newly_ready)
        
        # Create process pool
        with mp.Pool(processes=self.max_parallel) as pool:
            pool_creation_time = time.time() - pool_start
            print(f"Pool creation took: {pool_creation_time:.4f}s")
            
            # Submit initial batch of tasks
            ready_tasks = sort_tasks(get_ready_tasks())
            first_submit_time = time.time()
            
            for task in ready_tasks[:self.max_parallel]:
                self.processes[task]['start_time'] = time.time()
                self.scheduled.add(task)
                running_count += 1
                submission_count += 1
                
                # Submit async task with callback
                pool.apply_async(
                    solve_system_parallel_wrapper,
                    [(task, self.processes[task]['system_size'])],
                    callback=on_task_complete
                )
                print(f"SUBMIT: {task}, running: {running_count}/{self.max_parallel}")
            
            # Event loop: submit tasks as slots become available
            while completed_count < total_processes:
                with lock:
                    if ready_queue and running_count < self.max_parallel:
                        ready_queue = sort_tasks(ready_queue)
                        
                        # Fill available slots
                        while ready_queue and running_count < self.max_parallel:
                            task = ready_queue.pop(0)
                            
                            # Skip if already scheduled
                            if task in self.scheduled:
                                continue
                            
                            self.processes[task]['start_time'] = time.time()
                            self.scheduled.add(task)
                            running_count += 1
                            submission_count += 1
                            
                            pool.apply_async(
                                solve_system_parallel_wrapper,
                                [(task, self.processes[task]['system_size'])],
                                callback=on_task_complete
                            )
                            print(f"SUBMIT: {task}, running: {running_count}/{self.max_parallel}")
                
                time.sleep(0.001)  # Brief sleep to avoid busy waiting
        
        makespan = time.time() - start_time
        
        # Print timing breakdown
        if first_submit_time and last_complete_time:
            time_to_first_submit = first_submit_time - start_time
            actual_execution_time = last_complete_time - first_submit_time
            print(f"\nTIMING BREAKDOWN:")
            print(f"  Pool creation: {pool_creation_time:.4f}s ({pool_creation_time/makespan*100:.1f}%)")
            print(f"  Time to first submit: {time_to_first_submit:.4f}s ({time_to_first_submit/makespan*100:.1f}%)")
            print(f"  Actual execution: {actual_execution_time:.4f}s ({actual_execution_time/makespan*100:.1f}%)")
            print(f"  Total makespan: {makespan:.4f}s")
        
        # Print critical path info
        critical_path_nodes, critical_path_duration, _ = self.get_critical_path_tasks()
        G = nx.DiGraph()
        for name, info in self.processes.items():
            G.add_node(name, weight=self._estimate_duration(name))
            for dep in info['dependencies']:
                G.add_edge(dep, name)
        longest_path = nx.dag_longest_path(G, weight='weight')
        
        print(f"\nCritical Path for {self.scheduler_mode}:")
        print(" -> ".join(longest_path))
        print(f"Total Duration: {critical_path_duration:.4f} seconds")
        
        self.verify_critical_path()
        
        # Calculate performance metrics
        utilization = total_duration / (self.max_parallel * makespan) if makespan > 0 else 0.0
        overhead = makespan - critical_path_duration
        
        # Parallel efficiency for fully parallel workloads
        if not has_dependencies and total_processes > 1:
            theoretical_makespan = critical_path_duration / min(total_processes, self.max_parallel)
            efficiency = (theoretical_makespan / makespan) * 100 if makespan > 0 else 0
            print(f"Parallel Efficiency: {efficiency:.1f}%")
        
        print(f"\nMetrics for {self.scheduler_mode}:")
        print(f"Makespan: {makespan:.4f} seconds")
        print(f"Critical Path Duration: {critical_path_duration:.4f} seconds")
        print(f"CPU Utilization: {utilization:.4f} ({utilization*100:.1f}%)")
        print(f"Scheduling Overhead: {scheduling_time:.4f} seconds")
        print(f"Total Overhead (Makespan - CP Duration): {overhead:.4f} seconds")
        print(f"Total Task Duration Sum: {total_duration:.4f} seconds")
        
        return makespan, critical_path_duration, utilization, scheduling_time, overhead
    
def equations(x):
    # Simple quadratic equation
    return x**2 - 4

def solve_system_parallel_wrapper(args):
    """Wrapper for multiprocessing"""
    name, system_size = args
    pid = os.getpid()
    print(f"START: {name} on PID {pid}", flush=True)
    
    x0 = np.ones(system_size) * 1.5  # Initial guess
    start = time.time()
    try:
        # Solve nonlinear system
        with np.errstate(all='raise'):
            solution, info, ier, msg = fsolve(equations, x0, full_output=True)
        end = time.time()
        elapsed = round(end - start, 8)
        
        if ier == 1:  # Success
            print(f"END: {name} solved in {elapsed:.4f}s", flush=True)
            return name, elapsed, True
        else:
            print(f"FAIL: {name} did not converge", flush=True)
            return name, 0.0, False
    except Exception as e:
        print(f"ERROR: {name} failed: {e}", flush=True)
        return name, 0.0, False
    
def generate_random_dag(S, edge_prob=0.3, max_parents=3, remove_transitive=True, structure='balanced'):
    """Generate random DAG with controlled structure"""
    G = nx.DiGraph()
    nodes = [str(i) for i in range(S)]
    G.add_nodes_from(nodes)
    
    if structure == 'balanced':
        # Create balanced levels of parallelism
        num_levels = max(3, int(S ** 0.5))
        nodes_per_level = S // num_levels
        
        # Assign nodes to levels
        level_assignment = {}
        for i, node in enumerate(nodes):
            level = min(i // max(1, nodes_per_level), num_levels - 1)
            level_assignment[node] = level
        
        # Connect nodes to next level
        for node_i in nodes:
            level_i = level_assignment[node_i]
            next_level_nodes = [n for n in nodes if level_assignment[n] == level_i + 1]
            
            if next_level_nodes:
                num_connections = min(random.randint(1, 2), len(next_level_nodes))
                targets = random.sample(next_level_nodes, num_connections)
                for target in targets:
                    if len(list(G.predecessors(target))) < max_parents:
                        G.add_edge(node_i, target)
    
    elif structure == 'serial':
        # Create more serial dependencies
        for i in range(S):
            for j in range(i + 1, S):
                if random.random() < edge_prob:
                    current_parents = len(list(G.predecessors(nodes[j])))
                    if current_parents < max_parents:
                        G.add_edge(nodes[i], nodes[j])
    
    else:  # mixed
        # Mix of parallel and serial paths
        num_levels = max(4, int(S ** 0.6))
        nodes_per_level = S // num_levels
        
        level_assignment = {}
        for i, node in enumerate(nodes):
            level = min(i // max(1, nodes_per_level), num_levels - 1)
            level_assignment[node] = level
        
        # Allow connections across multiple levels
        for node_i in nodes:
            level_i = level_assignment[node_i]
            for level_offset in [1, 2]:
                target_level = level_i + level_offset
                if target_level >= num_levels:
                    continue
                    
                candidates = [n for n in nodes if level_assignment[n] == target_level]
                if candidates and random.random() < (0.4 / level_offset):
                    num_connections = random.randint(1, 2)
                    for target in random.sample(candidates, min(num_connections, len(candidates))):
                        if len(list(G.predecessors(target))) < max_parents:
                            G.add_edge(node_i, target)
    
    # Connect isolated nodes
    for node in nodes:
        if G.in_degree(node) == 0 and G.out_degree(node) == 0:
            later_nodes = [n for n in nodes if int(n) > int(node)]
            if later_nodes:
                target = random.choice(later_nodes[:min(5, len(later_nodes))])
                if len(list(G.predecessors(target))) < max_parents:
                    G.add_edge(node, target)
    
    # Remove redundant transitive edges
    if remove_transitive:
        transitive_reduction = nx.transitive_reduction(G)
        G_reduced = nx.DiGraph()
        G_reduced.add_nodes_from(G.nodes())
        G_reduced.add_edges_from(transitive_reduction.edges())
        return G_reduced
    
    return G

def generate_random_dag_precedence_config(S, N, Min_size, Max_size, remove_transitive=True, structure='balanced'):
    """Generate DAG configuration - OPTION A: Allocate sizes randomly, THEN find CP"""
    dag = generate_random_dag(S, edge_prob=0.3, max_parents=3, remove_transitive=remove_transitive, structure=structure)
    config = {}
    topo_order = list(nx.topological_sort(dag))
    
    # Assign nodes to dependency levels
    level_assignment = {}
    for node in topo_order:
        if not list(dag.predecessors(node)):
            level_assignment[node] = 0  # Root nodes
        else:
            level_assignment[node] = max(level_assignment[pred] for pred in dag.predecessors(node)) + 1
    
    # Calculate level statistics
    max_level = max(level_assignment.values()) if level_assignment else 0
    level_widths = [sum(1 for n in topo_order if level_assignment.get(n) == lvl) for lvl in range(max_level + 1)]
    max_width = max(level_widths) if level_widths else 1
    
    # Random size allocation strategy
    node_sizes = {}
    remaining_N = N
    all_nodes = topo_order.copy()
    allocation_order = all_nodes.copy()
    random.shuffle(allocation_order)
    
    # Give each node minimum size
    for node in all_nodes:
        node_sizes[node] = Min_size
        remaining_N -= Min_size
    
    # Distribute remaining size randomly
    while remaining_N > 0 and allocation_order:
        node = allocation_order[random.randint(0, len(allocation_order) - 1)]
        max_addition = min(Max_size - node_sizes[node], remaining_N)
        
        if max_addition > 0:
            # Random allocation between 20-80% of available
            addition = random.randint(
                max(1, int(max_addition * 0.2)),
                max(1, int(max_addition * 0.8))
            )
            node_sizes[node] += addition
            remaining_N -= addition
        
        # Remove nodes at max size
        if node_sizes[node] >= Max_size:
            allocation_order = [n for n in allocation_order if n != node]
        
        if not allocation_order:
            break
    
    # Distribute any leftover
    while remaining_N > 0:
        for node in all_nodes:
            if remaining_N <= 0:
                break
            if node_sizes[node] < Max_size:
                addition = min(remaining_N, Max_size - node_sizes[node])
                node_sizes[node] += addition
                remaining_N -= addition
        
        if all(node_sizes[n] >= Max_size for n in all_nodes):
            break
    
    # Build config with dependencies
    for node in topo_order:
        dependencies = list(dag.predecessors(node))
        config[node] = {'system_size': node_sizes[node], 'dependencies': dependencies}
    
    # Find actual critical path with durations
    G_with_durations = nx.DiGraph()
    for node in topo_order:
        duration = 1e-9 * node_sizes[node] ** 3
        G_with_durations.add_node(node, weight=duration)
        for dep in dag.predecessors(node):
            G_with_durations.add_edge(dep, node)
    
    try:
        actual_cp = nx.dag_longest_path(G_with_durations, weight='weight')
        cp_nodes = set(actual_cp)
        cp_duration = sum(G_with_durations.nodes[node]['weight'] for node in actual_cp)
    except:
        cp_nodes = set()
        cp_duration = 0
        actual_cp = []
    
    # Calculate statistics
    total_size = sum(data['system_size'] for data in config.values())
    cp_size = sum(node_sizes[n] for n in cp_nodes) if cp_nodes else 0
    cp_pct = (cp_size / total_size * 100) if total_size > 0 else 0
    
    if cp_nodes:
        cp_sizes = [node_sizes[n] for n in cp_nodes]
        non_cp_sizes = [node_sizes[n] for n in all_nodes if n not in cp_nodes]
        cp_avg = sum(cp_sizes) / len(cp_sizes) if cp_sizes else 0
        non_cp_avg = sum(non_cp_sizes) / len(non_cp_sizes) if non_cp_sizes else 0
        size_ratio = cp_avg / non_cp_avg if non_cp_avg > 0 else 0
    else:
        size_ratio = 0
    
    print(f"Generated DAG ({structure}): {len(dag.edges())} edges, max width: {max_width}")
    print(f"  Total size: {total_size}/{N}")
    print(f"  Critical Path (discovered): {len(cp_nodes)} nodes, {cp_pct:.1f}% of work, size_ratio: {size_ratio:.2f}x")
    print(f"  CP duration: {cp_duration:.4f}s")
    print(f"  Actual CP: {' -> '.join(actual_cp)}")
    
    return config

def validate_critical_path(config, critical_path_nodes):
    """Validate that the critical path respects dependencies"""
    critical_path_list = list(critical_path_nodes)
    # Check each node's dependencies appear earlier in path
    for i in range(len(critical_path_list)):
        current_node = critical_path_list[i]
        for dep in config[current_node]['dependencies']:
            if dep in critical_path_list and critical_path_list.index(dep) >= i:
                return False, f"Node {current_node} depends on {dep} but appears after"
    return True, "Valid"

if __name__ == "__main__":
    # ========== TEST SCENARIOS ==========
    
    # Define multiple test scenarios
    test_scenarios = {
        'small_balanced': {
            'S': 10,
            'N': 4000,
            'structure': 'balanced',
            'edge_prob': 0.3,
            'description': 'Small graph with balanced parallelism'
        },
        'medium_balanced': {
            'S': 20,
            'N': 20000,
            'structure': 'balanced',
            'edge_prob': 0.3,
            'description': 'Medium graph with good parallelism opportunities'
        },
        'large_balanced': {
            'S': 40,
            'N': 40000,
            'structure': 'balanced',
            'edge_prob': 0.3,
            'description': 'Large graph with high parallelism'
        },
        'medium_serial': {
            'S': 20,
            'N': 20000,
            'structure': 'serial',
            'edge_prob': 0.4,
            'description': 'Medium graph with deep dependencies (serial)'
        },
        'medium_mixed': {
            'S': 20,
            'N': 20000,
            'structure': 'mixed',
            'edge_prob': 0.35,
            'description': 'Medium graph with mixed parallel/serial structure'
        },
        'sparse_graph': {
            'S': 30,
            'N': 24000,
            'structure': 'balanced',
            'edge_prob': 0.15,
            'description': 'Sparse graph (low connectivity, high parallelism)'
        },
        'dense_graph': {
            'S': 20,
            'N': 20000,
            'structure': 'balanced',
            'edge_prob': 0.5,
            'description': 'Dense graph (high connectivity, limited parallelism)'
        },
        'wide_shallow': {
            'S': 50,
            'N': 40000,
            'structure': 'balanced',
            'edge_prob': 0.2,
            'description': 'Wide shallow graph (many parallel paths)'
        },
        'narrow_deep': {
            'S': 15,
            'N': 20000,
            'structure': 'serial',
            'edge_prob': 0.5,
            'description': 'Narrow deep graph (long critical path)'
        },
        'quick_test': {
            'S': 10,
            'N': 4000,
            'structure': 'balanced',
            'edge_prob': 0.3,
            'description': 'Quick test scenario (fast execution)'
        }
    }
    
    # SELECT WHICH SCENARIOS TO RUN
    run_mode = 'structure_comparison'  # Change this to select different test sets
    
    # Select scenarios based on run mode
    if run_mode == 'all':
        scenarios_to_run = list(test_scenarios.keys())
    elif run_mode == 'quick':
        scenarios_to_run = ['quick_test']
    elif run_mode == 'balanced_only':
        scenarios_to_run = ['small_balanced', 'medium_balanced', 'large_balanced']
    elif run_mode == 'structure_comparison':
        scenarios_to_run = ['medium_balanced', 'medium_serial', 'medium_mixed']
    elif run_mode == 'density_comparison':
        scenarios_to_run = ['sparse_graph', 'medium_balanced', 'dense_graph']
    elif run_mode == 'shape_comparison':
        scenarios_to_run = ['wide_shallow', 'medium_balanced', 'narrow_deep']
    elif isinstance(run_mode, list):
        scenarios_to_run = run_mode
    else:
        scenarios_to_run = ['medium_balanced']  # Default
    
    # Scheduler configuration
    scheduler_modes = ['cp_priority', 'heft', 'etf', 'level_by_level', 
                       'largest_first', 'smallest_first', 'most_successors_first', 
                       'cpop', 'fifo', 'random']
    
    # Test configuration flags
    test_no_decomposition = False  # Baseline: no task decomposition
    test_sequential = True         # Sequential: 1 core
    test_parallel_cores = [2, 4, 8]  # Parallel: 2, 4, 8 cores
    
    num_graphs = 10  # Number of trials for statistical significance
    num_graphs = 3  # Use for faster testing
    
    # Output files
    output_file = os.path.join(os.getcwd(), "comparison_tables.txt")
    graph_file = os.path.join(os.getcwd(), "precedence_graphs.txt")
    
    # Print benchmark configuration
    print("="*70)
    print("DAG SCHEDULER BENCHMARK")
    print("="*70)
    print(f"Running {len(scenarios_to_run)} scenario(s)")
    print(f"Tests:")
    print(f"  1. No decomposition: {test_no_decomposition}")
    print(f"  2. Sequential (1 core): {test_sequential}")
    print(f"  3. Parallel cores: {test_parallel_cores}")
    print(f"Schedulers: {len(scheduler_modes)}")
    print(f"Trials per test: {num_graphs}")
    print("="*70)
    
    # Initialize output files
    with open(output_file, 'w') as f:
        f.write("# DAG Scheduler Comparison Results\n\n")
        f.write(f"Run Mode: {run_mode}\n")
        f.write(f"Tests: No decomposition, Sequential (1 core), Parallel {test_parallel_cores}\n")
        f.write(f"Schedulers: {', '.join(scheduler_modes)}\n")
        f.write(f"Trials per test: {num_graphs}\n\n")
    
    with open(graph_file, 'w') as f:
        f.write(f"# Precedence Graphs\n\n")
    
    # Main benchmark loop: iterate through scenarios
    for scenario_name in scenarios_to_run:
        scenario = test_scenarios[scenario_name]
        
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario_name}")
        print(f"Description: {scenario['description']}")
        print(f"{'='*70}")
        
        # Extract scenario parameters
        S = scenario['S']
        N = scenario['N']
        structure = scenario['structure']
        edge_prob = scenario.get('edge_prob', 0.3)
        
        # Calculate size bounds
        Min_size = math.floor(N / (S * 2.5))
        Max_size = math.floor(4 * N / S)
        
        print(f"S={S} nodes, N={N} total size")
        print(f"Min_size={Min_size}, Max_size={Max_size}")
        print(f"Structure={structure}, Edge probability={edge_prob}")
        print(f"Expected task durations: {1e-9 * Min_size**3:.2f}s to {1e-9 * Max_size**3:.2f}s")
        print(f"Expected total work: ~{1e-9 * N**3 / S:.1f}s")
        
        # Generate test graphs
        configs = []
        for _ in range(num_graphs):
            config = generate_random_dag_precedence_config(S, N, Min_size, Max_size, structure=structure)
            configs.append(config)
        
        # Write graph info to file
        with open(graph_file, 'a') as f:
            f.write(f"\n## Scenario: {scenario_name}\n")
            f.write(f"Description: {scenario['description']}\n")
            f.write(f"Parameters: S={S}, N={N}, structure={structure}\n\n")
            
            for i, config in enumerate(configs, 1):
                f.write(f"### Graph {i}\n")
                total_size = sum(data['system_size'] for data in config.values())
                f.write(f"Total System Size: {total_size}\n")
                
                # Create temp graph to get critical path
                temp_graph = PrecedenceGraph(max_parallel=1, scheduler_mode='cp_priority')
                for name, data in config.items():
                    temp_graph.add_process(name, system_size=data['system_size'], dependencies=data['dependencies'])
                
                critical_path_nodes, critical_path_duration, _ = temp_graph.get_critical_path_tasks()
                
                # Build NetworkX graph for verification
                G = nx.DiGraph()
                for name, data in config.items():
                    G.add_node(name, weight=1e-9 * data['system_size'] ** 3)
                    for dep in data['dependencies']:
                        G.add_edge(dep, name)
                actual_critical_path = nx.dag_longest_path(G, weight='weight')
                
                is_valid, validation_msg = validate_critical_path(config, actual_critical_path)
                
                # Write critical path info
                f.write(f"Critical Path: {' -> '.join(actual_critical_path)}\n")
                f.write(f"Critical Path Duration: {critical_path_duration:.4f}s\n")
                f.write(f"Valid: {validation_msg}\n")
                
                # Calculate size statistics
                cp_sizes = [config[node]['system_size'] for node in actual_critical_path]
                non_cp_sizes = [config[node]['system_size'] for node in config if node not in critical_path_nodes]
                
                f.write(f"\nSize Statistics:\n")
                f.write(f"  CP nodes: avg={np.mean(cp_sizes):.1f}, min={min(cp_sizes)}, max={max(cp_sizes)}\n")
                if non_cp_sizes:
                    f.write(f"  Non-CP nodes: avg={np.mean(non_cp_sizes):.1f}, min={min(non_cp_sizes)}, max={max(non_cp_sizes)}\n")
                    f.write(f"  CP/Non-CP size ratio: {np.mean(cp_sizes)/np.mean(non_cp_sizes):.2f}x\n")
                
                # Write graph statistics
                f.write(f"\nGraph Statistics:\n")
                f.write(f"  Nodes: {len(config)}\n")
                f.write(f"  Edges: {sum(len(data['dependencies']) for data in config.values())}\n")
                f.write(f"  Avg dependencies per node: {sum(len(data['dependencies']) for data in config.values())/len(config):.2f}\n")
                f.write("\n")
        
        # Write scenario header to results file
        with open(output_file, 'a') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"# Scenario: {scenario_name}\n")
            f.write(f"Description: {scenario['description']}\n")
            f.write(f"Parameters: S={S}, N={N}, structure={structure}\n")
            f.write(f"{'='*70}\n\n")
        
        # ========== TEST 1: NO DECOMPOSITION (BASELINE) ==========
        if test_no_decomposition:
            print(f"\n{'='*50}")
            print(f"TEST 1: NO DECOMPOSITION (Monolithic System)")
            print('='*50)
            
            baseline_times = []
            # Solve entire system as single task
            for trial in range(num_graphs):
                print(f"Trial {trial+1}/{num_graphs}")
                start = time.time()
                x0 = np.ones(N) * 1.5
                try:
                    with np.errstate(all='raise'):
                        solution, info, ier, msg = fsolve(equations, x0, full_output=True)
                    elapsed = time.time() - start
                    if ier == 1:
                        baseline_times.append(elapsed)
                        print(f"  Solved in {elapsed:.2f}s")
                    else:
                        print(f"  Failed to converge")
                except Exception as e:
                    print(f"  Error: {e}")
            
            # Calculate baseline statistics
            if baseline_times:
                baseline_mean = np.mean(baseline_times)
                baseline_std = np.std(baseline_times, ddof=1) if len(baseline_times) > 1 else 0.0
                baseline_min = min(baseline_times)
                baseline_max = max(baseline_times)
                
                print(f"\nBaseline Results (No Decomposition):")
                print(f"  Mean: {baseline_mean:.2f}s")
                print(f"  Std Dev: {baseline_std:.2f}s")
                print(f"  Range: {baseline_min:.2f}s - {baseline_max:.2f}s")
                
                # Write results to file
                with open(output_file, 'a') as f:
                    f.write(f"## TEST 1: No Decomposition (Baseline)\n")
                    f.write(f"Solving entire system (N={N}) as monolithic task\n\n")
                    f.write(f"| Metric | Value |\n")
                    f.write(f"|--------|-------|\n")
                    f.write(f"| Mean Time | {baseline_mean:.2f}s |\n")
                    f.write(f"| Std Dev | {baseline_std:.2f}s |\n")
                    f.write(f"| Min-Max | {baseline_min:.2f}s - {baseline_max:.2f}s |\n")
                    f.write(f"| Trials | {len(baseline_times)}/{num_graphs} successful |\n\n")
            else:
                baseline_mean = None
                print("No successful baseline runs")
        else:
            baseline_mean = None
        
        # ========== TEST 2: SEQUENTIAL WITH DECOMPOSITION ==========
        if test_sequential:
            print(f"\n{'='*50}")
            print(f"TEST 2: SEQUENTIAL (1 core, dependency order)")
            print('='*50)
            
            # Sequential uses FIFO (order doesn't matter with 1 core)
            sequential_results = {
                'makespan_list': [],
                'cp_duration': 0.0
            }
            
            for i, config in enumerate(configs):
                print(f"Trial {i+1}/{num_graphs}")
                g = PrecedenceGraph(max_parallel=1, scheduler_mode='fifo')
                for name, data in config.items():
                    g.add_process(name, system_size=data['system_size'], dependencies=data['dependencies'])
                start_time = time.time()
                makespan, cp_dur, util, sched_time, overhead = g.run(start_time)
                sequential_results['makespan_list'].append(makespan)
                sequential_results['cp_duration'] = cp_dur
            
            # Calculate statistics
            seq_mean = np.mean(sequential_results['makespan_list'])
            seq_std = np.std(sequential_results['makespan_list'], ddof=1) if num_graphs > 1 else 0.0
            seq_min = min(sequential_results['makespan_list'])
            seq_max = max(sequential_results['makespan_list'])
            
            print(f"\nSequential Results (1 core):")
            print(f"  Mean: {seq_mean:.2f}s")
            print(f"  Std Dev: {seq_std:.2f}s")
            print(f"  Range: {seq_min:.2f}s - {seq_max:.2f}s")
            print(f"  Critical Path: {sequential_results['cp_duration']:.2f}s")
            
            # Calculate speedup vs baseline
            if baseline_mean:
                speedup = baseline_mean / seq_mean
                print(f"  Speedup vs No Decomposition: {speedup:.2f}x")
            
            # Write to file
            with open(output_file, 'a') as f:
                f.write(f"## TEST 2: Sequential with Decomposition (1 core)\n")
                f.write(f"Decomposed into {S} tasks, executed sequentially\n\n")
                f.write(f"| Metric | Value |\n")
                f.write(f"|--------|-------|\n")
                f.write(f"| Mean Time | {seq_mean:.2f}s |\n")
                f.write(f"| Std Dev | {seq_std:.2f}s |\n")
                f.write(f"| Min-Max | {seq_min:.2f}s - {seq_max:.2f}s |\n")
                f.write(f"| Critical Path | {sequential_results['cp_duration']:.2f}s |\n")
                if baseline_mean:
                    speedup = baseline_mean / seq_mean
                    f.write(f"| **Speedup vs Baseline** | **{speedup:.2f}x** |\n")
                f.write(f"\n")
        else:
            seq_mean = None
        
        # ========== TEST 3: PARALLEL WITH DECOMPOSITION ==========
        for core_count in test_parallel_cores:
            print(f"\n{'='*50}")
            print(f"TEST 3: PARALLEL with {core_count} cores")
            print('='*50)
            
            # Initialize results storage
            averages = {
                mode: {
                    'makespan': 0.0, 'cp_duration': 0.0, 'utilization': 0.0,
                    'scheduling_overhead': 0.0, 'total_overhead': 0.0, 
                    'makespan_list': [],
                    'utilization_list': [],
                    'overhead_list': []
                } for mode in scheduler_modes
            }

            # Test each scheduler
            for mode in scheduler_modes:
                print(f"\n--- {mode} scheduler ---")
                for i, config in enumerate(configs):
                    print(f"Graph {i+1}/{num_graphs}")
                    g = PrecedenceGraph(max_parallel=core_count, scheduler_mode=mode)
                    for name, data in config.items():
                        g.add_process(name, system_size=data['system_size'], dependencies=data['dependencies'])
                    start_time = time.time()
                    makespan, cp_dur, util, sched_time, overhead = g.run(start_time)
                    # Accumulate metrics
                    averages[mode]['makespan'] += makespan
                    averages[mode]['cp_duration'] += cp_dur
                    averages[mode]['utilization'] += util
                    averages[mode]['scheduling_overhead'] += sched_time
                    averages[mode]['total_overhead'] += overhead
                    averages[mode]['makespan_list'].append(makespan)
                    averages[mode]['utilization_list'].append(util)
                    averages[mode]['overhead_list'].append(overhead)
            
                # Calculate means
                averages[mode]['makespan'] /= num_graphs
                averages[mode]['cp_duration'] /= num_graphs
                averages[mode]['utilization'] /= num_graphs
                averages[mode]['scheduling_overhead'] /= num_graphs
                averages[mode]['total_overhead'] /= num_graphs
                
                # Calculate standard deviations
                averages[mode]['makespan_std'] = np.std(averages[mode]['makespan_list'], ddof=1) if num_graphs > 1 else 0.0
                averages[mode]['makespan_min'] = min(averages[mode]['makespan_list'])
                averages[mode]['makespan_max'] = max(averages[mode]['makespan_list'])
                
                averages[mode]['util_std'] = np.std(averages[mode]['utilization_list'], ddof=1) if num_graphs > 1 else 0.0
                averages[mode]['overhead_std'] = np.std(averages[mode]['overhead_list'], ddof=1) if num_graphs > 1 else 0.0
                
                # 95% Confidence interval
                if num_graphs >= 10:
                    std_error = averages[mode]['makespan_std'] / np.sqrt(num_graphs)
                    averages[mode]['ci_95'] = 1.96 * std_error
                else:
                    averages[mode]['ci_95'] = 0.0

            # Create results table
            table_lines = [
                f"\n## TEST 3: Parallel with {core_count} cores ({num_graphs} trials)",
                "| Scheduler          | Mean (s) | Std Dev | Min-Max | Util (%) | Speedup vs Seq | Speedup vs Baseline |",
                "|--------------------|---------:|--------:|---------|----------|----------------|---------------------|"
            ]
            for mode in scheduler_modes:
                util_pct = averages[mode]['utilization'] * 100
                util_std_pct = averages[mode]['util_std'] * 100
                
                minmax_str = f"{averages[mode]['makespan_min']:.1f}-{averages[mode]['makespan_max']:.1f}"
                
                # Calculate speedups
                speedup_vs_seq = seq_mean / averages[mode]['makespan'] if seq_mean else 0
                speedup_vs_baseline = baseline_mean / averages[mode]['makespan'] if baseline_mean else 0
                
                speedup_seq_str = f"{speedup_vs_seq:.2f}x" if seq_mean else "N/A"
                speedup_base_str = f"{speedup_vs_baseline:.2f}x" if baseline_mean else "N/A"
                
                table_lines.append(
                    f"| {mode:<18} | {averages[mode]['makespan']:>8.2f} | "
                    f"{averages[mode]['makespan_std']:>7.2f} | {minmax_str:>7} | "
                    f"{util_pct:>5.1f}±{util_std_pct:.1f} | {speedup_seq_str:>14} | {speedup_base_str:>19} |"
                )
            
            # Add summary notes
            table_lines.append("")
            table_lines.append(f"**Notes:**")
            if baseline_mean:
                table_lines.append(f"- Baseline (no decomposition): {baseline_mean:.2f}s")
            if seq_mean:
                table_lines.append(f"- Sequential (1 core): {seq_mean:.2f}s")
            table_lines.append(f"- Each result is mean of {num_graphs} trials")
            
            # Find best scheduler
            best_scheduler = min(scheduler_modes, key=lambda m: averages[m]['makespan'])
            best_time = averages[best_scheduler]['makespan']
            best_speedup_vs_seq = seq_mean / best_time if seq_mean else 0
            best_speedup_vs_base = baseline_mean / best_time if baseline_mean else 0
            
            table_lines.append(f"\n**Best Scheduler:** {best_scheduler} ({best_time:.2f}s)")
            if seq_mean:
                table_lines.append(f"- Speedup vs Sequential: {best_speedup_vs_seq:.2f}x")
            if baseline_mean:
                table_lines.append(f"- Speedup vs Baseline: {best_speedup_vs_base:.2f}x")
                efficiency = (best_speedup_vs_base / core_count) * 100
                table_lines.append(f"- Parallel Efficiency: {efficiency:.1f}%")
            
            # Scheduler rankings
            table_lines.append(f"\n**Scheduler Rankings (by makespan):**")
            for rank, mode in enumerate(sorted(scheduler_modes, key=lambda m: averages[m]['makespan']), 1):
                slowdown = ((averages[mode]['makespan'] / best_time) - 1) * 100
                if mode == best_scheduler:
                    table_lines.append(f"{rank}. {mode}: BEST")
                else:
                    table_lines.append(f"{rank}. {mode}: +{slowdown:.1f}% slower")
            
            # Print and write to file
            print("\n" + "\n".join(table_lines[1:]))
            
            with open(output_file, 'a') as f:
                f.write("\n".join(table_lines) + "\n\n")

    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ Benchmark complete!")
    print(f"Results written to:")
    print(f"  - {output_file}")
    print(f"  - {graph_file}")
    print(f"{'='*70}")
