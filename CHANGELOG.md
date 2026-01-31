# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## \[1.0.0] - 2026-01-31

### Added

* Initial release of DAG Task Scheduler Benchmark
* 10 scheduling algorithms: HEFT, ETF, CP-Priority, CPOP, Level-by-Level, Largest-First, Smallest-First, Most-Successors-First, FIFO, Random
* Multiple DAG structure types: balanced, serial, mixed, sparse, dense, wide-shallow, narrow-deep
* Comprehensive performance metrics: makespan, CPU utilization, speedup, parallel efficiency
* Statistical analysis across multiple trials
* Three test modes: no decomposition baseline, sequential execution, parallel execution
* Configurable test scenarios with 10 predefined scenarios
* Automated graph generation with NetworkX
* Output to text files with formatted tables
* Critical path analysis and validation
* Support for multi-core parallel execution (2, 4, 8 cores)

### Features

* Event-driven task scheduling with callbacks
* Topological sorting for dependency resolution
* Bottom-level and top-level computation for HEFT/ETF algorithms
* Critical path caching for performance
* Timing breakdown analysis
* Graph statistics and validation

### Documentation

* Comprehensive README with usage examples
* Contributing guidelines
* MIT License
* Citation file for academic use
* Requirements specification
