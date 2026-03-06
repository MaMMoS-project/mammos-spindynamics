# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This project uses [towncrier](https://towncrier.readthedocs.io/) and the changes for the upcoming release can be found in [changes](changes).

<!-- towncrier release notes start -->

## [mammos-spindynamics 0.4.0](https://github.com/MaMMoS-project/mammos-spindynamics/tree/0.4.0) – 2026-03-06

### Changed

- Changed unit for spontaneous magnetization entities from `A/m` to `kA/m`. ([#68](https://github.com/MaMMoS-project/mammos-spindynamics/pull/68))

### Removed

- Material Nd2Fe14B has been removed from the database. ([#14](https://github.com/MaMMoS-project/mammos-spindynamics/pull/14))

### Misc

- Improved documentation for uppasd module. ([#58](https://github.com/MaMMoS-project/mammos-spindynamics/pull/58))
- The directory structure of the internal database is now defined by the chemical material rather than by an integer index. ([#59](https://github.com/MaMMoS-project/mammos-spindynamics/pull/59))


## [mammos-spindynamics 0.3.1](https://github.com/MaMMoS-project/mammos-spindynamics/tree/0.3.1) – 2025-12-18

### Fixed

- Indexing of `TemperatureSweepData` sub-runs. ([#54](https://github.com/MaMMoS-project/mammos-spindynamics/pull/54))
- Set the available number of `OMP_NUM_THREADS` in the notebooks run on Binder. ([#55](https://github.com/MaMMoS-project/mammos-spindynamics/pull/55))


## [mammos-spindynamics 0.3.0](https://github.com/MaMMoS-project/mammos-spindynamics/tree/0.3.0) – 2025-12-17

### Added

- Python interface for UppASD. ([#43](https://github.com/MaMMoS-project/mammos-spindynamics/pull/43))
- Implement Data classes to parse UppASD output. ([#42](https://github.com/MaMMoS-project/mammos-spindynamics/pull/42))


## [mammos-spindynamics 0.2.6](https://github.com/MaMMoS-project/mammos-spindynamics/tree/0.2.6) – 2025-12-12

### Fixed

- Fixed header of `M.csv` for Fe3Y and Fe2.33Ta0.67Y. ([#45](https://github.com/MaMMoS-project/mammos-spindynamics/pull/45))


## [mammos-spindynamics 0.2.5](https://github.com/MaMMoS-project/mammos-spindynamics/tree/0.2.5) – 2025-12-10

### Misc

- Materials Fe3Y and Fe2.33Ta0.67Y were added to the database. ([#41](https://github.com/MaMMoS-project/mammos-spindynamics/pull/41))


## [mammos-spindynamics 0.2.4](https://github.com/MaMMoS-project/mammos-spindynamics/tree/0.2.4) – 2025-12-02

### Misc

- Fix dependencies: add `numpy` as an explicit dependency. ([#38](https://github.com/MaMMoS-project/mammos-spindynamics/pull/38))


## [mammos-spindynamics 0.2.3](https://github.com/MaMMoS-project/mammos-spindynamics/tree/0.2.3) – 2025-08-12

### Misc

- Use [towncrier](https://towncrier.readthedocs.io) to generate changelog from fragments. Each new PR must include a changelog fragment. ([#21](https://github.com/MaMMoS-project/mammos-spindynamics/pull/21))
