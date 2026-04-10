# Changelog

All notable changes to this project will be documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.0.1] - 2025-01-01

### Added

- `bars` module: time, tick, volume, and dollar bar constructors (López de Prado, Ch. 2)
- `filters` module: symmetric CUSUM filter for event-driven sampling (López de Prado, Snippet 2.4)
- `labeling` module: triple-barrier labeling with configurable profit-take, stop-loss, and vertical barriers (López de Prado, Ch. 3)
- `weights` module: concurrent label counting, average uniqueness, and exponential time-decay sample weights (López de Prado, Ch. 4)
- `esl_loaders` module: ATP/WTA tennis and TMDB movie dataset loaders for ESL notebooks
- Marimo notebooks for AFML Chapters 2–4 and ESL Chapters 2–12
- MkDocs Material documentation site with API reference
- GitHub Actions CI: lint, type check, tests with coverage, docs build
