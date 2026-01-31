# Contributing to DAG Scheduler Benchmark

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and collaborative environment for all contributors.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Relevant code snippets or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- Clear description of the proposed feature
- Rationale for why it would be useful
- Example use cases
- Potential implementation approach (if applicable)

### Pull Requests

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation as needed

3. **Test your changes**
   - Run the benchmark with different scenarios
   - Verify output files are generated correctly
   - Test with different core counts

4. **Commit your changes**
   ```bash
   git commit -m "Add: Brief description of your changes"
   ```
   Use conventional commit messages:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Refactor:` for code improvements
   - `Docs:` for documentation changes
   - `Test:` for test additions/modifications

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Provide clear description of changes
   - Reference any related issues
   - Include test results if applicable

## Development Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use descriptive variable names
- Keep functions focused and modular
- Add docstrings for classes and functions
- Keep comments concise and meaningful

### Adding New Schedulers

To add a new scheduling algorithm:

1. Add the algorithm name to `scheduler_modes` list
2. Implement sorting logic in the `sort_tasks()` function
3. Add any required pre-computation (like b-levels, t-levels)
4. Document the algorithm in README.md
5. Test with multiple scenarios

Example:
```python
elif self.scheduler_mode == 'your_algorithm':
    # Compute any required metrics
    tasks.sort(key=lambda x: your_priority_function(x))
```

### Adding New Test Scenarios

To add a new test scenario:

1. Add entry to `test_scenarios` dictionary
2. Specify parameters: S (nodes), N (total size), structure, edge_prob
3. Add description
4. Test to verify graph generation works correctly

Example:
```python
'your_scenario': {
    'S': 25,
    'N': 15000,
    'structure': 'balanced',
    'edge_prob': 0.3,
    'description': 'Your scenario description'
}
```

### Testing

Before submitting a PR:

- Test with `run_mode = 'quick'` for fast verification
- Run with at least 3 trials (`num_graphs = 3`)
- Verify output files are generated
- Check that metrics are calculated correctly
- Test with different core counts

## Areas for Contribution

We welcome contributions in these areas:

- **New Scheduling Algorithms**: Implement additional DAG scheduling algorithms
- **Performance Optimizations**: Improve execution speed or memory usage
- **Visualization**: Add graph visualization or performance plots
- **Documentation**: Improve examples, tutorials, or explanations
- **Testing**: Add unit tests or integration tests
- **Output Formats**: Support for CSV, JSON, or other output formats
- **Platform Support**: Testing and fixes for different operating systems

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the "question" label
- Contact the maintainer

## Recognition

Contributors will be acknowledged in the project documentation. Thank you for helping improve this project!
