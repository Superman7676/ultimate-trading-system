# Contributing to Ultimate Trading System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

- Be respectful and professional
- Provide constructive feedback
- Report bugs and issues responsibly
- No harassment or discrimination

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/Superman7676/ultimate-trading-system/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Error messages/logs

### Requesting Features

1. Check existing feature requests
2. Create a new issue with:
   - Clear use case
   - Expected behavior
   - Why this feature is valuable
   - Any alternative approaches

### Submitting Code Changes

1. **Fork the repository**
   ```bash
   # Click "Fork" button on GitHub
   git clone https://github.com/YOUR_USERNAME/ultimate-trading-system.git
   cd ultimate-trading-system
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/your-bug-fix
   ```

3. **Make your changes**
   - Follow PEP 8 style guide
   - Add comments for complex logic
   - Update docstrings
   - Add/update tests

4. **Test your changes**
   ```bash
   # Run tests
   pytest tests/
   
   # Check code quality
   pylint src/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description"
   # or
   git commit -m "Fix: description"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Go to GitHub and click "Compare & pull request"
   - Provide clear description of changes
   - Link related issues
   - Ensure CI passes

## Code Style Guidelines

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use type hints where possible

### Example:

```python
#!/usr/bin/env python3
"""
Module description
"""

from typing import Optional, List


def calculate_rsi(
    prices: List[float],
    period: int = 14
) -> Optional[List[float]]:
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: List of closing prices
        period: RSI period (default 14)
    
    Returns:
        List of RSI values or None if insufficient data
    
    Raises:
        ValueError: If period is invalid
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    if len(prices) < period:
        return None
    
    # Implementation here
    rsi_values = []
    # ...
    
    return rsi_values
```

### Naming Conventions

```python
# Classes: PascalCase
class TechnicalIndicators:
    pass

# Functions/methods: snake_case
def calculate_moving_average():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_POSITION_SIZE = 10000

# Private: _leading_underscore
def _internal_function():
    pass
```

## Testing

### Writing Tests

Create tests in `tests/` directory:

```python
import unittest
from indicators import TechnicalIndicators


class TestIndicators(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.prices = [100, 101, 102, 103, 104]
    
    def test_sma_calculation(self):
        """Test SMA calculation"""
        result = TechnicalIndicators.simple_moving_average(
            self.prices, period=2
        )
        self.assertIsNotNone(result)
    
    def test_invalid_period(self):
        """Test with invalid period"""
        with self.assertRaises(ValueError):
            TechnicalIndicators.relative_strength_index(
                self.prices, period=-1
            )


if __name__ == '__main__':
    unittest.main()
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_indicators.py

# Run with coverage
pytest --cov=src tests/
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def complex_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Longer description explaining what the function does,
    any important details, and edge cases.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When value is invalid
        RuntimeError: When runtime error occurs
    
    Example:
        >>> result = complex_function('test', 42)
        >>> result
        True
    """
    pass
```

### README Updates

If you add new features:
1. Update README.md with new usage examples
2. Add to feature list
3. Update project structure if needed

## Commit Message Guidelines

### Format

```
<type>: <subject>

<body>

<footer>
```

### Type

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Maintenance

### Example

```
feat: add Stochastic indicator

Implement Stochastic oscillator calculation with
%K and %D smoothing. Includes overbought/oversold
level detection.

Closes #123
```

## Pull Request Process

1. **Before submitting:**
   - Run tests locally
   - Update documentation
   - Follow code style
   - Rebase on main branch

2. **PR Description:**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation
   
   ## Testing
   Describe testing performed
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] No new warnings generated
   - [ ] Changes work locally
   
   ## Related Issues
   Closes #123
   ```

3. **Review process:**
   - Address feedback promptly
   - Push updates to same branch
   - Comment on feedback
   - Request re-review when ready

## Development Setup

```bash
# Clone repository
git clone https://github.com/Superman7676/ultimate-trading-system.git
cd ultimate-trading-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Install pre-commit hooks
pre-commit install
```

## Areas for Contribution

### High Priority
- [ ] More backtesting strategies
- [ ] Risk management modules
- [ ] Performance optimization
- [ ] Test coverage improvement

### Medium Priority
- [ ] Additional indicators
- [ ] Better error handling
- [ ] Code documentation
- [ ] Example notebooks

### Nice to Have
- [ ] UI improvements
- [ ] Additional data sources
- [ ] Mobile app
- [ ] Cloud deployment guides

## Questions?

- **GitHub Issues**: For bugs and features
- **Discussions**: For questions and ideas
- **Email**: Superman7676@example.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Ultimate Trading System! 🚀
