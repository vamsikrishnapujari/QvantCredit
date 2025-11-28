# QvantCredit Example Usage and Testing

This document provides examples and test cases for QvantCredit.

## Quick Start Example

### 1. Basic Portfolio Optimization

```python
# Configuration:
- Number of Loans: 15
- Max Risk Level: 20%
- Target Return: 12%
- Portfolio Size: 8
- Quantum Reads: 100
- Use D-Wave QPU: Yes

# Expected Results:
- Selected loans: 7-9 loans
- Portfolio value: $1.5M - $3M
- Average risk: 4-7%
- ROI: 10-15%
```

### 2. Conservative Portfolio

```python
# Configuration:
- Number of Loans: 20
- Max Risk Level: 15%
- Target Return: 8%
- Portfolio Size: 5
- Quantum Reads: 150

# Expected Results:
- Focus on high credit score loans
- Lower default probabilities
- More stable returns
```

### 3. Aggressive Portfolio

```python
# Configuration:
- Number of Loans: 30
- Max Risk Level: 40%
- Target Return: 25%
- Portfolio Size: 15
- Quantum Reads: 200

# Expected Results:
- Higher potential returns
- Increased risk exposure
- More diversified sectors
```

## Test Scenarios

### Scenario 1: Tech-Heavy Portfolio
Generate a portfolio with seed 42, observe tech sector concentration.

### Scenario 2: Risk Minimization
Set Max Risk to 10%, observe conservative loan selection.

### Scenario 3: Return Maximization
Set Target Return to 25%, observe aggressive loan selection.

### Scenario 4: Quantum vs Classical
Run same parameters with QPU enabled/disabled, compare results.

## Performance Benchmarks

| Portfolio Size | Quantum Reads | Execution Time | Solution Quality |
|---------------|---------------|----------------|------------------|
| 10 loans      | 100 reads     | ~2-3 seconds   | Excellent        |
| 20 loans      | 200 reads     | ~5-7 seconds   | Excellent        |
| 30 loans      | 300 reads     | ~10-15 seconds | Good             |
| 50 loans      | 500 reads     | ~20-30 seconds | Fair             |

## API Usage Examples

### Using QvantCredit Functions Directly

```python
import pandas as pd
from app import create_credit_portfolio, build_qubo_model, solve_with_dwave

# Generate portfolio
portfolio = create_credit_portfolio(n_loans=15, seed=42)

# Build QUBO model
bqm = build_qubo_model(
    portfolio=portfolio,
    max_risk=20,
    target_return=12,
    portfolio_size_limit=8
)

# Solve with D-Wave
sampleset, used_qpu = solve_with_dwave(bqm, num_reads=100, use_real_qpu=True)

# Get best solution
best_solution = sampleset.first.sample
selected_loans = [i for i, v in best_solution.items() if v == 1]
print(f"Selected {len(selected_loans)} loans: {selected_loans}")
```

## Troubleshooting Tests

### Test 1: Verify D-Wave Connection
```bash
dwave ping
```

Expected output: Connection successful with timing information

### Test 2: Check Dependencies
```python
import streamlit
import dimod
from dwave.system import DWaveSampler
print("All imports successful!")
```

### Test 3: Simulated Annealing Fallback
Disable D-Wave QPU checkbox, verify app runs with Neal sampler.

## Advanced Usage

### Custom Sector Weights
Modify the `create_credit_portfolio` function to adjust sector distributions.

### Custom Risk Models
Extend the `build_qubo_model` function to add additional constraints.

### Integration with Real Data
Replace synthetic data with actual loan data:

```python
# Load real data
real_portfolio = pd.read_csv('actual_loans.csv')

# Ensure required columns exist
required_columns = ['amount', 'default_prob', 'correlation', 'sector', 
                   'duration', 'credit_score', 'ltv_ratio']
```

## Validation Checklist

- [ ] App launches without errors
- [ ] Portfolio generation works
- [ ] Optimization completes successfully
- [ ] Results displayed correctly
- [ ] Energy landscape shows variation
- [ ] Export functions work
- [ ] D-Wave connection established (if using QPU)
- [ ] Fallback to simulated annealing works

## Performance Tips

1. **Start small**: Test with 10-15 loans first
2. **Increase gradually**: Scale up as needed
3. **Monitor timing**: QPU time vs quality tradeoff
4. **Use caching**: Streamlit caches repeated operations
5. **Optimize reads**: 100-200 reads usually sufficient

## Expected Behavior

### Energy Landscapes
- Should show clear low-energy solutions
- Best solution should have lowest energy
- Multiple solutions may have similar energy

### Portfolio Composition
- Diversification across sectors
- Balance between risk and return
- Credit scores typically above 650 for selected loans

### Quantum Advantage
- More consistent results with QPU
- Better exploration of solution space
- Faster convergence to optimal solutions

## Known Limitations

1. Synthetic data may not reflect real-world complexities
2. QUBO formulation is simplified for demonstration
3. Sector correlations are simplified
4. No time-series or market condition modeling
5. Limited to binary selection (all-or-nothing loans)

## Future Enhancements

- [ ] Partial loan allocation (continuous variables)
- [ ] Multi-period optimization
- [ ] Market condition scenarios
- [ ] Historical backtesting
- [ ] Integration with credit bureaus
- [ ] Machine learning risk prediction
- [ ] Real-time market data feeds
