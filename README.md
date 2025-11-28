# 🔮 QvantCredit - Quantum Credit Risk Portfolio Optimization

A sophisticated Streamlit application that leverages **D-Wave quantum annealing** to optimize credit portfolios by balancing risk and return through quantum computing.

![QvantCredit](https://img.shields.io/badge/Quantum-Computing-blueviolet)
![D-Wave](https://img.shields.io/badge/D--Wave-Powered-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)

## ✨ Features

- **🔮 Real D-Wave Quantum Execution**: Connect to actual D-Wave quantum computers via Ocean SDK
- **💎 Beautiful Modern UI**: Gradient designs with interactive Plotly visualizations
- **📊 Portfolio Generation**: Create synthetic credit portfolios with realistic risk parameters
- **🧮 QUBO Optimization**: Transform portfolio optimization into quantum-ready QUBO models
- **📈 Interactive Analysis**: Multiple tabs for comprehensive portfolio and quantum analysis
- **⚡ Real-time Visualizations**: Energy landscapes, sector distributions, and risk metrics
- **💾 Export Capabilities**: Download optimized portfolios and analysis reports
- **🎯 Multi-objective Optimization**: Balance returns, minimize risk, control diversification

## 🚀 Quick Start

### Installation

```bash
# Clone or download the repository
cd Hackathon

# Install dependencies
pip install -r requirements.txt
```

### Configure D-Wave Access

To use real D-Wave quantum hardware:

```bash
# Set up D-Wave configuration
dwave config create
```

When prompted, enter your D-Wave API token. You can get a free token by signing up at [D-Wave Leap](https://cloud.dwavesys.com/leap/).

### Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

### 1. **Configure Parameters** (Sidebar)
   - Set the number of loans (5-50)
   - Adjust risk tolerance (5-50%)
   - Define target returns (5-30%)
   - Set maximum portfolio size
   - Configure quantum reads (50-1000)
   - Enable/disable real D-Wave QPU

### 2. **Generate Portfolio** (Tab 1: Portfolio Analysis)
   - Click "🎲 Generate Portfolio"
   - Review synthetic loan characteristics
   - Analyze sector distribution and risk profiles
   - Examine credit scores and default probabilities
   - View detailed portfolio data table

### 3. **Run Quantum Optimization** (Tab 2: Quantum Optimization)
   - Review optimization parameters
   - Click "🚀 Run Quantum Optimization"
   - Watch real-time progress:
     - QUBO model construction
     - Quantum annealing execution
     - Solution analysis
   - Get instant results from quantum computer

### 4. **Analyze Results** (Tab 3: Results)
   - View optimized portfolio metrics
   - Compare before/after statistics
   - Explore interactive visualizations:
     - Sector composition
     - Risk-return profiles
     - Energy landscapes
     - Credit score distributions
   - Download results (CSV/JSON)

### 5. **Technical Details** (Tab 4: Details)
   - Examine QUBO model structure
   - Review quantum execution statistics
   - Analyze solution quality metrics
   - View top quantum solutions

## 🔬 Technology Stack

- **Quantum Computing**: D-Wave Ocean SDK (dwave-ocean-sdk, dwave-system)
- **Web Framework**: Streamlit
- **Optimization**: DIMOD (Binary Quadratic Models)
- **Visualization**: Plotly Express & Graph Objects
- **Data Processing**: NumPy, Pandas
- **Graph Analysis**: NetworkX
- **Fallback Solver**: Neal (Simulated Annealing)

## 🧮 How It Works

### QUBO Formulation

The portfolio optimization problem is transformed into a **Quadratic Unconstrained Binary Optimization** (QUBO) problem:

**Minimize**: `E = Σᵢ hᵢxᵢ + Σᵢⱼ Jᵢⱼxᵢxⱼ`

Where:
- **xᵢ ∈ {0,1}**: Binary decision variable (0 = reject loan, 1 = accept loan)
- **hᵢ**: Linear coefficients representing expected returns (negative for maximization)
- **Jᵢⱼ**: Quadratic coefficients representing correlation-based risk penalties
- **E**: Total energy (lower energy = better solution)

### Quantum Annealing Process

1. **Initialization**: Map QUBO to quantum hardware qubits
2. **Quantum Tunneling**: Qubits explore solution space via quantum superposition
3. **Annealing**: Gradually reduce quantum fluctuations
4. **Measurement**: Sample low-energy states (optimal solutions)
5. **Analysis**: Extract and evaluate best portfolio configuration

### Why Quantum?

Portfolio optimization with correlation constraints is **NP-hard**. Classical algorithms struggle with:
- Exponential growth in solution space
- Local minima traps
- Computational time scaling

**Quantum annealing advantages**:
- Quantum tunneling escapes local minima
- Parallel exploration of solution space
- Natural fit for QUBO problems
- Superior performance on complex correlations

## 📊 Key Metrics

| Metric | Description |
|--------|-------------|
| **Default Probability** | Likelihood of borrower default (0-100%) |
| **Expected Return** | Amount × (1 - Default Probability) |
| **Portfolio Risk** | Weighted average default probability |
| **ROI** | Return on Investment percentage |
| **Credit Score** | FICO-style creditworthiness (300-850) |
| **LTV Ratio** | Loan-to-Value ratio (lower = less risky) |
| **Correlation** | Cross-loan risk correlation measure |

## 🎨 UI Features

- **Gradient Color Schemes**: Purple-blue gradients for quantum aesthetic
- **Responsive Layout**: Wide-screen optimized with flexible columns
- **Interactive Charts**: Hover, zoom, and explore data dynamically
- **Real-time Progress**: Live updates during quantum execution
- **Custom Styling**: Beautiful metric cards and info boxes
- **Animated Feedback**: Success animations and progress indicators

## 🔐 D-Wave Setup

### Free Tier Access

1. Visit [D-Wave Leap](https://cloud.dwavesys.com/leap/)
2. Sign up for free account (includes QPU time)
3. Navigate to API Token section
4. Copy your API token
5. Run `dwave config create` and paste token

### Configuration File

The D-Wave configuration is stored in `~/.config/dwave/dwave.conf`:

```ini
[defaults]
endpoint = https://cloud.dwavesys.com/sapi/
token = YOUR_API_TOKEN_HERE
```

## 💡 Tips for Best Results

- **Start Small**: Begin with 10-20 loans to understand the system
- **Quantum Reads**: Use 100-200 reads for good quality (higher = better but slower)
- **Portfolio Size**: Limit to 5-15 loans for focused optimization
- **Risk Tolerance**: Lower values produce conservative portfolios
- **QPU vs Simulation**: Real QPU gives authentic quantum results, simulation is faster for testing
- **Sector Diversity**: More sectors available increases diversification potential

## 🐛 Troubleshooting

### "D-Wave QPU not available"
- **Cause**: No API token or invalid credentials
- **Solution**: Run `dwave config create` and enter valid token
- **Workaround**: Uncheck "Use Real D-Wave QPU" for simulated annealing

### "No valid solution found"
- **Cause**: Constraints too restrictive
- **Solution**: Increase portfolio size limit or adjust risk parameters
- **Alternative**: Generate new portfolio with different seed

### Import errors
- **Cause**: Missing dependencies
- **Solution**: `pip install -r requirements.txt --upgrade`

### Slow execution
- **Cause**: Too many quantum reads or large portfolio
- **Solution**: Reduce reads to 50-100 or decrease loan count

## 📚 Documentation & Resources

- **D-Wave Ocean Documentation**: https://docs.ocean.dwavesys.com/
- **Quantum Annealing Tutorial**: https://www.dwavesys.com/learn
- **QUBO Formulation Guide**: https://docs.ocean.dwavesys.com/en/stable/concepts/qubo.html
- **Streamlit Documentation**: https://docs.streamlit.io/

## 🎓 Educational Value

This application demonstrates:
- Practical quantum computing applications in finance
- QUBO problem formulation techniques
- Quantum annealing vs classical optimization
- Real-world credit risk modeling
- Interactive data visualization best practices

## 🔒 Disclaimer

QvantCredit is an **educational and research tool**. For production financial applications:
- Implement proper risk management practices
- Ensure regulatory compliance
- Validate with domain experts
- Use certified financial models
- Conduct thorough backtesting

## 📄 License

MIT License - Feel free to use, modify, and distribute with attribution.

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional portfolio constraints
- More sophisticated risk models
- Historical backtesting capabilities
- Multi-period optimization
- Machine learning integration
- Additional quantum solvers

## 💬 Support

For issues or questions:
- Check the Documentation tab in the app
- Review D-Wave Ocean SDK docs
- Submit issues on GitHub (if hosted)
- Consult D-Wave community forums

## 🌟 Acknowledgments

- **D-Wave Systems** for quantum computing access
- **Streamlit** for the amazing web framework
- **Plotly** for interactive visualizations
- Quantum computing research community

---

<div align="center">

**🔮 Powered by D-Wave Quantum Computing | Built with ❤️ and Streamlit**

*Leveraging quantum annealing for superior financial optimization*

</div>
