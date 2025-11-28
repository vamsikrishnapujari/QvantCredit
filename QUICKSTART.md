# 🚀 QvantCredit - Quick Start Guide

## Choose Your Version

### Option 1: Original Quantum-Only Version
**File:** `app.py`

**Features:**
- ✅ D-Wave quantum annealing
- ✅ Portfolio visualization
- ✅ Basic optimization
- ✅ Lighter dependencies
- ✅ Faster startup

**Best for:**
- Learning quantum computing
- Quick demos
- Limited computing resources
- Focus on quantum algorithms

**Run:**
```bash
streamlit run app.py
```

---

### Option 2: AI-Enhanced Version (Recommended)
**File:** `app_ai_enhanced.py`

**Features:**
- ✅ Everything from Option 1, PLUS:
- 🤖 **Machine Learning**: XGBoost credit risk prediction
- 🧠 **Neural Networks**: Deep learning models
- 🔍 **Anomaly Detection**: Isolation Forest
- 💬 **NLP Analysis**: Transformer-based sentiment
- 📊 **Explainable AI**: SHAP values
- 🎮 **Reinforcement Learning**: Q-learning agent

**Best for:**
- Production use cases
- Comprehensive analysis
- AI/ML research
- Maximum capabilities

**Run:**
```bash
streamlit run app_ai_enhanced.py
```

---

## Installation Steps

### 1. Basic Installation (Both Versions)

```bash
# Navigate to project
cd /home/vamsikrishna-pujari/Desktop/Hackathon

# Install core dependencies
pip install streamlit numpy pandas plotly dwave-ocean-sdk networkx
```

### 2. AI Dependencies (For AI-Enhanced Version Only)

```bash
# Install AI/ML libraries
pip install scikit-learn xgboost transformers shap

# Install PyTorch (choose based on your system)
# For CPU:
pip install torch

# For CUDA 11.8 (NVIDIA GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Complete Installation (All at Once)

```bash
# Install everything from requirements
pip install -r requirements.txt
```

---

## Feature Comparison

| Feature | app.py | app_ai_enhanced.py |
|---------|--------|-------------------|
| **Quantum Annealing** | ✅ | ✅ |
| **D-Wave Integration** | ✅ | ✅ |
| **Portfolio Optimization** | ✅ | ✅ |
| **Visualization** | ✅ | ✅ |
| **ML Risk Prediction** | ❌ | ✅ |
| **Neural Networks** | ❌ | ✅ |
| **Anomaly Detection** | ❌ | ✅ |
| **NLP Analysis** | ❌ | ✅ |
| **Explainable AI (SHAP)** | ❌ | ✅ |
| **Reinforcement Learning** | ❌ | ✅ |
| **Installation Size** | ~200 MB | ~2-3 GB |
| **Startup Time** | 2-3 sec | 5-10 sec |
| **Memory Usage** | 200-500 MB | 1-2 GB |

---

## Testing Your Installation

### Test Original Version
```bash
streamlit run app.py
```

Expected output:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### Test AI-Enhanced Version
```bash
streamlit run app_ai_enhanced.py
```

First run will download transformer models (~500 MB). This is normal and happens once.

---

## Common Issues

### Issue 1: "streamlit: command not found"
**Solution:**
```bash
# Make sure pip packages are in PATH
export PATH="$HOME/.local/bin:$PATH"

# Or use python -m
python -m streamlit run app.py
```

### Issue 2: Import errors for AI libraries
**Solution:**
```bash
# Reinstall specific package
pip install --upgrade transformers

# Or use the original app.py instead
streamlit run app.py
```

### Issue 3: CUDA/GPU errors
**Solution:**
```bash
# Force CPU mode for PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue 4: Slow first load (AI version)
**Reason:** Downloading transformer models (happens once)
**Wait time:** 2-5 minutes depending on internet speed

---

## Performance Tips

### For Faster AI-Enhanced Version:
1. **Reduce portfolio size** (start with 20 loans)
2. **Lower quantum reads** (use 100 instead of 1000)
3. **Disable unused features** in sidebar
4. **Use CPU if GPU causes issues**

### For Maximum Performance:
1. Install with GPU support
2. Use SSD for model storage
3. Allocate 4+ GB RAM
4. Close other applications

---

## What to Run First?

### Beginners: Start with `app.py`
- Simpler interface
- Focus on quantum concepts
- Faster to learn
- Less overwhelming

### Advanced Users: Jump to `app_ai_enhanced.py`
- All capabilities available
- Production-ready features
- Comprehensive analysis
- Research-grade tools

---

## System Requirements

### Minimum (app.py):
- CPU: 2 cores
- RAM: 2 GB
- Storage: 500 MB
- Internet: For D-Wave API

### Recommended (app_ai_enhanced.py):
- CPU: 4+ cores
- RAM: 8 GB
- Storage: 5 GB
- GPU: Optional but helpful
- Internet: For model downloads

---

## Next Steps

1. ✅ Choose your version
2. ✅ Install dependencies
3. ✅ Run the application
4. 📖 Read the README (README.md or README_AI.md)
5. 🎮 Try the demo with sample data
6. 🔑 Get D-Wave API token (optional)
7. 🚀 Optimize real portfolios!

---

## Support

- **Documentation**: See README.md and README_AI.md
- **D-Wave Setup**: Visit https://cloud.dwavesys.com/leap/
- **Issues**: Check troubleshooting sections in READMEs

---

<div align="center">

**🔮 Choose Quantum | 🤖 Choose AI | 🚀 Choose Both!**

*QvantCredit - Your Credit Risk Analysis Platform*

</div>
