#!/bin/bash

# QvantCredit Setup Script
# This script installs all dependencies and configures D-Wave access

echo "🔮 QvantCredit Setup Script"
echo "============================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Found Python $python_version"
echo ""

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt
echo ""

# Check if D-Wave is configured
echo "🔐 Checking D-Wave configuration..."
if dwave config inspect 2>/dev/null | grep -q "endpoint"; then
    echo "✅ D-Wave configuration found!"
    echo ""
else
    echo "⚠️  D-Wave not configured."
    echo ""
    echo "To use real D-Wave quantum hardware:"
    echo "1. Sign up at https://cloud.dwavesys.com/leap/"
    echo "2. Get your API token from the dashboard"
    echo "3. Run: dwave config create"
    echo "4. Enter your API token when prompted"
    echo ""
    read -p "Would you like to configure D-Wave now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        dwave config create
    else
        echo "⏭️  Skipping D-Wave configuration."
        echo "   You can still use the app with simulated annealing."
    fi
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start the application, run:"
echo "   streamlit run app.py"
echo ""
