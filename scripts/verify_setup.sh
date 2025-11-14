#!/bin/bash

# Setup Verification Script
# Checks if all prerequisites are installed

echo "======================================"
echo "Verifying Setup Prerequisites"
echo "======================================"

MISSING=0

# Check Python
echo -n "Checking Python 3.9+... "
if command -v python3 &> /dev/null; then
    VERSION=$(python3 --version | awk '{print $2}')
    echo "✅ Found: $VERSION"
else
    echo "❌ Not found"
    MISSING=1
fi

# Check Docker
echo -n "Checking Docker... "
if command -v docker &> /dev/null; then
    VERSION=$(docker --version | awk '{print $3}' | tr -d ',')
    echo "✅ Found: $VERSION"
else
    echo "❌ Not found - Install Docker Desktop from https://docker.com"
    MISSING=1
fi

# Check Java
echo -n "Checking Java 11+... "
if command -v java &> /dev/null; then
    VERSION=$(java -version 2>&1 | head -n 1)
    echo "✅ Found: $VERSION"
else
    echo "❌ Not found - Install Java from https://adoptium.net"
    MISSING=1
fi

# Check pip packages
echo -n "Checking Python packages... "
if python3 -c "import fastapi, pyspark, catboost" 2>/dev/null; then
    echo "✅ Installed"
else
    echo "⚠️  Not all packages installed - Run: pip3 install -r requirements.txt"
fi

echo "======================================"
if [ $MISSING -eq 0 ]; then
    echo "✅ All prerequisites installed!"
else
    echo "⚠️  Some prerequisites missing - see above"
fi
echo "======================================"
