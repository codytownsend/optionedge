# Options Trading Engine

A Python-based options trade discovery engine that simulates the research workflow of a quantitative options desk. This system integrates live financial data from multiple APIs and outputs the 5 most favorable options trades based on quantitative selection rules.

## 🎯 Core Objectives

- **Input**: Real-time options chain data, fundamentals, technicals, macro data, sentiment
- **Processing**: Generate strategies, apply risk constraints, score and rank trades  
- **Output**: Console-friendly table with top 5 executable trades meeting all criteria

## 🔧 Key Features

- **Modular Architecture**: Clean separation of concerns with testable components
- **Multiple Data Sources**: Tradier, Yahoo Finance, FRED, QuiverQuant APIs
- **Risk Management**: Portfolio Greeks limits, sector diversification, capital constraints
- **Strategy Generation**: Credit spreads, iron condors, covered calls, calendar spreads
- **Real-time Filtering**: Quote freshness, liquidity, probability of profit constraints

## 📊 Trade Selection Criteria

- Maximum 5 trades selected
- Probability of Profit (POP) ≥ 65%
- Credit-to-max-loss ratio ≥ 33%
- Portfolio delta within [-0.30, +0.30] × (NAV/100k)
- Portfolio vega ≥ -0.05 × (NAV/100k)
- Max 2 trades per GICS sector
- Quote age ≤ 10 minutes
- Max loss ≤ $500 per trade

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd options-trading-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Required: TRADIER_API_KEY, YAHOO_RAPID_API_KEY, FRED_API_KEY
# Optional: QUIVER_API_KEY

# Customize settings
cp config/settings.yaml config/my_settings.yaml
# Edit my_settings.yaml with your portfolio parameters
```

### 3. Test Setup

```bash
# Test API connections
python -m src.presentation.cli.main --test-apis

# Validate configuration
python -m src.presentation.cli.main --validate-config

# View current settings
python -m src.presentation.cli.main --show-config
```

### 4. Run Market Scan

```bash
# Full market scan (Phase 2+)
python -m src.presentation.cli.main --scan

# Custom watchlist scan
python -m src.application.use_cases.scan_market --tickers AAPL,MSFT,GOOGL
```

## 📁 Project Structure

```
options_engine/
├── src/                     # Main source code
│   ├── data/               # Data models and repositories
│   ├── domain/             # Business logic and entities
│   ├── infrastructure/     # External integrations (APIs, cache)
│   ├── application/        # Use cases and configuration
│   └── presentation/       # Output formatting and CLI
├── tests/                  # Comprehensive test suite
├── config/                 # Configuration files
├── docs/                   # Documentation
└── scripts/                # Utility scripts
```

## 🔑 API Keys Required

| Service | Purpose | Required |
|---------|---------|----------|
| **Tradier** | Options chain data | ✅ Yes |
| **Yahoo Finance (RapidAPI)** | Fundamental data | ✅ Yes |
| **FRED** | Economic indicators | ✅ Yes |
| **QuiverQuant** | Sentiment data | ⚪ Optional |

## 🏗️ Development Phases

The project is built in 8 phases:

1. **Foundation & Data Infrastructure** ✅
2. **Options Data Collection & Processing** 🔄
3. **Strategy Generation & Greeks Management** 📅
4. **Filtering & Constraint Engine** 📅
5. **Scoring & Ranking Engine** 📅
6. **Output Generation & Validation** 📅
7. **Testing & Quality Assurance** 📅
8. **Performance Optimization & Monitoring** 📅

## 🛠️ Development

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit

# With coverage
pytest --cov=src

# Specific module
pytest tests/unit/test_models/
```

### Code Quality

```bash
# Format code
black src tests
isort src tests

# Type checking
mypy src

# Linting
flake8 src
```

## 📈 Output Format

The engine outputs a fixed-width console table:

```
Ticker | Strategy         | Legs                    | Thesis                         | POP
-------|------------------|-------------------------|--------------------------------|------
AAPL   | Put Credit Spread| 180P/175P Mar 15       | Bullish momentum, low IV      | 0.72
MSFT   | Iron Condor      | 300P/310C/320C/330C    | Range-bound, high IV crush    | 0.68
```

## ⚠️ Risk Disclaimer

This software is for educational and research purposes. Options trading involves substantial risk and is not suitable for all investors. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/your-org/options-trading-engine/issues)
- Discussions: [GitHub Discussions](https://github.com/your-org/options-trading-engine/discussions)