# Options Trading Engine - Deployment Guide

## Overview
This guide provides comprehensive instructions for deploying the Options Trading Engine in production environments.

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+) or macOS 10.15+
- **Python**: Version 3.9 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **CPU**: Minimum 4 cores (8 cores recommended)
- **Storage**: 100GB available disk space
- **Network**: Stable internet connection with low latency

### Required Dependencies
```bash
# System packages
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-venv python3.9-dev
sudo apt-get install -y build-essential libssl-dev libffi-dev
sudo apt-get install -y postgresql postgresql-contrib redis-server

# Python packages (installed via pip)
pip install -r requirements.txt
```

## Environment Setup

### 1. Create Virtual Environment
```bash
# Create virtual environment
python3.9 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 2. Install Dependencies
```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### 3. Environment Configuration
Create `.env` file in the project root:

```bash
# API Keys (Required)
TRADIER_API_KEY=your_tradier_api_key_here
YAHOO_FINANCE_API_KEY=your_yahoo_finance_api_key_here
FRED_API_KEY=your_fred_api_key_here
QUIVER_QUANT_API_KEY=your_quiver_quant_api_key_here

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/options_engine
REDIS_URL=redis://localhost:6379/0

# Application Configuration
ENV=production
DEBUG=false
LOG_LEVEL=INFO
LOG_FILE=/var/log/options_engine/app.log

# Performance Configuration
MAX_WORKERS=8
CACHE_TTL=300
API_TIMEOUT=30
BATCH_SIZE=100

# Monitoring Configuration
MONITORING_ENABLED=true
METRICS_PORT=8080
HEALTH_CHECK_INTERVAL=60

# Alert Configuration
ALERT_EMAIL_ENABLED=true
ALERT_EMAIL_FROM=alerts@yourcompany.com
ALERT_EMAIL_TO=admin@yourcompany.com
ALERT_SMTP_SERVER=smtp.gmail.com
ALERT_SMTP_PORT=587
ALERT_SMTP_USERNAME=your_smtp_username
ALERT_SMTP_PASSWORD=your_smtp_password

# Slack Alerts (Optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

### 4. Database Setup
```bash
# Create database user and database
sudo -u postgres psql
CREATE USER options_engine_user WITH PASSWORD 'secure_password';
CREATE DATABASE options_engine OWNER options_engine_user;
GRANT ALL PRIVILEGES ON DATABASE options_engine TO options_engine_user;
\q

# Initialize database schema
python scripts/init_database.py
```

### 5. Directory Structure
```bash
# Create required directories
mkdir -p /var/log/options_engine
mkdir -p /var/lib/options_engine/data
mkdir -p /var/lib/options_engine/cache
mkdir -p /etc/options_engine

# Set permissions
sudo chown -R $USER:$USER /var/log/options_engine
sudo chown -R $USER:$USER /var/lib/options_engine
sudo chown -R $USER:$USER /etc/options_engine
```

## Configuration

### 1. Application Configuration
Copy and modify the configuration files:

```bash
# Copy configuration files
cp config/settings.yaml /etc/options_engine/settings.yaml
cp config/logging.yaml /etc/options_engine/logging.yaml

# Edit configuration as needed
nano /etc/options_engine/settings.yaml
```

### 2. Logging Configuration
Update logging configuration in `/etc/options_engine/logging.yaml`:

```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
  detailed:
    format: '%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: detailed
    filename: /var/log/options_engine/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: /var/log/options_engine/error.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  options_engine:
    level: INFO
    handlers: [console, file, error_file]
    propagate: false

root:
  level: INFO
  handlers: [console, file]
```

### 3. Service Configuration
Create systemd service file `/etc/systemd/system/options-engine.service`:

```ini
[Unit]
Description=Options Trading Engine
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=simple
User=options_engine
Group=options_engine
WorkingDirectory=/opt/options_engine
Environment=PATH=/opt/options_engine/venv/bin
ExecStart=/opt/options_engine/venv/bin/python -m src.presentation.cli.main
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

# Environment variables
EnvironmentFile=/etc/options_engine/environment

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/options_engine /var/lib/options_engine

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

## Deployment Steps

### 1. Pre-deployment Validation
```bash
# Run configuration validation
python scripts/validate_config.py

# Run database connectivity test
python scripts/test_database.py

# Run API connectivity test
python scripts/test_api_connections.py

# Run unit tests
python -m pytest tests/unit/

# Run integration tests
python -m pytest tests/integration/
```

### 2. Deploy Application
```bash
# Clone repository
git clone https://github.com/your-org/options_engine.git
cd options_engine

# Switch to production branch
git checkout production

# Create deployment user
sudo useradd -r -s /bin/false options_engine

# Copy application to deployment directory
sudo cp -r . /opt/options_engine
sudo chown -R options_engine:options_engine /opt/options_engine

# Install dependencies
sudo -u options_engine -H /opt/options_engine/venv/bin/pip install -r requirements.txt
```

### 3. Database Migration
```bash
# Run database migrations
sudo -u options_engine python /opt/options_engine/scripts/migrate_database.py

# Verify database schema
sudo -u options_engine python /opt/options_engine/scripts/verify_schema.py
```

### 4. Service Deployment
```bash
# Create environment file
sudo cp /opt/options_engine/.env /etc/options_engine/environment

# Install systemd service
sudo cp /opt/options_engine/deploy/options-engine.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start service
sudo systemctl enable options-engine
sudo systemctl start options-engine

# Check service status
sudo systemctl status options-engine
```

### 5. Post-deployment Verification
```bash
# Check service logs
sudo journalctl -u options-engine -f

# Check application logs
tail -f /var/log/options_engine/app.log

# Run health check
curl http://localhost:8080/health

# Run end-to-end test
python scripts/e2e_test.py
```

## Monitoring and Maintenance

### 1. Log Management
```bash
# Set up log rotation
sudo cp /opt/options_engine/deploy/logrotate.conf /etc/logrotate.d/options-engine

# Test log rotation
sudo logrotate -d /etc/logrotate.d/options-engine
```

### 2. Monitoring Setup
```bash
# Install monitoring agent (if using external monitoring)
# Configure metrics collection
# Set up alerting rules

# Internal monitoring endpoints
curl http://localhost:8080/metrics
curl http://localhost:8080/health
curl http://localhost:8080/status
```

### 3. Backup Strategy
```bash
# Database backup
pg_dump options_engine > backup_$(date +%Y%m%d_%H%M%S).sql

# Configuration backup
tar -czf config_backup_$(date +%Y%m%d_%H%M%S).tar.gz /etc/options_engine/

# Create automated backup script
sudo cp /opt/options_engine/scripts/backup.sh /etc/cron.daily/options-engine-backup
sudo chmod +x /etc/cron.daily/options-engine-backup
```

## Scaling and High Availability

### 1. Load Balancing
```bash
# Configure nginx as reverse proxy
sudo apt-get install nginx

# Copy nginx configuration
sudo cp /opt/options_engine/deploy/nginx.conf /etc/nginx/sites-available/options-engine
sudo ln -s /etc/nginx/sites-available/options-engine /etc/nginx/sites-enabled/

# Test and reload nginx
sudo nginx -t
sudo systemctl reload nginx
```

### 2. Database Clustering
```bash
# Set up PostgreSQL streaming replication
# Configure Redis cluster for caching
# Implement connection pooling
```

### 3. Application Clustering
```bash
# Run multiple instances
sudo systemctl start options-engine@1
sudo systemctl start options-engine@2
sudo systemctl start options-engine@3

# Configure load balancer
# Set up shared session storage
```

## Security

### 1. SSL/TLS Configuration
```bash
# Generate SSL certificates
sudo certbot --nginx -d your-domain.com

# Configure SSL in nginx
# Set up certificate auto-renewal
```

### 2. Firewall Configuration
```bash
# Configure firewall rules
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow from 10.0.0.0/8 to any port 8080
sudo ufw enable
```

### 3. Security Hardening
```bash
# Set up fail2ban
sudo apt-get install fail2ban
sudo cp /opt/options_engine/deploy/fail2ban.conf /etc/fail2ban/jail.local

# Configure AppArmor/SELinux
# Set up intrusion detection
# Configure security monitoring
```

## Troubleshooting

### 1. Common Issues
```bash
# Service won't start
sudo systemctl status options-engine
sudo journalctl -u options-engine --no-pager -l

# Database connection issues
python scripts/test_database.py

# API connection issues
python scripts/test_api_connections.py

# Performance issues
python scripts/performance_check.py
```

### 2. Log Analysis
```bash
# Application logs
tail -f /var/log/options_engine/app.log

# Error logs
tail -f /var/log/options_engine/error.log

# System logs
sudo journalctl -u options-engine -f
```

### 3. Recovery Procedures
```bash
# Service recovery
sudo systemctl restart options-engine

# Database recovery
# Restore from backup
psql options_engine < backup_file.sql

# Configuration recovery
tar -xzf config_backup.tar.gz -C /
sudo systemctl restart options-engine
```

## Performance Optimization

### 1. Database Optimization
```sql
-- Create indexes for better performance
CREATE INDEX idx_options_expiration ON options (expiration_date);
CREATE INDEX idx_trades_timestamp ON trades (created_at);
CREATE INDEX idx_market_data_symbol ON market_data (symbol);
```

### 2. Caching Strategy
```bash
# Redis configuration optimization
# Connection pooling
# Cache warming strategies
```

### 3. Application Optimization
```python
# Enable performance monitoring
# Configure thread pool sizes
# Optimize API batch sizes
```

## Maintenance

### 1. Regular Maintenance Tasks
```bash
# Database maintenance
sudo -u options_engine python scripts/db_maintenance.py

# Log cleanup
sudo logrotate -f /etc/logrotate.d/options-engine

# Cache cleanup
redis-cli FLUSHEXPIRED

# System updates
sudo apt-get update && sudo apt-get upgrade
```

### 2. Health Checks
```bash
# Automated health monitoring
curl http://localhost:8080/health

# Performance monitoring
curl http://localhost:8080/metrics

# Resource monitoring
python scripts/resource_check.py
```

## Rollback Procedures

### 1. Application Rollback
```bash
# Stop current service
sudo systemctl stop options-engine

# Restore previous version
sudo cp -r /opt/options_engine_backup /opt/options_engine

# Restart service
sudo systemctl start options-engine
```

### 2. Database Rollback
```bash
# Restore database from backup
psql options_engine < previous_backup.sql

# Run migration rollback if needed
python scripts/rollback_migration.py
```

## Support and Maintenance

### 1. Contact Information
- **Operations Team**: ops@yourcompany.com
- **Development Team**: dev@yourcompany.com
- **Emergency Contact**: +1-XXX-XXX-XXXX

### 2. Documentation
- **API Documentation**: /docs/api/
- **Configuration Reference**: /docs/configuration/
- **Troubleshooting Guide**: /docs/troubleshooting/

### 3. Monitoring Dashboards
- **Application Metrics**: http://monitoring.yourcompany.com/options-engine
- **System Metrics**: http://monitoring.yourcompany.com/system
- **Alert Management**: http://alerts.yourcompany.com