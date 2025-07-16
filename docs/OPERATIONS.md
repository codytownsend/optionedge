# Options Trading Engine - Operations Guide

## Overview
This guide provides comprehensive operational procedures for running and maintaining the Options Trading Engine in production.

## Daily Operations

### 1. System Health Check
```bash
# Morning health check routine
#!/bin/bash

echo "=== Options Trading Engine Health Check ==="
echo "Timestamp: $(date)"
echo ""

# Check system status
echo "1. System Status:"
sudo systemctl status options-engine --no-pager

# Check resource usage
echo "2. Resource Usage:"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')%"
echo "Memory: $(free -h | awk '/^Mem/ {print $3 "/" $2 " (" $3/$2*100 "%)"}')%"
echo "Disk: $(df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 ")"}')%"

# Check application health
echo "3. Application Health:"
curl -s http://localhost:8080/health | jq '.'

# Check database connectivity
echo "4. Database Status:"
python3 -c "
import psycopg2
try:
    conn = psycopg2.connect(host='localhost', database='options_engine', user='options_engine_user')
    print('Database: Connected')
    conn.close()
except Exception as e:
    print(f'Database: Error - {e}')
"

# Check API connectivity
echo "5. API Status:"
python3 scripts/test_api_connections.py

# Check recent errors
echo "6. Recent Errors:"
tail -20 /var/log/options_engine/error.log

echo ""
echo "=== Health Check Complete ==="
```

### 2. Performance Monitoring
```bash
# Performance monitoring script
#!/bin/bash

echo "=== Performance Monitoring ==="

# Application metrics
echo "Application Metrics:"
curl -s http://localhost:8080/metrics | jq '.performance'

# Database performance
echo "Database Performance:"
psql -d options_engine -c "
SELECT 
    schemaname,
    tablename,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch
FROM pg_stat_user_tables
ORDER BY seq_tup_read DESC
LIMIT 10;
"

# Cache performance
echo "Cache Performance:"
redis-cli info stats | grep -E "(keyspace_hits|keyspace_misses)"

# Network performance
echo "Network Performance:"
ping -c 5 api.tradier.com
ping -c 5 finance.yahoo.com
```

### 3. Log Analysis
```bash
# Log analysis routine
#!/bin/bash

echo "=== Log Analysis ==="

# Error log analysis
echo "Error Summary (last 24 hours):"
grep "ERROR" /var/log/options_engine/app.log | \
    grep "$(date -d '24 hours ago' +'%Y-%m-%d')" | \
    cut -d' ' -f4- | sort | uniq -c | sort -nr

# Performance warnings
echo "Performance Warnings:"
grep "WARN.*slow\|WARN.*timeout\|WARN.*memory" /var/log/options_engine/app.log | tail -10

# API errors
echo "API Errors:"
grep "API.*error\|API.*failed" /var/log/options_engine/app.log | tail -10

# Database issues
echo "Database Issues:"
grep "database\|postgres\|connection" /var/log/options_engine/app.log | grep -i error | tail -10
```

## Weekly Operations

### 1. Performance Review
```bash
# Weekly performance review
#!/bin/bash

echo "=== Weekly Performance Review ==="
echo "Week of: $(date -d '7 days ago' +'%Y-%m-%d') to $(date +'%Y-%m-%d')"

# Generate performance report
python3 scripts/performance_report.py --days 7

# Database maintenance
echo "Database Maintenance:"
psql -d options_engine -c "
ANALYZE;
VACUUM ANALYZE;
REINDEX DATABASE options_engine;
"

# Cache optimization
echo "Cache Optimization:"
redis-cli FLUSHEXPIRED
redis-cli MEMORY USAGE options_engine:*

# Log rotation
echo "Log Rotation:"
sudo logrotate -f /etc/logrotate.d/options-engine

# Backup verification
echo "Backup Verification:"
ls -la /var/backups/options_engine/ | tail -7
```

### 2. Security Audit
```bash
# Security audit routine
#!/bin/bash

echo "=== Security Audit ==="

# Check for failed login attempts
echo "Failed Login Attempts:"
sudo grep "Failed password" /var/log/auth.log | wc -l

# Check firewall status
echo "Firewall Status:"
sudo ufw status

# Check SSL certificate expiry
echo "SSL Certificate Status:"
openssl x509 -in /etc/ssl/certs/options-engine.pem -text -noout | grep "Not After"

# Check file permissions
echo "File Permissions:"
find /opt/options_engine -name "*.py" -not -perm 644 -ls
find /etc/options_engine -name "*.yaml" -not -perm 600 -ls

# Check for suspicious activity
echo "Suspicious Activity:"
sudo grep -i "suspicious\|attack\|intrusion" /var/log/syslog | tail -10
```

## Monthly Operations

### 1. Capacity Planning
```bash
# Monthly capacity planning
#!/bin/bash

echo "=== Monthly Capacity Planning ==="

# Storage usage trends
echo "Storage Usage Trends:"
du -sh /var/log/options_engine/
du -sh /var/lib/options_engine/
du -sh /var/backups/options_engine/

# Database size analysis
echo "Database Size Analysis:"
psql -d options_engine -c "
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"

# Performance baseline update
echo "Performance Baseline Update:"
python3 scripts/update_performance_baselines.py

# Resource utilization report
echo "Resource Utilization Report:"
python3 scripts/resource_utilization_report.py --days 30
```

### 2. Backup Management
```bash
# Monthly backup management
#!/bin/bash

echo "=== Backup Management ==="

# Create full backup
echo "Creating Full Backup:"
pg_dump options_engine > /var/backups/options_engine/full_backup_$(date +%Y%m%d).sql

# Backup configuration
echo "Backing up Configuration:"
tar -czf /var/backups/options_engine/config_backup_$(date +%Y%m%d).tar.gz /etc/options_engine/

# Cleanup old backups
echo "Cleaning up Old Backups:"
find /var/backups/options_engine/ -name "*.sql" -mtime +30 -delete
find /var/backups/options_engine/ -name "*.tar.gz" -mtime +30 -delete

# Backup verification
echo "Backup Verification:"
pg_restore --list /var/backups/options_engine/full_backup_$(date +%Y%m%d).sql | head -20
```

## Incident Response

### 1. Critical System Failure
```bash
# Emergency response procedure
#!/bin/bash

echo "=== EMERGENCY RESPONSE ==="
echo "Incident: Critical System Failure"
echo "Time: $(date)"

# Immediate actions
echo "1. Stopping affected services:"
sudo systemctl stop options-engine

echo "2. Checking system resources:"
free -h
df -h
ps aux | grep options | head -10

echo "3. Checking recent logs:"
tail -50 /var/log/options_engine/error.log

echo "4. Database status:"
sudo systemctl status postgresql

echo "5. Initiating recovery..."
# Recovery steps would go here
```

### 2. Performance Degradation
```bash
# Performance degradation response
#!/bin/bash

echo "=== Performance Degradation Response ==="

# Identify resource bottlenecks
echo "1. Resource Analysis:"
top -bn1 | head -20
iostat -x 1 3

# Check database performance
echo "2. Database Performance:"
psql -d options_engine -c "
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
"

# Check API response times
echo "3. API Response Times:"
curl -s http://localhost:8080/metrics | jq '.api_response_times'

# Optimize if needed
echo "4. Optimization Actions:"
redis-cli FLUSHEXPIRED
sudo systemctl restart options-engine
```

### 3. Data Quality Issues
```bash
# Data quality issue response
#!/bin/bash

echo "=== Data Quality Issue Response ==="

# Validate data integrity
echo "1. Data Integrity Check:"
python3 scripts/validate_data_integrity.py

# Check API data sources
echo "2. API Data Validation:"
python3 scripts/validate_api_data.py

# Analyze data discrepancies
echo "3. Data Discrepancy Analysis:"
python3 scripts/analyze_data_discrepancies.py

# Implement data correction
echo "4. Data Correction:"
# Correction procedures would go here
```

## Alerting and Notifications

### 1. Alert Configuration
```yaml
# Alert configuration in monitoring.yaml
alerts:
  system:
    cpu_usage:
      threshold: 80
      duration: 5m
      severity: warning
    memory_usage:
      threshold: 90
      duration: 5m
      severity: critical
    disk_usage:
      threshold: 85
      duration: 5m
      severity: warning
    
  application:
    error_rate:
      threshold: 5
      duration: 1m
      severity: warning
    response_time:
      threshold: 10
      duration: 1m
      severity: warning
    
  database:
    connection_errors:
      threshold: 5
      duration: 1m
      severity: critical
    query_time:
      threshold: 30
      duration: 1m
      severity: warning
```

### 2. Notification Channels
```bash
# Email notification script
#!/bin/bash

SUBJECT="Options Engine Alert: $1"
BODY="$2"
TO="ops@yourcompany.com"

echo "$BODY" | mail -s "$SUBJECT" "$TO"

# Slack notification
curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$SUBJECT\n$BODY\"}" \
    $SLACK_WEBHOOK_URL
```

## Maintenance Windows

### 1. Scheduled Maintenance
```bash
# Scheduled maintenance procedure
#!/bin/bash

echo "=== Scheduled Maintenance Window ==="
echo "Start Time: $(date)"

# Pre-maintenance checks
echo "1. Pre-maintenance System Check:"
sudo systemctl status options-engine
curl -s http://localhost:8080/health

# Create maintenance backup
echo "2. Creating Maintenance Backup:"
pg_dump options_engine > /var/backups/options_engine/maintenance_backup_$(date +%Y%m%d_%H%M%S).sql

# Stop services
echo "3. Stopping Services:"
sudo systemctl stop options-engine

# Perform maintenance tasks
echo "4. Performing Maintenance:"
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Database maintenance
psql -d options_engine -c "VACUUM FULL; ANALYZE; REINDEX DATABASE options_engine;"

# Log cleanup
sudo logrotate -f /etc/logrotate.d/options-engine

# Clear cache
redis-cli FLUSHALL

# Restart services
echo "5. Restarting Services:"
sudo systemctl start options-engine

# Post-maintenance verification
echo "6. Post-maintenance Verification:"
sleep 30
sudo systemctl status options-engine
curl -s http://localhost:8080/health

echo "Maintenance Complete: $(date)"
```

### 2. Emergency Maintenance
```bash
# Emergency maintenance procedure
#!/bin/bash

echo "=== EMERGENCY MAINTENANCE ==="
echo "Emergency Type: $1"
echo "Start Time: $(date)"

# Immediate service stop
sudo systemctl stop options-engine

# Emergency backup
pg_dump options_engine > /var/backups/options_engine/emergency_backup_$(date +%Y%m%d_%H%M%S).sql

# Apply emergency fix
case "$1" in
    "security")
        # Apply security patches
        sudo apt-get update && sudo apt-get install -y --only-upgrade security-updates
        ;;
    "database")
        # Database emergency procedures
        psql -d options_engine -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='options_engine';"
        ;;
    "application")
        # Application emergency procedures
        killall -9 python3
        ;;
esac

# Restart services
sudo systemctl start options-engine

# Verify recovery
sleep 30
sudo systemctl status options-engine
curl -s http://localhost:8080/health

echo "Emergency Maintenance Complete: $(date)"
```

## Performance Tuning

### 1. Database Tuning
```sql
-- Database performance tuning
-- Connection settings
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';

-- Query optimization
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Checkpoint settings
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET max_wal_size = '1GB';

-- Apply changes
SELECT pg_reload_conf();
```

### 2. Application Tuning
```python
# Application performance tuning
# Thread pool optimization
THREAD_POOL_SIZE = min(32, (os.cpu_count() or 1) + 4)

# Connection pool settings
DATABASE_POOL_SIZE = 20
DATABASE_POOL_OVERFLOW = 30

# Cache settings
CACHE_DEFAULT_TIMEOUT = 300
CACHE_THRESHOLD = 500

# API settings
API_TIMEOUT = 30
API_RETRY_COUNT = 3
API_BATCH_SIZE = 100
```

### 3. System Tuning
```bash
# System-level tuning
echo "# Options Engine Tuning" >> /etc/sysctl.conf
echo "net.core.somaxconn = 1024" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 1024" >> /etc/sysctl.conf
echo "vm.swappiness = 10" >> /etc/sysctl.conf
echo "vm.dirty_ratio = 15" >> /etc/sysctl.conf
echo "vm.dirty_background_ratio = 5" >> /etc/sysctl.conf

# Apply settings
sudo sysctl -p
```

## Monitoring and Metrics

### 1. Key Performance Indicators
```bash
# KPI monitoring dashboard
#!/bin/bash

echo "=== Key Performance Indicators ==="

# System KPIs
echo "System KPIs:"
echo "  CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')%"
echo "  Memory Usage: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')%"
echo "  Disk Usage: $(df / | tail -1 | awk '{print $5}')%"

# Application KPIs
echo "Application KPIs:"
curl -s http://localhost:8080/metrics | jq -r '
  "  Average Response Time: \(.avg_response_time)ms",
  "  Error Rate: \(.error_rate)%",
  "  Cache Hit Rate: \(.cache_hit_rate)%",
  "  Active Connections: \(.active_connections)"
'

# Business KPIs
echo "Business KPIs:"
python3 scripts/business_metrics.py
```

### 2. Custom Metrics
```python
# Custom metrics collection
import psutil
import time
from dataclasses import dataclass

@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: dict
    process_count: int
    timestamp: float

def collect_system_metrics():
    return SystemMetrics(
        cpu_usage=psutil.cpu_percent(interval=1),
        memory_usage=psutil.virtual_memory().percent,
        disk_usage=psutil.disk_usage('/').percent,
        network_io=psutil.net_io_counters()._asdict(),
        process_count=len(psutil.pids()),
        timestamp=time.time()
    )

# Usage
metrics = collect_system_metrics()
print(f"System metrics: {metrics}")
```

## Disaster Recovery

### 1. Backup Strategy
```bash
# Comprehensive backup strategy
#!/bin/bash

BACKUP_DIR="/var/backups/options_engine"
DATE=$(date +%Y%m%d_%H%M%S)

# Database backup
pg_dump options_engine > $BACKUP_DIR/db_backup_$DATE.sql

# Configuration backup
tar -czf $BACKUP_DIR/config_backup_$DATE.tar.gz /etc/options_engine/

# Application backup
tar -czf $BACKUP_DIR/app_backup_$DATE.tar.gz /opt/options_engine/

# Log backup
tar -czf $BACKUP_DIR/log_backup_$DATE.tar.gz /var/log/options_engine/

# Verify backups
echo "Backup verification:"
ls -la $BACKUP_DIR/*_$DATE.*
```

### 2. Recovery Procedures
```bash
# System recovery procedure
#!/bin/bash

echo "=== System Recovery Procedure ==="
echo "Recovery Type: $1"
echo "Backup Date: $2"

BACKUP_DIR="/var/backups/options_engine"

case "$1" in
    "full")
        # Full system recovery
        sudo systemctl stop options-engine
        
        # Restore database
        dropdb options_engine
        createdb options_engine
        psql options_engine < $BACKUP_DIR/db_backup_$2.sql
        
        # Restore configuration
        sudo rm -rf /etc/options_engine/
        sudo tar -xzf $BACKUP_DIR/config_backup_$2.tar.gz -C /
        
        # Restore application
        sudo rm -rf /opt/options_engine/
        sudo tar -xzf $BACKUP_DIR/app_backup_$2.tar.gz -C /
        
        sudo systemctl start options-engine
        ;;
    "database")
        # Database recovery only
        sudo systemctl stop options-engine
        psql options_engine < $BACKUP_DIR/db_backup_$2.sql
        sudo systemctl start options-engine
        ;;
    "config")
        # Configuration recovery only
        sudo systemctl stop options-engine
        sudo tar -xzf $BACKUP_DIR/config_backup_$2.tar.gz -C /
        sudo systemctl start options-engine
        ;;
esac

echo "Recovery Complete"
```

## Documentation and Runbooks

### 1. Operational Runbooks
- **Service Restart**: Standard procedure for service restart
- **Database Maintenance**: Monthly database maintenance tasks
- **Performance Tuning**: Steps for performance optimization
- **Security Incident**: Response procedures for security incidents
- **Backup and Recovery**: Comprehensive backup and recovery procedures

### 2. Emergency Contacts
- **Primary On-Call**: +1-XXX-XXX-XXXX
- **Secondary On-Call**: +1-XXX-XXX-XXXX
- **Database Admin**: +1-XXX-XXX-XXXX
- **Security Team**: +1-XXX-XXX-XXXX
- **Management**: +1-XXX-XXX-XXXX

### 3. Escalation Procedures
1. **Level 1**: Operations team handles routine issues
2. **Level 2**: Development team for application issues
3. **Level 3**: Senior engineers for complex problems
4. **Level 4**: Management for business-critical issues