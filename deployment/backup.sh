#!/bin/bash
#
# CropTap ML API - Backup Script
# This script backs up critical files and data
#
# Usage: 
#   chmod +x backup.sh
#   ./backup.sh
#
# Setup cron job:
#   sudo crontab -e
#   Add: 0 2 * * * /opt/croptap-api/deployment/backup.sh
#

set -e

# Configuration
BACKUP_DIR="/backups/croptap-api"
APP_DIR="/opt/croptap-api"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/croptap-api-$DATE.tar.gz"
RETENTION_DAYS=7

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

echo "Starting backup at $(date)"

# Create backup
tar -czf $BACKUP_FILE \
    $APP_DIR/models/*.pkl \
    $APP_DIR/models/*.json \
    $APP_DIR/raw_datasets/*.csv \
    $APP_DIR/.env \
    $APP_DIR/gunicorn_config.py \
    2>/dev/null || true

# Check if backup was created
if [ -f "$BACKUP_FILE" ]; then
    BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo "Backup created successfully: $BACKUP_FILE ($BACKUP_SIZE)"
else
    echo "ERROR: Backup failed!"
    exit 1
fi

# Clean old backups (keep only last N days)
echo "Cleaning old backups (keeping last $RETENTION_DAYS days)..."
find $BACKUP_DIR -name "croptap-api-*.tar.gz" -mtime +$RETENTION_DAYS -delete

# List current backups
echo "Current backups:"
ls -lh $BACKUP_DIR/croptap-api-*.tar.gz 2>/dev/null || echo "No backups found"

echo "Backup completed at $(date)"
