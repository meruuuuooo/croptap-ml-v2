# Deployment Files for Hostinger VPS

This directory contains all the configuration files and scripts needed to deploy the CropTap ML API to a Hostinger VPS.

## Files Overview

| File | Description | Destination |
|------|-------------|-------------|
| `deploy.sh` | Automated deployment script | Run on VPS |
| `croptap-api.service` | Systemd service configuration | `/etc/systemd/system/` |
| `nginx-croptap-api.conf` | Nginx reverse proxy config | `/etc/nginx/sites-available/` |
| `backup.sh` | Automated backup script | `/opt/croptap-api/deployment/` |

## Quick Start Deployment

### 1. Upload Files to VPS

```bash
# From your local machine
scp -r machine_learning_api root@your-vps-ip:/opt/croptap-api/
```

### 2. Run Deployment Script

```bash
# On VPS
cd /opt/croptap-api
chmod +x deployment/deploy.sh
sudo ./deployment/deploy.sh
```

### 3. Configure Domain

Edit Nginx configuration and replace `your-domain.com`:

```bash
sudo nano /etc/nginx/sites-available/croptap-api
# Update: server_name your-domain.com www.your-domain.com;
sudo systemctl restart nginx
```

### 4. Setup SSL Certificate

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

### 5. Verify Deployment

```bash
# Check service status
sudo systemctl status croptap-api

# Test API
curl http://localhost:8000/health
curl https://your-domain.com/health

# View logs
sudo journalctl -u croptap-api -f
```

## Manual Deployment

If you prefer manual deployment, follow the complete deployment plan in the project documentation.

## Configuration Files Explained

### croptap-api.service

Systemd service file that manages the API process:
- Automatically starts on boot
- Restarts on failure
- Manages environment variables
- Handles graceful shutdowns

### nginx-croptap-api.conf

Nginx configuration that:
- Acts as reverse proxy to Gunicorn
- Handles SSL termination
- Enables gzip compression
- Sets security headers
- Manages timeouts and buffering

### gunicorn_config.py (in project root)

Gunicorn configuration that:
- Sets optimal worker count based on CPU cores
- Configures Uvicorn workers for async support
- Manages logging
- Sets timeouts and keep-alive

## Post-Deployment Tasks

### Setup Automated Backups

```bash
# Make backup script executable
chmod +x /opt/croptap-api/deployment/backup.sh

# Add to crontab (runs daily at 2 AM)
sudo crontab -e
# Add line: 0 2 * * * /opt/croptap-api/deployment/backup.sh
```

### Setup Log Rotation

```bash
sudo nano /etc/logrotate.d/croptap-api
```

Add:
```
/var/log/croptap-api/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 croptap croptap
    sharedscripts
    postrotate
        systemctl reload croptap-api > /dev/null
    endscript
}
```

## Common Commands

### Service Management

```bash
# Start service
sudo systemctl start croptap-api

# Stop service
sudo systemctl stop croptap-api

# Restart service
sudo systemctl restart croptap-api

# View status
sudo systemctl status croptap-api

# View logs
sudo journalctl -u croptap-api -f
```

### Nginx Management

```bash
# Test configuration
sudo nginx -t

# Reload configuration
sudo systemctl reload nginx

# Restart Nginx
sudo systemctl restart nginx

# View logs
sudo tail -f /var/log/nginx/croptap-api-error.log
```

### Application Updates

```bash
# Pull latest code
cd /opt/croptap-api
git pull

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# Restart service
sudo systemctl restart croptap-api
```

## Troubleshooting

### Service won't start

```bash
# Check detailed logs
sudo journalctl -u croptap-api -n 100 --no-pager

# Check file permissions
ls -la /opt/croptap-api
sudo chown -R croptap:croptap /opt/croptap-api
```

### 502 Bad Gateway

```bash
# Verify Gunicorn is running
sudo systemctl status croptap-api

# Check port binding
sudo netstat -tulpn | grep 8000

# Check Nginx error logs
sudo tail -f /var/log/nginx/croptap-api-error.log
```

### SSL Certificate Issues

```bash
# Renew certificate manually
sudo certbot renew --dry-run
sudo certbot renew

# Check certificate status
sudo certbot certificates
```

## Security Checklist

- [ ] Changed default SSH port
- [ ] Disabled root SSH login
- [ ] Setup SSH key authentication
- [ ] Configured UFW firewall
- [ ] Installed fail2ban
- [ ] Setup SSL certificate
- [ ] Configured automated backups
- [ ] Setup log rotation
- [ ] Created non-root application user

## Resource Monitoring

```bash
# Check disk usage
df -h

# Check memory usage
free -h

# Check process resources
htop

# Check application logs size
du -sh /var/log/croptap-api/
```

## Support

For issues or questions, refer to the main deployment plan documentation.
