#!/bin/bash
#
# CropTap ML API - Deployment Script for Hostinger VPS
# This script automates the deployment process
#
# Usage: 
#   chmod +x deploy.sh
#   ./deploy.sh
#

set -e  # Exit on error

echo "=========================================="
echo "CropTap ML API Deployment Script"
echo "=========================================="
echo ""

# Configuration
APP_DIR="/opt/croptap-api"
APP_USER="croptap"
VENV_PATH="$APP_DIR/venv"
SERVICE_NAME="croptap-api"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}➜ $1${NC}"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root (use sudo)"
   exit 1
fi

# Step 1: Update system
print_info "Updating system packages..."
apt update && apt upgrade -y
print_success "System updated"

# Step 2: Install dependencies
print_info "Installing system dependencies..."
apt install -y python3.11 python3.11-venv python3-pip nginx git curl htop ufw
print_success "Dependencies installed"

# Step 3: Create application user if doesn't exist
if id "$APP_USER" &>/dev/null; then
    print_info "User $APP_USER already exists"
else
    print_info "Creating application user..."
    adduser --system --group --home $APP_DIR $APP_USER
    print_success "User created"
fi

# Step 4: Create application directory
print_info "Setting up application directory..."
mkdir -p $APP_DIR
mkdir -p /var/log/croptap-api
chown -R $APP_USER:$APP_USER $APP_DIR
chown -R $APP_USER:$APP_USER /var/log/croptap-api
print_success "Directory structure created"

# Step 5: Setup Python virtual environment
print_info "Creating Python virtual environment..."
sudo -u $APP_USER python3.11 -m venv $VENV_PATH
print_success "Virtual environment created"

# Step 6: Install Python dependencies
print_info "Installing Python packages..."
cd $APP_DIR
sudo -u $APP_USER $VENV_PATH/bin/pip install --upgrade pip
sudo -u $APP_USER $VENV_PATH/bin/pip install -r requirements.txt
sudo -u $APP_USER $VENV_PATH/bin/pip install gunicorn
print_success "Python packages installed"

# Step 7: Copy systemd service file
print_info "Installing systemd service..."
cp deployment/croptap-api.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable $SERVICE_NAME
print_success "Systemd service installed"

# Step 8: Configure Nginx
print_info "Configuring Nginx..."
cp deployment/nginx-croptap-api.conf /etc/nginx/sites-available/croptap-api

# Remove default site if exists
if [ -f /etc/nginx/sites-enabled/default ]; then
    rm /etc/nginx/sites-enabled/default
fi

# Enable site
ln -sf /etc/nginx/sites-available/croptap-api /etc/nginx/sites-enabled/
nginx -t
print_success "Nginx configured"

# Step 9: Configure firewall
print_info "Configuring firewall..."
ufw allow 'OpenSSH'
ufw allow 'Nginx Full'
echo "y" | ufw enable
print_success "Firewall configured"

# Step 10: Start services
print_info "Starting services..."
systemctl restart nginx
systemctl start $SERVICE_NAME
print_success "Services started"

# Step 11: Check service status
print_info "Checking service status..."
sleep 2
if systemctl is-active --quiet $SERVICE_NAME; then
    print_success "API service is running"
else
    print_error "API service failed to start. Check logs: journalctl -u $SERVICE_NAME -n 50"
    exit 1
fi

if systemctl is-active --quiet nginx; then
    print_success "Nginx is running"
else
    print_error "Nginx failed to start. Check logs: journalctl -u nginx -n 50"
    exit 1
fi

# Final message
echo ""
echo "=========================================="
echo -e "${GREEN}Deployment completed successfully!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update your domain in: /etc/nginx/sites-available/croptap-api"
echo "2. Run: sudo systemctl restart nginx"
echo "3. Setup SSL: sudo certbot --nginx -d your-domain.com"
echo "4. Test API: curl http://localhost:8000/health"
echo ""
echo "Service management:"
echo "  - View logs: sudo journalctl -u $SERVICE_NAME -f"
echo "  - Restart: sudo systemctl restart $SERVICE_NAME"
echo "  - Status: sudo systemctl status $SERVICE_NAME"
echo ""
