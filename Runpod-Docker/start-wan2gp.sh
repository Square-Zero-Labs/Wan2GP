#!/bin/bash
set -e

echo "=== Wan2GP Container Startup ==="

# Restore application files if needed (handles volume mount scenario)
if [ ! -f "/workspace/Wan2GP/wgp.py" ]; then
    echo "Restoring application files..."
    mkdir -p /workspace/Wan2GP
    rsync -a /opt/wan2gp_source/ /workspace/Wan2GP/
    echo "Application files restored"
else
    echo "Application files already present"
fi

# Set up nginx authentication using RunPod's existing infrastructure
echo "Setting up authentication..."
PASSWORD=${WAN2GP_PASSWORD:-"gpuPoor2025"}
USERNAME=${WAN2GP_USERNAME:-"admin"}

# Verify required tools are available
if ! command -v htpasswd &> /dev/null; then
    echo "❌ ERROR: htpasswd command not found (apache2-utils not installed)"
    exit 1
fi

if ! command -v nginx &> /dev/null; then
    echo "❌ ERROR: nginx command not found"
    echo "RunPod base image may have changed - nginx CLI tools not available"
    exit 1
fi

# Create password file
htpasswd -cb /etc/nginx/.htpasswd "$USERNAME" "$PASSWORD"

# Create our own reliable nginx authentication proxy
echo "Setting up authenticated nginx proxy..."
cat > /etc/nginx/conf.d/wan2gp-auth.conf << 'EOF'
# Wan2GP Authentication Proxy
server {
    listen 7862;
    
    location / {
        auth_basic "Wan2GP Access Required";
        auth_basic_user_file /etc/nginx/.htpasswd;
        
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Gradio
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

# Ensure the main nginx config includes our proxy config from the conf.d directory
if ! grep -q "include /etc/nginx/conf.d/.*.conf;" /etc/nginx/nginx.conf; then
    echo "Adding 'include' for conf.d to nginx.conf..."
    sed -i '/http {/a \    include /etc/nginx/conf.d/*.conf;' /etc/nginx/nginx.conf
fi

# Test nginx config
if ! nginx -t 2>/dev/null; then
    echo "❌ ERROR: nginx configuration test failed"
    echo "Unable to create authentication proxy"
    exit 1
fi


echo "✅ Authentication proxy configured - Username: $USERNAME, Password: $PASSWORD"
echo "🌐 Access via port 7862 with authentication"
WAN2GP_ACCESS_PORT=7862

# Start our application in the background
echo "Starting Wan2GP application in background..."
cd /workspace/Wan2GP

# Use our own nginx proxy: 7862 (nginx with auth) → 7860 (gradio)
SERVER_NAME="127.0.0.1"
SERVER_PORT="7860"
echo "Using our nginx proxy: nginx on 7862 → gradio on 7860"

echo "Starting Wan2GP on $SERVER_NAME:$SERVER_PORT"
nohup python3 wgp.py --server-name $SERVER_NAME --server-port $SERVER_PORT > /workspace/wan2gp.log 2>&1 &
echo "Wan2GP started on internal port $SERVER_PORT, logs in /workspace/wan2gp.log"
echo ""
echo "🔐 AUTHENTICATION REQUIRED:"
echo "   Username: $USERNAME"
echo "   Password: $PASSWORD"
echo ""

echo "Starting RunPod services..."
if [ -f "/start.sh" ]; then
    /start.sh
else
    echo "No /start.sh found, keeping container alive by monitoring log for debugging."
    tail -f /workspace/wan2gp.log
fi 