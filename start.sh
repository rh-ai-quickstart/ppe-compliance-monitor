#!/bin/bash

# Debug: Print current working directory and list its contents
echo "Current working directory:"
pwd
echo "Contents of current directory:"
ls -la

# Debug: List contents of /app/backend
echo "Contents of /app/backend:"
ls -la /app/backend

# Debug: Check if ppe.pt exists
if [ -f "/app/models/ppe.pt" ]; then
    echo "ppe.pt file exists"
else
    echo "ppe.pt file does not exist"
fi

# Debug: List contents of /app/frontend/build
echo "Contents of /app/frontend/build:"
ls -la /app/frontend/build

# Write runtime frontend config if provided
if [ -n "${FRONTEND_API_URL}" ]; then
  printf "window.__ENV__ = { API_URL: \"%s\" };\n" "${FRONTEND_API_URL}" > /app/frontend/build/env.js
fi

# Start the Python backend
python /app/backend/app.py &

# Start the frontend server
serve -s /app/frontend/build -l tcp://0.0.0.0:3000 &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?