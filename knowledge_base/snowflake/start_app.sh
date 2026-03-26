#!/bin/bash
cd /home/azureuser/datapilotv2
sudo pkill -f main.py
sudo pkill -f uvicorn
sudo nohup python3 main.py > server.log 2>&1 &
echo "Started main.py"
