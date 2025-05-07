#\!/bin/bash

# Set terminal colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Clear screen
clear

# Display the ASCII logo
echo -e "${GREEN} /\$\$     /\$\$ /\$\$\$\$\$\$  /\$\$        /\$\$\$\$\$\$  "
echo -e "|  \$\$   /\$\$//\$\$__  \$\$| \$\$       /\$\$__  \$\$| "
echo -e " \\  \$\$ /\$\$/| \$\$  \\ \$\$| \$\$      | \$\$  \\ \$\$| __ _ "
echo -e "  \\  \$\$\$\$/ | \$\$  | \$\$| \$\$      | \$\$  | \$\$|/ _| |"
echo -e "   \\  \$\$/  | \$\$  | \$\$| \$\$      | \$\$  | \$\$| |_| | ___ _      __"
echo -e "    | \$\$   | \$\$  | \$\$| \$\$      | \$\$  | \$\$|  _| |/ _ \\ \\ /\\ / /"
echo -e "    | \$\$   |  \$\$\$\$\$\$/| \$\$\$\$\$\$\$\$|  \$\$\$\$\$\$/| | | | (_) \\ V  V /"
echo -e "    |__/    \\______/ |________/ \\______/ |_| |_|\\___/ \\_/\\_/"
echo -e ""
echo -e "                    Version 0.5 - Object Detection Workflow Tool"
echo -e "                    ---------------------------------------------"
echo -e ""

# Check if Python is available
if \! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python not found\!${NC}"
    echo "Please install Python 3.11 or later."
    exit 1
fi

# Check if requirements are installed
echo "Checking dependencies..."
if \! python3 -c "import PyQt5, cv2, numpy" &> /dev/null; then
    echo "Installing required packages..."
    if \! pip3 install -r requirements.txt; then
        echo -e "${RED}ERROR: Failed to install dependencies.${NC}"
        echo "Please run manually: pip3 install -r requirements.txt"
        exit 1
    fi
fi

# Create data directory if it doesn't exist
mkdir -p data
mkdir -p runs

# Download default model if needed
if [ \! -f "models/yolo11n.pt" ] && [ \! -f "models/yolo11s.pt" ]; then
    echo "Downloading default model..."
    python3 download_models.py
fi

# Launch the application
echo "Launching YOLOflow..."
python3 main.py

# If the application exits with an error
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: YOLOflow exited with an error.${NC}"
    read -p "Press Enter to exit..."
fi
