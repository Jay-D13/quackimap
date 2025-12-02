#!/bin/bash
# Launch VNC with custom RViz configuration for quackimap project

ROBOT_NAME="${1:-vquarck}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="${SCRIPT_DIR}/assets/vnc"

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping VNC container..."
    if [ ! -z "$CONTAINER" ]; then
        docker stop "$CONTAINER" > /dev/null 2>&1
    fi
    exit 0
}

# Set up trap to catch Ctrl+C
trap cleanup SIGINT SIGTERM

echo "Launching VNC for ${ROBOT_NAME}..."

# Run in detached mode to avoid TTY issues
dts gui --vnc "$ROBOT_NAME" > /dev/null 2>&1 &

echo "Waiting for VNC container to start..."
sleep 5

# Find the container
CONTAINER=$(docker ps --filter "ancestor=duckietown/dt-gui-tools:ente-amd64" -q | head -1)

if [ -z "$CONTAINER" ]; then
    echo "Error: Could not find VNC container"
    exit 1
fi

echo "Found container: $CONTAINER"

# Copy custom files
echo "Copying custom RViz configs..."
docker cp "${ASSETS_DIR}/opt/ros/noetic/share/rviz/." "$CONTAINER:/opt/ros/noetic/share/rviz/"

echo "Copying desktop shortcuts..."
docker cp "${ASSETS_DIR}/root/." "$CONTAINER:/root/"

echo "Copying launch scripts..."
docker cp "${ASSETS_DIR}/usr/bin/." "$CONTAINER:/usr/bin/"
docker exec "$CONTAINER" chmod +x /usr/bin/launch-odometry.sh
docker exec "$CONTAINER" chmod +x /usr/bin/launch-slam-rviz.sh

echo ""
echo "Done! Custom RViz configs are now available."
echo ""
echo "Open http://localhost:8087 in your browser."
echo "Press Ctrl+C to stop the VNC."

# Keep the script running and monitor the container
while docker ps -q --filter id="$CONTAINER" | grep -q .; do
    sleep 1
done

echo "VNC container has stopped."