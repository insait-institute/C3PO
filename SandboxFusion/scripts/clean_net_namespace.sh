#!/bin/bash
# =============================================================================
# clean_net_namespace.sh -- Tear down a Linux network namespace
# =============================================================================
# Usage: sudo bash clean_net_namespace.sh <namespace_name> <subnet> [--no-bridge]
#
# Reverses the setup performed by create_net_namespace.sh. Removes the NAT
# iptables rule (unless --no-bridge was used) and deletes the namespace.
# Deleting the namespace automatically destroys the associated veth pair.
# =============================================================================
set -e

usage() {
    echo "Usage: $0 <namespace_name> <subnet> [--no-bridge]"
    exit 1
}

# --- Argument validation ---
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    usage
fi

NAMESPACE=$1
SUBNET=$2
CLEAN_BRIDGE=true

if [ "$#" -eq 3 ]; then
    if [ "$3" == "--no-bridge" ]; then
        CLEAN_BRIDGE=false
    else
        usage
    fi
fi

# Cleanup commands must not abort the script on failure — each resource
# should be cleaned independently so one failure doesn't orphan the rest.
set +e

if $CLEAN_BRIDGE; then
    # --- Remove the NAT masquerade rule that was added during namespace creation ---
    EXTERNAL_IFACE=$(ip route | grep default | awk '{print $5}')
    if [ -n "$EXTERNAL_IFACE" ]; then
        sudo iptables -w -t nat -D POSTROUTING -s ${SUBNET}.0/24 -o $EXTERNAL_IFACE -j MASQUERADE 2>/dev/null
    fi
fi

# --- Delete the namespace (also destroys the associated veth pair) ---
sudo ip netns delete $NAMESPACE 2>/dev/null

echo "Namespace $NAMESPACE cleaned up"