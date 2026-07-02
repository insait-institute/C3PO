#!/bin/bash
# =============================================================================
# create_net_namespace.sh -- Create a Linux network namespace for lite sandbox isolation
# =============================================================================
# Usage: sudo bash create_net_namespace.sh <namespace_name> <subnet> [--no-bridge]
#
# Creates a Linux network namespace with optional veth-pair bridging to the
# host. This provides lightweight network isolation for sandboxed processes:
#   - The host side gets <subnet>.1, the namespace gets <subnet>.2 (/24).
#   - A default route inside the namespace points back to the host.
#   - IP forwarding and NAT masquerading are enabled so the namespace can
#     reach external networks through the host.
#
# Veth interface names are derived from the first 6 characters of the
# namespace name (e.g. ve-a1b2c3 / ve-a1b2c3-br) to stay within Linux's
# 15-character interface name limit.
#
# Pass --no-bridge to create the namespace with only a loopback interface
# (fully network-isolated, no external connectivity).
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
CREATE_BRIDGE=true
# Linux interface names are limited to 15 characters.
# Use last 6 chars of the namespace name as the veth suffix to maximise
# randomness (the namespace name may carry a fixed prefix like "sbox_").
VETH_ID=${NAMESPACE: -6}
VETH_NS="ve-${VETH_ID}"
VETH_BR="ve-${VETH_ID}-br"

if [ "$#" -eq 3 ]; then
    if [ "$3" == "--no-bridge" ]; then
        CREATE_BRIDGE=false
    else
        usage
    fi
fi

# IP addresses: .1 for the host end, .2 for the namespace end
HOST_IP="${SUBNET}.1"
NS_IP="${SUBNET}.2"

# On any failure, clean up whatever was partially created so we don't leak
# namespaces, veth pairs, or iptables rules.
cleanup_on_failure() {
    set +e
    if $CREATE_BRIDGE; then
        EXTERNAL_IFACE=$(ip route | grep default | awk '{print $5}')
        [ -n "$EXTERNAL_IFACE" ] && \
            sudo iptables -w -t nat -D POSTROUTING -s ${SUBNET}.0/24 -o $EXTERNAL_IFACE -j MASQUERADE 2>/dev/null
        sudo ip link delete $VETH_BR 2>/dev/null
    fi
    sudo ip netns delete $NAMESPACE 2>/dev/null
}
trap cleanup_on_failure ERR

# --- Create the network namespace ---
sudo ip netns add $NAMESPACE

if $CREATE_BRIDGE; then
    sudo ip link add $VETH_NS type veth peer name $VETH_BR
    sudo ip link set $VETH_NS netns $NAMESPACE

    sudo ip addr add $HOST_IP/24 dev $VETH_BR
    sudo ip link set $VETH_BR up

    sudo ip netns exec $NAMESPACE ip addr add $NS_IP/24 dev $VETH_NS
    sudo ip netns exec $NAMESPACE ip link set $VETH_NS up
fi

# --- Bring up the loopback interface inside the namespace ---
sudo ip netns exec $NAMESPACE ip link set lo up

if $CREATE_BRIDGE; then
    sudo ip netns exec $NAMESPACE ip route add default via $HOST_IP
    sudo sysctl -w net.ipv4.ip_forward=1

    EXTERNAL_IFACE=$(ip route | grep default | awk '{print $5}')
    sudo iptables -w -t nat -A POSTROUTING -s ${SUBNET}.0/24 -o $EXTERNAL_IFACE -j MASQUERADE
fi

trap - ERR
echo "Namespace $NAMESPACE created with subnet ${SUBNET}.0/24"