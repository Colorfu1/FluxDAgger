#!/bin/bash

# Configure and rename CAN interfaces for the four-arm FluxDAgger setup.
#
# Usage:
#   sudo ./tools/hardware/can_config_modified.sh
#
# Requirements:
#   sudo apt install ethtool can-utils
#   gs_usb driver available on the host.
#
# Notes:
#   For multi-CAN setups, USB_PORTS maps ethtool bus-info values to
#   "<target interface>:<bitrate>". If USB-CAN devices are moved to different
#   ports, inspect the bus-info values with:
#
#       sudo ethtool -i can0 | grep bus-info
#
#   Then update USB_PORTS below.

# Expected CAN module count.
EXPECTED_CAN_COUNT=4

if [ "$EXPECTED_CAN_COUNT" -eq 1 ]; then
    # Default CAN name and bitrate for a single CAN module.
    DEFAULT_CAN_NAME="${1:-can0}"
    DEFAULT_BITRATE="${2:-1000000}"
    USB_ADDRESS="${3}"
fi

# USB bus-info to target interface mapping for multi-CAN setups.
if [ "$EXPECTED_CAN_COUNT" -ne 1 ]; then
    declare -A USB_PORTS
    USB_PORTS["1-13:1.0"]="c_left_slave:1000000"
    USB_PORTS["1-12:1.0"]="c_left_master:1000000"
    USB_PORTS["1-5:1.0"]="c_right_master:1000000"
    USB_PORTS["1-2:1.0"]="c_right_slave:1000000"
fi

CURRENT_CAN_COUNT=$(ip link show type can | grep -c "link/can")

if [ "$CURRENT_CAN_COUNT" -ne "$EXPECTED_CAN_COUNT" ]; then
    echo "Error: detected CAN module count ($CURRENT_CAN_COUNT) does not match expected count ($EXPECTED_CAN_COUNT)."
    exit 1
fi

sudo modprobe gs_usb
if [ $? -ne 0 ]; then
    echo "Error: failed to load gs_usb module."
    exit 1
fi

if [ "$EXPECTED_CAN_COUNT" -eq 1 ]; then
    if [ -n "$USB_ADDRESS" ]; then
        echo "Detected USB bus-info argument: $USB_ADDRESS"

        INTERFACE_NAME=""
        for iface in $(ip -br link show type can | awk '{print $1}'); do
            BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
            if [ "$BUS_INFO" == "$USB_ADDRESS" ]; then
                INTERFACE_NAME="$iface"
                break
            fi
        done

        if [ -z "$INTERFACE_NAME" ]; then
            echo "Error: no CAN interface found for USB bus-info $USB_ADDRESS."
            exit 1
        else
            echo "Found interface for USB bus-info $USB_ADDRESS: $INTERFACE_NAME"
        fi
    else
        INTERFACE_NAME=$(ip -br link show type can | awk '{print $1}')

        if [ -z "$INTERFACE_NAME" ]; then
            echo "Error: no CAN interface detected."
            exit 1
        fi

        echo "Expected one CAN module, detected interface $INTERFACE_NAME"
    fi

    IS_LINK_UP=$(ip link show "$INTERFACE_NAME" | grep -q "UP" && echo "yes" || echo "no")
    CURRENT_BITRATE=$(ip -details link show "$INTERFACE_NAME" | grep -oP 'bitrate \K\d+')

    if [ "$IS_LINK_UP" == "yes" ] && [ "$CURRENT_BITRATE" -eq "$DEFAULT_BITRATE" ]; then
        echo "Interface $INTERFACE_NAME is already up with bitrate $DEFAULT_BITRATE"

        if [ "$INTERFACE_NAME" != "$DEFAULT_CAN_NAME" ]; then
            echo "Renaming interface $INTERFACE_NAME to $DEFAULT_CAN_NAME"
            sudo ip link set "$INTERFACE_NAME" down
            sudo ip link set "$INTERFACE_NAME" name "$DEFAULT_CAN_NAME"
            sudo ip link set "$DEFAULT_CAN_NAME" up
            echo "Interface renamed to $DEFAULT_CAN_NAME and reactivated."
        else
            echo "Interface name is already $DEFAULT_CAN_NAME"
        fi
    else
        if [ "$IS_LINK_UP" == "yes" ]; then
            echo "Interface $INTERFACE_NAME is up, but bitrate $CURRENT_BITRATE does not match $DEFAULT_BITRATE."
        else
            echo "Interface $INTERFACE_NAME is down or has no bitrate configured."
        fi

        sudo ip link set "$INTERFACE_NAME" down
        sudo ip link set "$INTERFACE_NAME" type can bitrate $DEFAULT_BITRATE
        sudo ip link set "$INTERFACE_NAME" up
        echo "Interface $INTERFACE_NAME reset to bitrate $DEFAULT_BITRATE and activated."

        if [ "$INTERFACE_NAME" != "$DEFAULT_CAN_NAME" ]; then
            echo "Renaming interface $INTERFACE_NAME to $DEFAULT_CAN_NAME"
            sudo ip link set "$INTERFACE_NAME" down
            sudo ip link set "$INTERFACE_NAME" name "$DEFAULT_CAN_NAME"
            sudo ip link set "$DEFAULT_CAN_NAME" up
            echo "Interface renamed to $DEFAULT_CAN_NAME and reactivated."
        fi
    fi
else
    PREDEFINED_COUNT=${#USB_PORTS[@]}
    if [ "$EXPECTED_CAN_COUNT" -ne "$PREDEFINED_COUNT" ]; then
        echo "Error: expected CAN count ($EXPECTED_CAN_COUNT) does not match USB_PORTS count ($PREDEFINED_COUNT)."
        exit 1
    fi

    for iface in $(ip -br link show type can | awk '{print $1}'); do
        BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')

        if [ -z "$BUS_INFO" ];then
            echo "Error: failed to read bus-info for interface $iface."
            continue
        fi

        echo "Interface $iface is attached to USB bus-info $BUS_INFO"

        if [ -n "${USB_PORTS[$BUS_INFO]}" ];then
            IFS=':' read -r TARGET_NAME TARGET_BITRATE <<< "${USB_PORTS[$BUS_INFO]}"

            IS_LINK_UP=$(ip link show "$iface" | grep -q "UP" && echo "yes" || echo "no")
            CURRENT_BITRATE=$(ip -details link show "$iface" | grep -oP 'bitrate \K\d+')

            if [ "$IS_LINK_UP" == "yes" ] && [ "$CURRENT_BITRATE" -eq "$TARGET_BITRATE" ]; then
                echo "Interface $iface is already up with bitrate $TARGET_BITRATE"

                if [ "$iface" != "$TARGET_NAME" ]; then
                    echo "Renaming interface $iface to $TARGET_NAME"
                    sudo ip link set "$iface" down
                    sudo ip link set "$iface" name "$TARGET_NAME"
                    sudo ip link set "$TARGET_NAME" up
                    echo "Interface renamed to $TARGET_NAME and reactivated."
                else
                    echo "Interface name is already $TARGET_NAME"
                fi
            else
                if [ "$IS_LINK_UP" == "yes" ]; then
                    echo "Interface $iface is up, but bitrate $CURRENT_BITRATE does not match $TARGET_BITRATE."
                else
                    echo "Interface $iface is down or has no bitrate configured."
                fi

                sudo ip link set "$iface" down
                sudo ip link set "$iface" type can bitrate $TARGET_BITRATE
                sudo ip link set "$iface" up
                echo "Interface $iface reset to bitrate $TARGET_BITRATE and activated."

                if [ "$iface" != "$TARGET_NAME" ]; then
                    echo "Renaming interface $iface to $TARGET_NAME"
                    sudo ip link set "$iface" down
                    sudo ip link set "$iface" name "$TARGET_NAME"
                    sudo ip link set "$TARGET_NAME" up
                    echo "Interface renamed to $TARGET_NAME and reactivated."
                fi
            fi
        else
            echo "Error: unknown USB bus-info $BUS_INFO for interface $iface."
            exit 1
        fi
    done
fi

echo "All CAN interfaces were renamed and activated successfully."
