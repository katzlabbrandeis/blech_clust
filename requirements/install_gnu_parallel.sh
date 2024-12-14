#!/bin/bash
if [ -n "$1" ]; then
    echo "$1" | sudo -S apt install parallel
else
    sudo apt install parallel
fi
