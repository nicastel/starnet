#!/bin/bash
echo $@ > args.txt

# Loop through arguments
case "$1" in
  -v|--version)
    echo "starnet2  version: 1.0.0"
    exit 0
    ;;
esac

echo "MPS backend"
python3.12 open_starnet.py $@ 2> error.log
exit 0
