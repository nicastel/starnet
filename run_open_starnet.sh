#!/bin/bash
echo $@ > args.txt

# Loop through arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    -v|--version)
      echo "starnet1  version: 1.0.0"
      exit 0
      ;;
  esac
done

python3.12 open_starnet.py $@
