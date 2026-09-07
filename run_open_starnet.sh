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

export TF_USE_LEGACY_KERAS=1
python3.13 open_starnet.py $@
