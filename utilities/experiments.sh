#!/bin/bash

#############################################
# load servers
#############################################
. servers.sh

#############################################
# Declare parameters that change from server to server
#############################################
parameters=(
  20
  15
  10
  5
)

#############################################
# We iterate over the servers and run the script
# with different parameters in each one of them
# Note that distribute.sh requires TMUX and RSYNC to
# be installed locally
#############################################
for i in "${!parameters[@]}"; do
  ./distribute.sh --server "${servers[i]}"        \
                  --session TEST                  \
                  --location "~/remote/sparsegp2"
#                 "${parameters[i]}" # uncomment this line
#                                    # if your algorithm accepts parameters
done

