#!/bin/bash

#####################################################
# Name of the image and container
#####################################################
readonly name=test

#####################################################
# Load server list
#####################################################
. servers.sh

#####################################################
# for each server, stop containers and kill tmux server
#####################################################
command="docker stop $name; docker system prune --force; tmux kill-server"

for server in "${servers[@]}"; do
  ssh "$server" -t "$command"
done
