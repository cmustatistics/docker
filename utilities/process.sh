#!/bin/bash

#####################################################
# parse parameters
#####################################################
. parser.sh
if [ -z ${session+x} ]; then echo "Session is not specified"; exit; fi

#####################################################
# kill previous tmux session and create a new one
#####################################################
tmux kill-session -t monitor 2> /dev/null
tmux new -s monitor -d

#####################################################
# specify the command to be run inside tmux
#####################################################
command=" if ! type "tmux" > /dev/null 2>&1;
          then source ~/miniconda3/bin/activate base;
          fi;
          tmux attach -t $session;"

#####################################################
# if the server parameter is specified, connect to it and exit
#####################################################

if [ -z ${session+x} ]; then echo "Session is not specified"; exit; fi

#####################################################
# if server is not specified, connect to all servers
# and looked for the specified session
#####################################################

#####################################################
# load functions for creating tmux panes
#####################################################
. panes.sh

#####################################################
# load servers list
#####################################################
. servers.sh

#####################################################
# create the required number of panes
#####################################################
4panes

#####################################################
# connect to each server and attach to the remote tmux session 
#####################################################
COUNTER=0
for server in "${servers[@]}"; do
  tmux send-keys -t monitor.$COUNTER "ssh -t $server \"$command\"" ENTER
  tmux select-pane -t monitor.$COUNTER -T "$server"
  COUNTER=$((COUNTER+1))
done

#####################################################
# attach in read only mode to the tmux session 
# displaying all the remote tmux sessions
#####################################################
tmux attach -r -t monitor