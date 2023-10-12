#!/bin/bash

# The following are functions for creating nicely formatted tmux panes

function 2panes {
  tmux split-window -d
  tmux select-layout tiled

  tmux set -g pane-border-status top
  sleep 1
}

function 4panes {
  tmux split-window -d
  tmux split-window -d
  tmux split-window -d
  tmux select-layout tiled

  tmux set -g pane-border-status top
  sleep 1
}

function 6panes {
  tmux selectp -t 0    # select the first (0) pane
  tmux splitw -h -p 75 # split it into two horizontal parts
  tmux selectp -t 0    # select the first (0) pane
  tmux splitw -v -p 50 # split it into two vertical halves

  tmux selectp -t 2    # select the new, second (2) pane
  tmux splitw -h -p 66 # split it into two halves
  tmux selectp -t 2    # select the second (2) pane
  tmux splitw -v -p 50 # split it into two vertical halves

  tmux selectp -t 4    # select the new, sixth (6) pane
  tmux splitw -v -p 50 # split it into two halves
  tmux selectp -t 0    # go back to the first pane

  tmux set -g pane-border-status top
  sleep 1
}

function 8panes {
  tmux selectp -t 0    # select the first (0) pane
  tmux splitw -h -p 75 # split it into two horizontal parts
  tmux selectp -t 0    # select the first (0) pane
  tmux splitw -v -p 50 # split it into two vertical halves

  tmux selectp -t 2    # select the new, second (2) pane
  tmux splitw -h -p 66 # split it into two halves
  tmux selectp -t 2    # select the second (2) pane
  tmux splitw -v -p 50 # split it into two vertical halves

  tmux selectp -t 4    # select the new, fourth (4) pane
  tmux splitw -h -p 50 # split it into two halves
  tmux selectp -t 4    # select the fourth (4) pane
  tmux splitw -v -p 50 # split it into two halves

  tmux selectp -t 6    # select the new, sixth (6) pane
  tmux splitw -v -p 50 # split it into two halves
  tmux selectp -t 0    # go back to the first pane

  tmux set -g pane-border-status top
  sleep 1
}

function 10panes {
  tmux selectp -t 0    # select the first (0) pane
  tmux splitw -h -p 75 # split it into two horizontal parts
  tmux selectp -t 0    # select the first (0) pane
  tmux splitw -v -p 50 # split it into two vertical halves

  tmux selectp -t 2    # select the new, second (2) pane
  tmux splitw -h -p 66 # split it into two halves
  tmux selectp -t 2    # select the second (2) pane
  tmux splitw -v -p 50 # split it into two vertical halves

  tmux selectp -t 4    # select the new, fourth (4) pane
  tmux splitw -h -p 50 # split it into two halves
  tmux selectp -t 4    # select the fourth (4) pane
  tmux splitw -v -p 50 # split it into two halves

  tmux selectp -t 6    # select the new, fourth (6) pane
  tmux splitw -h -p 50 # split it into two halves
  tmux selectp -t 6    # select the fourth (4) pane
  tmux splitw -v -p 50 # split it into two halves

  tmux selectp -t 8    # select the new, sixth (6) pane
  tmux splitw -v -p 50 # split it into two halves
  tmux selectp -t 0    # go back to the first pane

  tmux set -g pane-border-status top
  sleep 2
}