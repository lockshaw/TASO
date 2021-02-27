source "$HOME/FlexFlow/env.sh"
ml restore taso
ml gdb 
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/TASO/build/"
export LIBRARY_PATH="$LIBRARY_PATH:$HOME/TASO/build/"
export TASO_HOME="$HOME/TASO"
