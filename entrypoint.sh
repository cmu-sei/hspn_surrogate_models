#!/bin/sh

uid=$(id -u)
gid=$(id -g)
[[ $uid -eq 0 ]] || export HOME=/home/$USER

[[ $uid -eq 0 ]] || su - root <<!
password

chmod 777 /root
chmod 777 /root/.cache

work_dir=$(pwd)
groupadd --gid $gid $USER
useradd --uid $uid --gid $gid --create-home --home-dir $HOME $USER
usermod -aG root $USER
passwd -d $USER
!
export PATH=$PATH:$HOME/.local/bin
pip install -e .

umask 0002
/opt/nvidia/nvidia_entrypoint.sh $@
