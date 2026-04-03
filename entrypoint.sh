#!/bin/sh
#
# HyperSPIN code - hspn_surrogate_models
# 
# Copyright 2025 Carnegie Mellon University.
# 
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
# 
# Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
# 
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
# 
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
# 
# DM25-0396
#
#!/bin/bash
set -e

groupadd -f -g $MY_GID devgroup 2>/dev/null || true
useradd -m -u $MY_UID -g $MY_GID -d /home/$MY_USER -s /bin/bash $MY_USER 2>/dev/null || true
usermod -aG root $MY_USER 2>/dev/null || true
passwd -d $MY_USER 2>/dev/null || true

export HOME=/home/$MY_USER
mkdir -p $HOME/.local
chown -R $MY_UID:$MY_GID $HOME

su $MY_USER -c "pip install -e /workspace"
umask 0002

if [ "$RUN_AS_ROOT" = "true" ]; then
    exec /opt/nvidia/nvidia_entrypoint.sh "$@"
elif [ $# -eq 0 ]; then
    exec su $MY_USER
else
    exec su $MY_USER -c "/opt/nvidia/nvidia_entrypoint.sh $*"
fi