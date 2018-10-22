FROM ubuntu:16.04

ARG DRAKE_URL

RUN apt-get update
RUN apt install -y sudo graphviz python-pip curl git openscad tmux screen
RUN pip install pip jupyter==4.10 graphviz meshcat numpy trimesh pyglet
RUN curl -o drake.tar.gz $DRAKE_URL && sudo tar -xzf drake.tar.gz -C /opt
RUN yes | sudo /opt/drake/share/drake/setup/install_prereqs
RUN git clone -b mrbv_tweaks https://github.com/gizatt/underactuated /underactuated
RUN yes | sudo /underactuated/scripts/setup/ubuntu/16.04/install_prereqs
RUN apt install -y python-tk xvfb mesa-utils libegl1-mesa libgl1-mesa-glx libglu1-mesa libx11-6 x11-common x11-xserver-utils

ENV PYTHONPATH=/underactuated/src:/opt/drake/lib/python2.7/site-packages

ENTRYPOINT bash
