#!/bin/sh
echo "Removing current autostart file"
rm /etc/xdg/lxsession/LXDE-pi/autostart
echo "Copying autostart file to boot into desktop"
cp autostart /etc/xdg/lxsession/LXDE-pi
