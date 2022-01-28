#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0103,C0111

"""Definition of enum Direc."""

from enum import IntEnum, unique


@unique
class Direc(IntEnum):
    """Directions on the game plane."""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    NONE = 4

    @staticmethod
    def opposite(direc):
        """Return the opposite direction."""
        if direc == Direc.LEFT:
            return Direc.RIGHT
        elif direc == Direc.RIGHT:
            return Direc.LEFT
        elif direc == Direc.UP:
            return Direc.DOWN
        elif direc == Direc.DOWN:
            return Direc.UP
        else:
            return Direc.NONE
