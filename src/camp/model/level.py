"""Utility functions for MTP level calculations."""


def get_level(u: int, v: int):
    return 2 + 4 * u + v


def get_u(level: int, v: int = 0):
    return (level - 2 - v) // 4


def get_v(level: int, u: int = 0):
    return level - 2 - 4 * u
