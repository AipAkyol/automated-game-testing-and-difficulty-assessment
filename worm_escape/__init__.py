"""
worm_escape — A console-based directional extraction puzzle game.

This package exposes the core modules needed to run the game or build
tools (e.g. a level editor) on top of it:

  constants      – ANSI codes, directional vectors, rendering chars.
  entities        – Worm class.
  level_manager   – load_level() parser.
  renderer        – clear_screen(), render().
  engine          – Collision / movement logic, attempt_extraction().
  levels          – Built-in level data dicts.
"""
