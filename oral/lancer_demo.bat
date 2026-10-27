@echo off
rem Lance l interface de demonstration : la page s ouvre dans le navigateur.
rem Pour l arreter : fermer cette fenetre (ou Ctrl + C).
cd /d "%~dp0.."
uv run python scripts\app.py
pause
