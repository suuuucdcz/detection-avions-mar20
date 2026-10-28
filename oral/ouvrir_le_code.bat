@echo off
rem Ouvre le projet dans VS Code, avec les 5 notebooks et l interface de la demo (partie E du guide).
cd /d "%~dp0.."
call code . etapes\1_analyse_des_donnees.ipynb etapes\2_preparation_des_donnees.ipynb etapes\3_detection.ipynb etapes\4_classification.ipynb etapes\5_integration.ipynb app.py
