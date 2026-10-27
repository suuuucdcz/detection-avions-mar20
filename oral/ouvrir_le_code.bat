@echo off
rem Ouvre le projet dans VS Code, avec les fichiers des 10 passages de la partie E du guide.
cd /d "%~dp0.."
call code . src\avions\data\voc.py src\avions\data\split.py src\avions\data\crops.py configs\detecteur_1classe.yaml scripts\train_detector.py src\avions\classif\models.py src\avions\classif\transforms.py scripts\train_classifier.py src\avions\pipeline.py src\avions\evaluation.py src\avions\app.py
