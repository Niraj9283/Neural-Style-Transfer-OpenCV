@echo off
setlocal
cd /d "%~dp0"

if not exist "models\candy.t7" (
    python scripts\download_models.py --model candy
)

python src\cli.py --image assets\sample_content\campus_scene.png --model models\candy.t7 --output outputs\campus_candy.jpg --width 700 --compare

echo.
echo Demo complete. Output saved in the outputs folder.
pause
