@echo off
REM Build and run main executable
echo Building main...
make main
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)
echo Running main...
bin\main.exe