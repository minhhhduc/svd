@echo off
mingw32-make clean
mingw32-make all USE_OPENCL=0
if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)
echo Build successful! Running main program...
.\bin\main.exe