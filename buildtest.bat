@echo off
rem buildtest.bat - build and run tests

setlocal enabledelayedexpansion

where mingw32-make >nul 2>&1
if %ERRORLEVEL%==0 (
	set MAKE=mingw32-make
) else (
	where make >nul 2>&1
	if %ERRORLEVEL%==0 (
		set MAKE=make
	) else (
		echo Neither mingw32-make nor make found in PATH. && exit /b 2
	)
)

echo Cleaning previous build...
%MAKE% clean

echo Building tests with OpenMP (OpenCL disabled)...
%MAKE% build-all-test USE_OPENCL=0 || (echo Build test failed && exit /b 3)

echo === All tests passed ===
endlocal
exit /b 0