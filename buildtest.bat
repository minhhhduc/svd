@echo off
rem buildtest.bat - build the test binary and run it (Windows cmd)

setlocal enabledelayedexpansion

echo === building test target ===
if not exist Makefile (
	echo Makefile not found in %CD% && exit /b 1
)

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

rem Build the test target (Makefile defines the 'test' target)
%MAKE% test || (echo Build test failed && exit /b 3)

rem Test binary location per Makefile
if exist bin\test_n2array.exe (
	set TESTBIN=bin\test_n2array.exe
) else if exist bin\test_n2array (
	set TESTBIN=bin\test_n2array
) else (
	echo Test binary not found in bin\ && exit /b 4
)

echo === running %TESTBIN% ===
%TESTBIN% || (echo Test exited with code %ERRORLEVEL% && exit /b 5)

echo === test finished ===
endlocal
exit /b 0
