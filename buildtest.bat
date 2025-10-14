@echo off
rem buildtest.bat - build all test binaries and run every bin\t_* executable (Windows cmd)

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

%MAKE% test || (echo Build test failed && exit /b 3)

set FOUND=0
for %%F in (bin\t_*) do (
	if exist "%%F.exe" (
		set FOUND=1
		echo ================Running %%~nF.exe...================
		"%%F.exe" || (echo Test %%~nF failed with code %%ERRORLEVEL% & exit /b 4)
	) else if exist "%%F" (
		set FOUND=1
		echo ================Running %%~nF...================
		"%%F" || (echo Test %%~nF failed with code %%ERRORLEVEL% & exit /b 4)
	)
)

if %FOUND%==0 (
	echo No test binaries found matching bin\t_* && exit /b 5
)

echo === all tests passed ===
endlocal
exit /b 0
