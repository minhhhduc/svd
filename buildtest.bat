@echo off
REM Build and run all test executables
echo Building ...
make tests >nul 2>&1
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)
for %%F in (bin\_*.exe) do (
	echo.
	echo.
    echo =======%%~nF=======
    %%F
    if %errorlevel% neq 0 (
        echo Test %%~nF failed with code %errorlevel%
    )
)
			set TEST_FAILED=1
		)
	)
)
endlocal
exit /b 0