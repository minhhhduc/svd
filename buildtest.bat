@echo off
REM Build and run all test executables
make tests
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

if %TEST_FAILED%==1 (
	echo Some tests failed && exit /b 6
)

echo === All tests passed ===
endlocal
exit /b 0