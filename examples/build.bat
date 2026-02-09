@echo off
setlocal

if not exist "../build" mkdir "../build"
if not exist "../build/third_party" mkdir "../build/third_party"

set flags=

for /D %%F in (*) do (
    if /I not "%%F"=="third_party" if /I not "%%F"=="shared" (
        odin build %%F %flags% -debug -out:../build/%%F.exe
        if errorlevel 1 (
            exit /b 1
        )
    )
)

for /D %%F in (third_party/*) do (
    odin build third_party/%%F %flags% -debug -out:../build/third_party/%%F.exe
    if errorlevel 1 (
        exit /b 1
    )
)
