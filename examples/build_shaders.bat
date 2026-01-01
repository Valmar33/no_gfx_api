@echo off
setlocal

set glsl_flags=

for /D %%F in (*) do (
    if exist "%%F\shaders\" (
        for %%S in ("%%F\shaders\*.musl") do (
            ..\build\gpu_compiler.exe "%%S" && (
                glslangvalidator %glsl_flags% -V "%%F\shaders\%%~nS.glsl" -o "%%F\shaders\%%~nS.spv"
            )
        )
    )
)
