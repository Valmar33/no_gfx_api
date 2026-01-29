@echo off
setlocal

set glsl_flags=

goto comment
for /D %%F in (*) do (
    if exist "%%F\shaders\" (
        for %%S in ("%%F\shaders\*.glsl") do (
            glslangvalidator %glsl_flags% -V "%%F\shaders\%%~nS.glsl" -o "%%F\shaders\%%~nS.spv"
        )
    )
)
:comment

for /D %%F in (*) do (
    if exist "%%F\shaders\" (
        for %%S in ("%%F\shaders\*.musl") do (
            ..\build\gpu_compiler "%%F\shaders\%%~nS.musl"
            glslangvalidator %glsl_flags% -V "%%F\shaders\%%~nS.glsl" -o "%%F\shaders\%%~nS.spv"
        )
    )
)
