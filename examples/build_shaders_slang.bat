@echo off
setlocal

for /D %%F in (*) do (
    if exist "%%F\shaders\" (
        for %%S in ("%%F\shaders\*.slang") do (
            echo Compiling %%S

            REM Compile vertex shader: both spirv and glsl targets
            slangc -target spirv -target glsl -fvk-use-scalar-layout -force-glsl-scalar-layout -entry vertexMain -stage vertex "%%F\shaders\%%~nS.slang" -o "%%F\shaders\%%~nS.vert.spv" -o "%%F\shaders\%%~nS.vert.glsl"
            
            REM Compile fragment shader: both spirv and glsl targets
            slangc -target spirv -target glsl -fvk-use-scalar-layout -force-glsl-scalar-layout -entry fragmentMain -stage fragment "%%F\shaders\%%~nS.slang" -o "%%F\shaders\%%~nS.frag.spv" -o "%%F\shaders\%%~nS.frag.glsl"
            
            REM Compile compute shader if computeMain exists (suppress errors if it doesn't)
            slangc -target spirv -target glsl -fvk-use-scalar-layout -force-glsl-scalar-layout -entry computeMain -stage compute "%%F\shaders\%%~nS.slang" -o "%%F\shaders\%%~nS.comp.spv" -o "%%F\shaders\%%~nS.comp.glsl" 2>nul
        )
    )
)
