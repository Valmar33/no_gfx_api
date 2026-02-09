@echo off
setlocal

for /D %%F in (*) do (
    if exist "%%F\shaders\" (
        del "%%F\shaders\*.spv"
        del "%%F\shaders\*.glsl"

        for %%S in ("%%F\shaders\*.slang") do (
            echo Compiling %%S

            if exist "%%F\shaders\%%~nS.vert.musl" (
              slangc -target spirv -target glsl -fvk-use-scalar-layout -force-glsl-scalar-layout -validate-ir -no-mangle -entry vertexMain -stage vertex "%%F\shaders\%%~nS.slang" -o "%%F\shaders\%%~nS.vert.spv" -o "%%F\shaders\%%~nS.vert.glsl"
              spirv-val "%%F\shaders\%%~nS.vert.spv" --relax-block-layout --scalar-block-layout --target-env vulkan1.3
            )
            
            if exist "%%F\shaders\%%~nS.frag.musl" (
              slangc -target spirv -target glsl -fvk-use-scalar-layout -force-glsl-scalar-layout -validate-ir -no-mangle -entry fragmentMain -stage fragment "%%F\shaders\%%~nS.slang" -o "%%F\shaders\%%~nS.frag.spv" -o "%%F\shaders\%%~nS.frag.glsl"
              spirv-val "%%F\shaders\%%~nS.frag.spv" --relax-block-layout --scalar-block-layout --target-env vulkan1.3
            )
            
            if exist "%%F\shaders\%%~nS.comp.musl" (
              slangc -target spirv -target glsl -fvk-use-scalar-layout -force-glsl-scalar-layout -validate-ir -no-mangle -entry computeMain -stage compute "%%F\shaders\%%~nS.slang" -o "%%F\shaders\%%~nS.comp.spv" -o "%%F\shaders\%%~nS.comp.glsl"
              spirv-val "%%F\shaders\%%~nS.comp.spv" --relax-block-layout --scalar-block-layout --target-env vulkan1.3
            )
        )
    )
)
