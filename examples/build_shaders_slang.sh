#!/usr/bin/env bash

folders=(
    1_triangle
    2_textures
    3_3D
    4_indirect_triangles
    5_compute_shaders
    6_deferred_async_load
    7_raytracing
)

txt_white=$(tput setaf 7)
txt_blue="$(tput setaf 4)"

for folder in "${folders[@]}"; do
    rm -f "${folder}"/shaders/*.spv
    rm -f "${folder}"/shaders/*.glsl
    
    for shader in "${folder}"/shaders/*.slang; do
        echo "${txt_blue}Trying to build ${shader} ...${txt_white}"
        
        if [ -f "${shader%.*}.vert.musl" ]; then
            slangc -target spirv -target glsl -fvk-use-scalar-layout -force-glsl-scalar-layout -entry vertexMain -stage vertex "${shader%.*}.slang" -o "${shader%.*}.vert.spv" -o "${shader%.*}.vert.glsl"
            spirv-val "${shader%.*}.vert.spv" --relax-block-layout --scalar-block-layout --target-env vulkan1.3
        fi

        if [ -f "${shader%.*}.frag.musl" ]; then
            slangc -target spirv -target glsl -fvk-use-scalar-layout -force-glsl-scalar-layout -entry fragmentMain -stage fragment "${shader%.*}.slang" -o "${shader%.*}.frag.spv" -o "${shader%.*}.frag.glsl"
            spirv-val "${shader%.*}.frag.spv" --relax-block-layout --scalar-block-layout --target-env vulkan1.3
        fi

        if [ -f "${shader%.*}.comp.musl" ]; then
            slangc -target spirv -target glsl -fvk-use-scalar-layout -force-glsl-scalar-layout -entry computeMain -stage compute "${shader%.*}.slang" -o "${shader%.*}.comp.spv" -o "${shader%.*}.comp.glsl"
            spirv-val "${shader%.*}.comp.spv" --relax-block-layout --scalar-block-layout --target-env vulkan1.3
        fi
    done
done