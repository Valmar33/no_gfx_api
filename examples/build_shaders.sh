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

for folder in "${folders[@]}"; do
    for shader in "${folder}"/shaders/*.musl; do
        ../build/gpu_compiler "${shader}"
        glslangValidator -V "${shader%.*}.glsl" -o "${shader%.*}.spv"
    done
done

# Why not ~ doesn't have any other dependencies
for shader in third_party/dear_imgui/shaders/*.musl; do
    ../build/gpu_compiler "${shader}"
    glslangValidator -V "${shader%.*}.glsl" -o "${shader%.*}.spv"
done
