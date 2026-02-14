#!/usr/bin/env bash

cd "$(dirname "$0")"

mkdir --verbose --parents "../build"

folders=(
    1_triangle
    2_textures
    3_3D
    4_indirect_triangles
    5_compute_shaders
    6_deferred_async_load
    7_raytracing
)

txt_white="$(tput setaf 7)"
txt_blue="$(tput setaf 4)"

# Print out extra checks for validations
for folder in "${folders[@]}"; do
    echo "${txt_blue}Testing strict compiler verification checks for ${folder} ...${txt_white}"
    odin build "${folder}" -vet -debug -out:"../build/${folder}"
done

# Build normally ~ shouldn't fail
for folder in "${folders[@]}"; do
    echo "${txt_blue}Compiling ${folder} ...${txt_white}"
    odin build "${folder}" -debug -out:"../build/${folder}"
done
