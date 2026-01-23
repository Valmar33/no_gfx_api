#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_image_load_formatted : require
layout(local_size_x_id = 13370, local_size_y_id = 13371, local_size_z_id = 13372) in;


layout(buffer_reference) readonly buffer _res_ptr_void;
layout(buffer_reference) readonly buffer _res_ptr_Data;

struct Data
{
    uint output_texture_id;
    vec2 resolution;
};

void main();
layout(buffer_reference, scalar) readonly buffer _res_ptr_void { uint _res_void_; };
layout(buffer_reference, scalar) readonly buffer _res_ptr_Data { Data _res_; };

layout(set = 0, binding = 0) uniform texture2D _res_textures_[];
layout(set = 1, binding = 0) uniform image2D _res_textures_rw_[];
layout(set = 2, binding = 0) uniform sampler _res_samplers_[];

layout(push_constant, scalar) uniform Push
{
    _res_ptr_Data _res_compute_data_;
};

void main()
{
    _res_ptr_Data data = _res_compute_data_;
    vec3 global_invocation_id = gl_GlobalInvocationID;
    if(global_invocation_id.x < data._res_.resolution.x)
    {
        if(global_invocation_id.y < data._res_.resolution.y)
        {
            vec4 color;
            color = vec4(1.0, 1.0, 1.0, 1.0);
            imageStore(_res_textures_rw_[data._res_.output_texture_id], ivec2(global_invocation_id.xy), color);
        }

    }

}

