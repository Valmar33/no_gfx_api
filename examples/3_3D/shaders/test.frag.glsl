#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_image_load_formatted : require

layout(location = 0) out vec4 _res_out_loc0_;
layout(location = 0) in vec4 _res_in_loc0_;

layout(buffer_reference) readonly buffer _res_ptr_void;

void main();
layout(buffer_reference, scalar) readonly buffer _res_ptr_void { uint _res_void_; };

layout(set = 0, binding = 0) uniform texture2D _res_textures_[];
layout(set = 1, binding = 0) uniform image2D _res_textures_rw_[];
layout(set = 2, binding = 0) uniform sampler _res_samplers_[];

layout(push_constant, scalar) uniform Push
{
    _res_ptr_void _res_vert_data_;
    _res_ptr_void _res_frag_data_;
    _res_ptr_void _res_indirect_data_;
};

void main()
{
    vec4 normal_vert = _res_in_loc0_;
    vec3 normal;
    normal = normalize(normal_vert.xyz);
    _res_out_loc0_ = vec4(((normal * 0.5) + 0.5), 1.0);
}

