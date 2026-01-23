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
    float time;
};

vec4 mat2_from_vec4(vec4 v);
vec2 mat2_mul(vec4 m, vec2 v);
vec4 raymarch(_res_ptr_Data data, vec3 global_invocation_id);
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

vec4 mat2_from_vec4(vec4 v)
{
    return v;
}

vec2 mat2_mul(vec4 m, vec2 v)
{
    return vec2(m.x * v.x + m.z * v.y, m.y * v.x + m.w * v.y);
}

vec4 raymarch(_res_ptr_Data data, vec3 global_invocation_id)
{
    vec2 C;
    float d;
    float z;
    vec4 o;
    vec4 p;
    vec2 r;
    vec4 O;
    C = global_invocation_id.xy;
    z = fract(dot(C, sin(C))) - 0.5;
    o = vec4(0.0, 0.0, 0.0, 0.0);
    r = data._res_.resolution;
    // for construct
    {
        float i;
        vec4 rot1;
        vec2 p_xy;
        vec4 rot2;
        vec4 numerator;
        float denominator;
        for(i = 1.0; i < 77.0; i = i + 1.0)
        {
            p = vec4(z * normalize(vec3(C - 0.5 * r, r.y)), 0.1 * data._res_.time);
            p = vec4(p.x, p.y, p.z + data._res_.time, p.w);
            O = p;
            rot1 = mat2_from_vec4(cos(2.0 + O.z + vec4(0.0, 11.0, 33.0, 0.0)));
            p_xy = mat2_mul(rot1, vec2(p.x, p.y));
            p = vec4(p_xy.x, p_xy.y, p.z, p.w);
            rot2 = mat2_from_vec4(cos(O + vec4(0.0, 11.0, 33.0, 0.0)));
            p_xy = mat2_mul(rot2, vec2(p.x, p.y));
            p = vec4(p_xy.x, p_xy.y, p.z, p.w);
            numerator = 1.0 + sin(0.5 * O.z + length(p - O) + vec4(0.0, 4.0, 3.0, 6.0));
            denominator = 0.5 + 2.0 * dot(O.xy, O.xy);
            O = numerator / denominator;
            p = abs(fract(p) - 0.5);
            d = abs(min(length(p.xy) - 0.125, min(p.x, p.y) + 0.001)) + 0.001;
            o = o + O.w / d * O;
            z = z + 0.6 * d;
        }
    }

    return tanh(o / 20000.0);
}

void main()
{
    _res_ptr_Data data = _res_compute_data_;
    vec3 global_invocation_id = gl_GlobalInvocationID;
    if(global_invocation_id.x < data._res_.resolution.x)
    {
        if(global_invocation_id.y < data._res_.resolution.y)
        {
            vec4 color;
            color = raymarch(data, global_invocation_id);
            imageStore(_res_textures_rw_[data._res_.output_texture_id], ivec2(global_invocation_id.xy), color);
        }

    }

}

