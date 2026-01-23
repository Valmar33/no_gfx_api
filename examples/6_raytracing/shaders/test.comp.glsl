#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_image_load_formatted : require
#extension GL_EXT_ray_query : require
layout(local_size_x_id = 13370, local_size_y_id = 13371, local_size_z_id = 13372) in;


layout(buffer_reference) readonly buffer _res_ptr_void;
layout(buffer_reference) readonly buffer _res_ptr_Data;

struct Data
{
    uint output_texture_id;
    vec2 resolution;
    mat4 camera_to_world;
};

void main();
layout(buffer_reference, scalar) readonly buffer _res_ptr_void { uint _res_void_; };
layout(buffer_reference, scalar) readonly buffer _res_ptr_Data { Data _res_; };

layout(set = 0, binding = 0) uniform texture2D _res_textures_[];
layout(set = 1, binding = 0) uniform image2D _res_textures_rw_[];
layout(set = 2, binding = 0) uniform sampler _res_samplers_[];
layout(set = 3, binding = 0) uniform accelerationStructureEXT bvhs[];

layout(push_constant, scalar) uniform Push
{
    _res_ptr_Data _res_compute_data_;
};

struct Ray
{
    vec3 ori;
    vec3 dir;
};

bool ray_hit(Ray ray)
{
    rayQueryEXT rq;

    rayQueryInitializeEXT(
        rq,
        bvhs[0],
        gl_RayFlagsTerminateOnFirstHitEXT |
        gl_RayFlagsOpaqueEXT |
        gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF,              // cull mask
        ray.ori,
        0.001,             // tMin (avoid self-intersection)
        normalize(ray.dir),
        1e30               // tMax
    );

    while(rayQueryProceedEXT(rq));

    return rayQueryGetIntersectionTypeEXT(rq, true)
           != gl_RayQueryCommittedIntersectionNoneEXT;
}

void main()
{
    _res_ptr_Data data = _res_compute_data_;
    vec3 global_invocation_id = gl_GlobalInvocationID;
    vec2 resolution = data._res_.resolution.xy;

    vec4 color = vec4(1.0, 0.0, 0.0, 1.0);

    vec2 uv = global_invocation_id.xy / resolution;
    vec2 coord = 2.0f * uv - 1.0f;

    coord *= tan((90.0f * 3.1415926f / 180.0f) / 2.0f);
    coord.y *= resolution.y / resolution.x;

    vec3 world_camera_pos = (data._res_.camera_to_world * vec4(0.0f, 0.0f, 0.0f, 1.0f)).xyz;
    vec3 camera_lookat = normalize(vec3(coord, 1.0f));
    vec3 world_camera_lookat = normalize(data._res_.camera_to_world * vec4(camera_lookat, 0.0f)).xyz;

    Ray camera_ray = Ray(world_camera_pos, world_camera_lookat);
    if(ray_hit(camera_ray)) color = vec4(0.0, 1.0, 0.0, 1.0);

    if(global_invocation_id.x < data._res_.resolution.x)
    {
        if(global_invocation_id.y < data._res_.resolution.y)
        {
            imageStore(_res_textures_rw_[data._res_.output_texture_id], ivec2(global_invocation_id.xy), color);
        }
    }
}
