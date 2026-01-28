#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_image_load_formatted : require
#extension GL_EXT_ray_query : require
layout(local_size_x_id = 13370, local_size_y_id = 13371, local_size_z_id = 13372) in;

// Raytracing intrinsics:

layout(set = 3, binding = 0) uniform accelerationStructureEXT _res_bvhs_[];

mat4 _res_mat4_from_mat4x3(mat4x3 m)
{
    // GLSL is column-major: m[col][row]
    return mat4(
        vec4(m[0], 0.0),
        vec4(m[1], 0.0),
        vec4(m[2], 0.0),
        vec4(m[3], 1.0)
    );
}

struct Ray_Desc
{
    uint flags;
    uint cull_mask;
    float t_min;
    float t_max;
    vec3 origin;
    vec3 dir;
};

struct Ray_Result
{
    uint kind;
    float t;
    uint instance_idx;
    uint primitive_idx;
    vec2 barycentrics;
    bool front_face;
    mat4 object_to_world;
    mat4 world_to_object;
};

Ray_Result rayquery_result(rayQueryEXT rq)
{
    Ray_Result res;
    res.kind = rayQueryGetIntersectionTypeEXT(rq, true);
    res.t = rayQueryGetIntersectionTEXT(rq, true);
    res.instance_idx  = rayQueryGetIntersectionInstanceIdEXT(rq, true);
    res.primitive_idx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    res.front_face    = rayQueryGetIntersectionFrontFaceEXT(rq, true);
    res.object_to_world = _res_mat4_from_mat4x3(rayQueryGetIntersectionObjectToWorldEXT(rq, true));
    res.world_to_object = _res_mat4_from_mat4x3(rayQueryGetIntersectionWorldToObjectEXT(rq, true));
    res.barycentrics    = rayQueryGetIntersectionBarycentricsEXT(rq, true);
    return res;
}

Ray_Result rayquery_candidate(rayQueryEXT rq)
{
    Ray_Result res;
    res.kind = rayQueryGetIntersectionTypeEXT(rq, false);
    res.t = rayQueryGetIntersectionTEXT(rq, false);
    res.instance_idx  = rayQueryGetIntersectionInstanceIdEXT(rq, false);
    res.primitive_idx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
    res.front_face    = rayQueryGetIntersectionFrontFaceEXT(rq, false);
    res.object_to_world = _res_mat4_from_mat4x3(rayQueryGetIntersectionObjectToWorldEXT(rq, false));
    res.world_to_object = _res_mat4_from_mat4x3(rayQueryGetIntersectionWorldToObjectEXT(rq, false));
    res.barycentrics    = rayQueryGetIntersectionBarycentricsEXT(rq, false);
    return res;
}

void rayquery_init(rayQueryEXT rq, Ray_Desc desc, uint bvh)
{
    rayQueryInitializeEXT(rq,
                          _res_bvhs_[nonuniformEXT(bvh)],
                          desc.flags,
                          desc.cull_mask,
                          desc.origin,
                          desc.t_min,
                          desc.dir,
                          desc.t_max);
}

bool rayquery_proceed(rayQueryEXT rq)
{
    return rayQueryProceedEXT(rq);
}

// Raytracing intrinsics end.



layout(buffer_reference) readonly buffer _res_ptr_void;
layout(buffer_reference) readonly buffer _res_slice_Instance;
layout(buffer_reference) readonly buffer _res_slice_Mesh;
layout(buffer_reference) readonly buffer _res_slice_vec3;
layout(buffer_reference) readonly buffer _res_slice_vec2;
layout(buffer_reference) readonly buffer _res_ptr_Data;

struct Scene
{
    _res_slice_Instance instances;
    uint instance_count;
    _res_slice_Mesh meshes;
    uint mesh_count;
};

struct Mesh
{
    _res_slice_vec3 pos;
    _res_slice_vec3 normal;
    _res_slice_vec2 uv;
};

struct Instance
{
    mat4 transform;
    uint mesh_idx;
};

struct Data
{
    uint output_texture_id;
    vec2 resolution;
    uint accum_counter;
    mat4 camera_to_world;
};

void main();
struct Ray
{
    vec3 ori;
    vec3 dir;
};

struct Hit_Info
{
    bool hit;
    float t;
    vec3 normal;
    vec2 uv;
};

Hit_Info ray_scene_intersection(Ray ray);
layout(buffer_reference, scalar) readonly buffer _res_ptr_void { uint _res_void_; };
layout(buffer_reference, scalar) readonly buffer _res_slice_Instance { Instance _res_[]; };
layout(buffer_reference, scalar) readonly buffer _res_slice_Mesh { Mesh _res_[]; };
layout(buffer_reference, scalar) readonly buffer _res_slice_vec3 { vec3 _res_[]; };
layout(buffer_reference, scalar) readonly buffer _res_slice_vec2 { vec2 _res_[]; };
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
    vec4 color;
    vec2 uv;
    vec2 coord;
    vec3 world_camera_pos;
    vec3 camera_lookat;
    vec3 world_camera_lookat;
    Ray camera_ray;
    color = vec4(1, 0, 0, 1);
    uv = (global_invocation_id.xy / data._res_.resolution);
    coord = ((2.0 * uv) - 1.0);
    coord = (coord * tan(((90.0 * (3.1415926 / 180.0)) / 2.0)));
    coord.y = (coord.y * (data._res_.resolution.y / data._res_.resolution.x));
    world_camera_pos = (data._res_.camera_to_world * vec4(0, 0, 0, 1)).xyz;
    camera_lookat = normalize(vec3(coord, 1));
    world_camera_lookat = normalize((data._res_.camera_to_world * vec4(camera_lookat, 0.0))).xyz;
    camera_ray.ori = world_camera_pos;
    camera_ray.dir = world_camera_lookat;
    if(ray_scene_intersection(camera_ray).hit)
    {
        color = vec4(0, 1, 0, 1);
    }

    if((global_invocation_id.x < data._res_.resolution.x))
    {
        if((global_invocation_id.y < data._res_.resolution.y))
        {
            imageStore(_res_textures_rw_[data._res_.output_texture_id], ivec2(global_invocation_id.xy), color);
        }

    }

}

Hit_Info ray_scene_intersection(Ray ray)
{
    uint Ray_Flags_Opaque;
    uint Ray_Flags_Terminate_On_First_Hit;
    uint Ray_Flags_Skip_Closest_Hit_Shader;
    uint Ray_Result_Kind_Miss;
    uint Ray_Result_Kind_Hit_Mesh;
    uint Ray_Result_Kind_Hit_AABB;
    Hit_Info hit_info;
    Ray_Desc desc;
    rayQueryEXT rq;
    Ray_Result hit;
    Ray_Flags_Opaque = 1;
    Ray_Flags_Terminate_On_First_Hit = 4;
    Ray_Flags_Skip_Closest_Hit_Shader = 8;
    Ray_Result_Kind_Miss = 0;
    Ray_Result_Kind_Hit_Mesh = 1;
    Ray_Result_Kind_Hit_AABB = 2;
    desc.flags = Ray_Flags_Opaque;
    desc.cull_mask = 0xFF;
    desc.t_min = 0.001;
    desc.t_max = 1000000000.0;
    desc.origin = ray.ori;
    desc.dir = ray.dir;
    rayquery_init(rq, desc, 0);
    rayquery_proceed(rq);
    hit = rayquery_result(rq);
    if((hit.kind != Ray_Result_Kind_Hit_Mesh))
    {
        hit_info.hit = false;
        return hit_info;
    }

    hit_info.hit = true;
    hit_info.t = hit.t;
    return hit_info;
}

