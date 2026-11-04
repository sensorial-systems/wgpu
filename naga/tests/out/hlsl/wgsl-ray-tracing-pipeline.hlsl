struct HitCounters {
    uint hit_num;
    uint selected_hit;
};

struct RayDesc_ {
    uint flags;
    uint cull_mask;
    float tmin;
    float tmax;
    float3 origin;
    int _pad5_0;
    float3 dir;
    int _end_pad_0;
};

RayDesc RayDescFromRayDesc_(RayDesc_ arg0) {
    RayDesc ret = (RayDesc)0;
    ret.Origin = arg0.origin;
    ret.TMin = arg0.tmin;
    ret.Direction = arg0.dir;
    ret.TMax = arg0.tmax;
    return ret;
}

static HitCounters hit_num;
RaytracingAccelerationStructure acc_struct : register(t0);
static HitCounters incoming_hit_num;

RayDesc_ ConstructRayDesc_(uint arg0, uint arg1, float arg2, float arg3, float3 arg4, float3 arg5) {
    RayDesc_ ret = (RayDesc_)0;
    ret.flags = arg0;
    ret.cull_mask = arg1;
    ret.tmin = arg2;
    ret.tmax = arg3;
    ret.origin = arg4;
    ret.dir = arg5;
    return ret;
}

HitCounters ZeroValueHitCounters() {
    return (HitCounters)0;
}

[shader("raygeneration")]
void ray_gen_main()
{
    uint3 id = DispatchRaysIndex();
    uint3 num_invocations = DispatchRaysDimensions();
    hit_num = ZeroValueHitCounters();
    float3 shift = (float3(id) / float3(num_invocations));
    float3 ray_shift = ((float3(shift.x, 0.0, shift.y) * 2.0) - (1.0).xxx);
    {
        HitCounters naga_payload;
        naga_payload = hit_num;
        TraceRay(acc_struct, ConstructRayDesc_(0u, 255u, 0.01, 100.0, (0.0).xxx, (float3(0.0, 1.0, 0.0) + ray_shift)).flags, ConstructRayDesc_(0u, 255u, 0.01, 100.0, (0.0).xxx, (float3(0.0, 1.0, 0.0) + ray_shift)).cull_mask, 0, 0, 0, RayDescFromRayDesc_(ConstructRayDesc_(0u, 255u, 0.01, 100.0, (0.0).xxx, (float3(0.0, 1.0, 0.0) + ray_shift))), naga_payload);
        hit_num = naga_payload;
    }
    return;
}

[shader("miss")]
void miss(inout HitCounters naga_incoming)
{
    float3 origin = WorldRayOrigin();
    float3 dir = WorldRayDirection();
    float t_min = RayTMin();
    incoming_hit_num = naga_incoming;
    naga_incoming = incoming_hit_num;
    return;
}

uint naga_f2u32(float value) {
    return uint(clamp(value, 0.0, 4294967000.0));
}

[shader("anyhit")]
void any_hit_main(inout HitCounters naga_incoming, in BuiltInTriangleIntersectionAttributes naga_attributes)
{
    uint data = InstanceID();
    uint geo_idx = GeometryIndex();
    float max_ = RayTCurrent();
    uint kind = HitKind();
    float2 bary = naga_attributes.barycentrics;
    incoming_hit_num = naga_incoming;
    uint _e8 = incoming_hit_num.hit_num;
    incoming_hit_num.hit_num = (_e8 + 1u);
    incoming_hit_num.selected_hit = (data + naga_f2u32((bary.x + bary.y)));
    naga_incoming = incoming_hit_num;
    return;
}

[shader("closesthit")]
void closest_hit_main(inout HitCounters naga_incoming, in BuiltInTriangleIntersectionAttributes naga_attributes)
{
    float3 origin_1 = ObjectRayOrigin();
    float3 dir_1 = ObjectRayDirection();
    float4x3 obj_to_world = ObjectToWorld4x3();
    float4x3 world_to_obj = WorldToObject4x3();
    float2 bary_1 = naga_attributes.barycentrics;
    incoming_hit_num = naga_incoming;
    incoming_hit_num.selected_hit = naga_f2u32(((1.0 - bary_1.x) - bary_1.y));
    naga_incoming = incoming_hit_num;
    return;
}

[shader("closesthit")]
void closest_hit_triangle(inout HitCounters naga_incoming, in BuiltInTriangleIntersectionAttributes naga_attributes)
{
    uint instance_index = InstanceIndex();
    uint primitive_index = PrimitiveIndex();
    incoming_hit_num = naga_incoming;
    incoming_hit_num.hit_num = primitive_index;
    incoming_hit_num.hit_num = instance_index;
    naga_incoming = incoming_hit_num;
    return;
}
