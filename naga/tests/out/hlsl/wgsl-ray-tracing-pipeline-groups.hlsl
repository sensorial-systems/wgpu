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

struct NagaRayValue {
    uint value;
};

RayDesc RayDescFromRayDesc_(RayDesc_ arg0) {
    RayDesc ret = (RayDesc)0;
    ret.Origin = arg0.origin;
    ret.TMin = arg0.tmin;
    ret.Direction = arg0.dir;
    ret.TMax = arg0.tmax;
    return ret;
}

static uint payload_;
RaytracingAccelerationStructure acc_struct : register(t0);
static uint incoming;

uint naga_mod(uint lhs, uint rhs) {
    return lhs % (rhs == 0u ? 1u : rhs);
}

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

[shader("raygeneration")]
void ray_gen_main()
{
    uint3 id = DispatchRaysIndex();
    RayDesc_ ray = ConstructRayDesc_(0u, 255u, 0.01, 100.0, (0.0).xxx, float3(0.0, 1.0, 0.0));
    {
        NagaRayValue naga_payload;
        naga_payload.value = payload_;
        TraceRay(acc_struct, ray.flags, ray.cull_mask, 0, 0, 0, RayDescFromRayDesc_(ray), naga_payload);
        payload_ = naga_payload.value;
    }
    {
        NagaRayValue naga_payload;
        naga_payload.value = payload_;
        TraceRay(acc_struct, ray.flags, ray.cull_mask, naga_mod(id.x, 3u), 0, naga_mod(id.y, 2u), RayDescFromRayDesc_(ray), naga_payload);
        payload_ = naga_payload.value;
    }
    return;
}

[shader("miss")]
void miss_main(inout NagaRayValue naga_incoming)
{
    incoming = naga_incoming.value;
    incoming = 0u;
    naga_incoming.value = incoming;
    return;
}

[shader("anyhit")]
void any_hit_main(inout NagaRayValue naga_incoming, in BuiltInTriangleIntersectionAttributes naga_attributes)
{
    uint primitive = PrimitiveIndex();
    incoming = naga_incoming.value;
    if ((primitive == 1u)) {
        naga_incoming.value = incoming;
        IgnoreHit();
    }
    if ((primitive == 2u)) {
        incoming = 2u;
        naga_incoming.value = incoming;
        AcceptHitAndEndSearch();
    }
    incoming = primitive;
    naga_incoming.value = incoming;
    return;
}

[shader("closesthit")]
void closest_hit_main(inout NagaRayValue naga_incoming, in BuiltInTriangleIntersectionAttributes naga_attributes)
{
    uint primitive_1 = PrimitiveIndex();
    incoming = naga_incoming.value;
    incoming = (primitive_1 + 1u);
    naga_incoming.value = incoming;
    return;
}
