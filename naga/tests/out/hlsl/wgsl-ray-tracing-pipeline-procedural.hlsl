struct Sphere {
    float3 normal;
    uint inside;
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

struct NagaRayValue {
    float4 value;
};

struct NagaNoAttributes {
    uint unused;
};

RayDesc RayDescFromRayDesc_(RayDesc_ arg0) {
    RayDesc ret = (RayDesc)0;
    ret.Origin = arg0.origin;
    ret.TMin = arg0.tmin;
    ret.Direction = arg0.dir;
    ret.TMax = arg0.tmax;
    return ret;
}

static float4 payload_;
static float4 shade;
RaytracingAccelerationStructure acc_struct : register(t0);
static Sphere sphere;
static float4 incoming;
static float4 callable_in;

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
        TraceRay(acc_struct, ray.flags, ray.cull_mask, 1u, 2u, naga_mod(id.y, 2u), RayDescFromRayDesc_(ray), naga_payload);
        payload_ = naga_payload.value;
    }
    float4 _e21 = payload_;
    shade = _e21;
    {
        NagaRayValue naga_data;
        naga_data.value = shade;
        CallShader(naga_mod(id.x, 2u), naga_data);
        shade = naga_data.value;
    }
    return;
}

[shader("intersection")]
void intersection_main()
{
    float3 origin = ObjectRayOrigin();
    float3 direction = ObjectRayDirection();
    float t_min = RayTMin();
    uint primitive = PrimitiveIndex();
    float b = dot(origin, direction);
    float c = (dot(origin, origin) - 1.0);
    float a = dot(direction, direction);
    float discriminant = ((b * b) - (a * c));
    if ((discriminant < 0.0)) {
        return;
    }
    float root = sqrt(discriminant);
    float near = ((-(b) - root) / a);
    float far = ((-(b) + root) / a);
    sphere.normal = normalize((origin + (near * direction)));
    sphere.inside = 0u;
    bool _e31 = false;
    {
        Sphere naga_hit;
        naga_hit = sphere;
        _e31 = ReportHit(near, naga_mod(primitive, 128u), naga_hit);
    }
    if (!(_e31)) {
        sphere.normal = -(normalize((origin + (far * direction))));
        sphere.inside = 1u;
        bool _e44 = false;
        {
            Sphere naga_hit;
            naga_hit = sphere;
            _e44 = ReportHit(far, naga_mod(primitive, 128u), naga_hit);
        }
        return;
    } else {
        return;
    }
}

[shader("anyhit")]
void any_hit_main(inout NagaRayValue naga_incoming, in Sphere naga_attributes)
{
    uint flags = RayFlags();
    incoming = naga_incoming.value;
    sphere = naga_attributes;
    bool local = (bool)0;

    uint _e3 = sphere.inside;
    if ((_e3 == 1u)) {
        local = ((flags & 16u) != 0u);
    } else {
        local = false;
    }
    bool _e13 = local;
    if (_e13) {
        naga_incoming.value = incoming;
        IgnoreHit();
    } else {
        naga_incoming.value = incoming;
        return;
    }
}

[shader("closesthit")]
void closest_hit_main(inout NagaRayValue naga_incoming, in Sphere naga_attributes)
{
    uint kind = HitKind();
    incoming = naga_incoming.value;
    sphere = naga_attributes;
    float3 _e4 = sphere.normal;
    incoming = float4(_e4, float(kind));
    naga_incoming.value = incoming;
    return;
}

[shader("miss")]
void miss_main(inout NagaRayValue naga_incoming)
{
    uint flags_1 = RayFlags();
    incoming = naga_incoming.value;
    incoming = (float(flags_1)).xxxx;
    naga_incoming.value = incoming;
    return;
}

[shader("callable")]
void callable_main(inout NagaRayValue naga_incoming)
{
    callable_in = naga_incoming.value;
    float4 _e2 = callable_in;
    callable_in = (_e2 * 0.5);
    naga_incoming.value = callable_in;
    return;
}
