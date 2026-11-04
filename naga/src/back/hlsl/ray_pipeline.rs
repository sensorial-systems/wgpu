//! Ray tracing pipeline stages, as DirectX Raytracing library shaders.
//!
//! A stage is a `[shader("...")]` function of a DXIL library. DXR passes a
//! stage its incoming payload (or a callable its data) as an `inout`
//! parameter, a hit stage its hit's attributes as an `in` one, and has every
//! other builtin an intrinsic function. Naga's IR has these as global
//! variables, which helper functions read and write too: each is written as
//! a `static` global, which the stage's prologue fills from its parameter
//! and which is written back to the payload parameter before each return
//! and each verdict -- DXR keeps a payload's writes past `IgnoreHit()` and
//! `AcceptHitAndEndSearch()`.
//!
//! DXR wants payloads, callable data and hit attributes of struct types:
//! those of other types are wrapped in a struct of one member, `value`.

use alloc::{
    format,
    string::{String, ToString},
};
use core::fmt::Write;

use super::{BackendResult, Error};
use crate::{
    back::{self, Level},
    proc::NameKey,
    AddressSpace, Handle, Module, ShaderStage, TypeInner,
};

/// The parameters DXR passes a stage its payload and its hit's attributes
/// by.
pub(super) const PAYLOAD_PARAMETER: &str = "naga_incoming";
pub(super) const ATTRIBUTES_PARAMETER: &str = "naga_attributes";
/// A triangle hit's attributes.
const TRIANGLE_ATTRIBUTES: &str = "BuiltInTriangleIntersectionAttributes";

pub(super) const fn is_ray_tracing_stage(stage: ShaderStage) -> bool {
    matches!(
        stage,
        ShaderStage::RayGeneration
            | ShaderStage::Miss
            | ShaderStage::ClosestHit
            | ShaderStage::AnyHit
            | ShaderStage::Intersection
            | ShaderStage::Callable
    )
}

/// The name of a stage in a `[shader("...")]` attribute.
pub(super) const fn shader_attribute(stage: ShaderStage) -> Option<&'static str> {
    Some(match stage {
        ShaderStage::RayGeneration => "raygeneration",
        ShaderStage::Miss => "miss",
        ShaderStage::ClosestHit => "closesthit",
        ShaderStage::AnyHit => "anyhit",
        ShaderStage::Intersection => "intersection",
        ShaderStage::Callable => "callable",
        _ => return None,
    })
}

/// A struct DXR takes as a payload, callable data or hit attributes, for a
/// type of the module.
#[derive(Clone, Debug)]
pub(super) struct RayStruct {
    pub name: String,
    /// Whether the type is wrapped, in member `value`.
    pub wrapped: bool,
}

impl RayStruct {
    pub const fn value(&self) -> &'static str {
        if self.wrapped {
            ".value"
        } else {
            ""
        }
    }
}

impl<W: Write> super::Writer<'_, W> {
    /// Write the structs DXR takes for the module's payloads, callable data
    /// and hit attributes of types that are not structs already.
    pub(super) fn write_ray_tracing_structs(&mut self, module: &Module) -> BackendResult {
        for (_, global) in module.global_variables.iter() {
            if !matches!(
                global.space,
                AddressSpace::RayPayload
                    | AddressSpace::IncomingRayPayload
                    | AddressSpace::RayHitAttribute
                    | AddressSpace::CallableData
                    | AddressSpace::IncomingCallableData
            ) || self.ray_structs.contains_key(&global.ty)
            {
                continue;
            }
            let ray_struct = match module.types[global.ty].inner {
                TypeInner::Struct { .. } => RayStruct {
                    name: self.names[&NameKey::Type(global.ty)].clone(),
                    wrapped: false,
                },
                _ => {
                    let name = self.namer.call("NagaRayValue");
                    write!(self.out, "struct {name} {{\n{}", back::INDENT)?;
                    self.write_type(module, global.ty)?;
                    write!(self.out, " value")?;
                    if let TypeInner::Array { base, size, .. } = module.types[global.ty].inner {
                        self.write_array_size(module, base, size)?;
                    }
                    writeln!(self.out, ";\n}};\n")?;
                    RayStruct {
                        name,
                        wrapped: true,
                    }
                }
            };
            self.ray_structs.insert(global.ty, ray_struct);
        }
        if module
            .entry_points
            .iter()
            .any(|ep| ep.stage == ShaderStage::Intersection)
        {
            let name = self.namer.call("NagaNoAttributes");
            writeln!(
                self.out,
                "struct {name} {{\n{}uint unused;\n}};\n",
                back::INDENT
            )?;
            self.no_attributes = Some(name);
        }
        Ok(())
    }

    /// The hit attribute global a function uses, if any; the module's only
    /// one where it uses none, as a helper reporting hits may not.
    fn hit_attribute(
        &self,
        module: &Module,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> Option<Handle<crate::GlobalVariable>> {
        let attributes = || {
            module
                .global_variables
                .iter()
                .filter(|&(_, global)| global.space == AddressSpace::RayHitAttribute)
                .map(|(handle, _)| handle)
        };
        let used = attributes().find(|&handle| !func_ctx.info[handle].is_empty());
        let mut all = attributes();
        used.or_else(|| match (all.next(), all.next()) {
            (Some(only), None) => Some(only),
            _ => None,
        })
    }

    /// The parameters of a ray tracing stage: its payload or data, and its
    /// hit's attributes.
    pub(super) fn write_ray_tracing_parameters(
        &mut self,
        module: &Module,
        ep_index: u16,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> BackendResult {
        let ep = &module.entry_points[ep_index as usize];
        let mut separator = "";
        if let Some(incoming) = ep.incoming_ray_payload {
            let ty = module.global_variables[incoming].ty;
            let name = self.ray_structs[&ty].name.clone();
            write!(self.out, "inout {name} {PAYLOAD_PARAMETER}")?;
            separator = ", ";
        }
        if matches!(ep.stage, ShaderStage::ClosestHit | ShaderStage::AnyHit) {
            let name = match self.hit_attribute(module, func_ctx) {
                Some(global) if !func_ctx.info[global].is_empty() => self.ray_structs
                    [&module.global_variables[global].ty]
                    .name
                    .clone(),
                _ => TRIANGLE_ATTRIBUTES.into(),
            };
            write!(self.out, "{separator}in {name} {ATTRIBUTES_PARAMETER}")?;
        }
        Ok(())
    }

    /// A ray tracing stage's prologue: its arguments, from the intrinsics
    /// they are, and its payload and hit attributes, from its parameters.
    pub(super) fn write_ray_tracing_prologue(
        &mut self,
        module: &Module,
        func: &crate::Function,
        ep_index: u16,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> BackendResult {
        let ep = &module.entry_points[ep_index as usize];
        for (index, argument) in func.arguments.iter().enumerate() {
            let name = self.names[&NameKey::EntryPointArgument(ep_index, index as u32)].clone();
            write!(self.out, "{}", back::INDENT)?;
            self.write_type(module, argument.ty)?;
            write!(self.out, " {name} = ")?;
            match module.types[argument.ty].inner {
                TypeInner::Struct { ref members, .. } => {
                    write!(self.out, "{{ ")?;
                    for (position, member) in members.iter().enumerate() {
                        if position != 0 {
                            write!(self.out, ", ")?;
                        }
                        write!(
                            self.out,
                            "{}",
                            ray_tracing_built_in(member.binding.as_ref())?
                        )?;
                    }
                    write!(self.out, " }}")?;
                }
                _ => write!(
                    self.out,
                    "{}",
                    ray_tracing_built_in(argument.binding.as_ref())?
                )?,
            }
            writeln!(self.out, ";")?;
        }
        if let Some(incoming) = ep.incoming_ray_payload {
            let name = self.names[&NameKey::GlobalVariable(incoming)].clone();
            let value = self.ray_structs[&module.global_variables[incoming].ty].value();
            writeln!(
                self.out,
                "{}{name} = {PAYLOAD_PARAMETER}{value};",
                back::INDENT
            )?;
        }
        if matches!(ep.stage, ShaderStage::ClosestHit | ShaderStage::AnyHit) {
            if let Some(global) = self.hit_attribute(module, func_ctx) {
                if !func_ctx.info[global].is_empty() {
                    let name = self.names[&NameKey::GlobalVariable(global)].clone();
                    let value = self.ray_structs[&module.global_variables[global].ty].value();
                    writeln!(
                        self.out,
                        "{}{name} = {ATTRIBUTES_PARAMETER}{value};",
                        back::INDENT
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Where `func_ctx` is a stage with an incoming payload, write it back
    /// to the stage's parameter, as the stage ends.
    pub(super) fn write_ray_tracing_payload_back(
        &mut self,
        module: &Module,
        level: Level,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> BackendResult {
        let back::FunctionType::EntryPoint(index) = func_ctx.ty else {
            return Ok(());
        };
        let ep = &module.entry_points[index as usize];
        if !is_ray_tracing_stage(ep.stage) {
            return Ok(());
        }
        if let Some(incoming) = ep.incoming_ray_payload {
            let name = self.names[&NameKey::GlobalVariable(incoming)].clone();
            let value = self.ray_structs[&module.global_variables[incoming].ty].value();
            writeln!(self.out, "{level}{PAYLOAD_PARAMETER}{value} = {name};")?;
        }
        Ok(())
    }

    /// A ray tracing pipeline function: a trace, a verdict, a reported hit
    /// or a callable's execution.
    pub(super) fn write_ray_pipeline_function(
        &mut self,
        module: &Module,
        function: &crate::RayPipelineFunction,
        func_ctx: &back::FunctionCtx<'_>,
        level: Level,
    ) -> BackendResult {
        let groups = self.options.ray_tracing_pipeline_groups;
        match *function {
            crate::RayPipelineFunction::TraceRay {
                acceleration_structure,
                descriptor,
                payload,
                groups: ref chosen,
                time,
                hit_object,
            } => {
                if hit_object.is_some() {
                    return Err(Error::Unimplemented(
                        "hit objects: shader execution reordering needs shader model 6.9".into(),
                    ));
                }
                if time.is_some() {
                    return Err(Error::Unimplemented(
                        "rays traced at a time: DXR has no motion blur".into(),
                    ));
                }
                let crate::Expression::GlobalVariable(global) = func_ctx.expressions[payload]
                else {
                    unreachable!("checked when validating the module")
                };
                let payload_name = self.names[&NameKey::GlobalVariable(global)].clone();
                let ray_struct = self.ray_structs[&module.global_variables[global].ty].clone();
                let value = ray_struct.value();
                writeln!(self.out, "{level}{{")?;
                let inner = level.next();
                writeln!(self.out, "{inner}{} naga_payload;", ray_struct.name)?;
                writeln!(self.out, "{inner}naga_payload{value} = {payload_name};")?;
                write!(self.out, "{inner}TraceRay(")?;
                self.write_expr(module, acceleration_structure, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, descriptor, func_ctx)?;
                write!(self.out, ".flags, ")?;
                self.write_expr(module, descriptor, func_ctx)?;
                write!(self.out, ".cull_mask, ")?;
                // The hit group offset, the geometry stride and the miss
                // index, of which DXR reads 4, 4 and 16 bits, clamped within
                // the pipeline's table where its counts are known.
                let index = |this: &mut Self,
                             expression: Option<Handle<crate::Expression>>,
                             bits: u32,
                             last: Option<u32>|
                 -> BackendResult {
                    let Some(expression) = expression else {
                        return Ok(write!(this.out, "0")?);
                    };
                    match last {
                        Some(last) => {
                            write!(this.out, "min(")?;
                            this.write_expr(module, expression, func_ctx)?;
                            write!(this.out, " & {}u, {last}u)", (1u32 << bits) - 1)?;
                        }
                        None => this.write_expr(module, expression, func_ctx)?,
                    }
                    Ok(())
                };
                index(
                    self,
                    chosen.map(|chosen| chosen.hit_group_offset),
                    4,
                    groups.map(|groups| groups.hit_groups.max(1) - 1),
                )?;
                write!(self.out, ", ")?;
                index(
                    self,
                    chosen.and_then(|chosen| chosen.geometry_stride),
                    4,
                    groups.map(|groups| groups.max_geometry_stride),
                )?;
                write!(self.out, ", ")?;
                index(
                    self,
                    chosen.map(|chosen| chosen.miss_index),
                    16,
                    groups.map(|groups| groups.misses.max(1) - 1),
                )?;
                write!(self.out, ", RayDescFromRayDesc_(")?;
                self.write_expr(module, descriptor, func_ctx)?;
                writeln!(self.out, "), naga_payload);")?;
                writeln!(self.out, "{inner}{payload_name} = naga_payload{value};")?;
                writeln!(self.out, "{level}}}")?;
            }
            crate::RayPipelineFunction::IgnoreIntersection
            | crate::RayPipelineFunction::TerminateRay => {
                // The payload's writes stand, but only the stage itself has
                // the parameter to write them to.
                if !matches!(func_ctx.ty, back::FunctionType::EntryPoint(_)) {
                    return Err(Error::Unimplemented(
                        "an any-hit verdict outside the stage's entry point".into(),
                    ));
                }
                self.write_ray_tracing_payload_back(module, level, func_ctx)?;
                let call = match *function {
                    crate::RayPipelineFunction::IgnoreIntersection => "IgnoreHit",
                    _ => "AcceptHitAndEndSearch",
                };
                writeln!(self.out, "{level}{call}();")?;
            }
            crate::RayPipelineFunction::ReportIntersection {
                hit_t,
                hit_kind,
                result,
            } => {
                let name = back::Baked(result).to_string();
                writeln!(self.out, "{level}bool {name} = false;")?;
                writeln!(self.out, "{level}{{")?;
                let inner = level.next();
                match self.hit_attribute(module, func_ctx) {
                    Some(global) => {
                        let attribute = self.names[&NameKey::GlobalVariable(global)].clone();
                        let ray_struct =
                            self.ray_structs[&module.global_variables[global].ty].clone();
                        writeln!(self.out, "{inner}{} naga_hit;", ray_struct.name)?;
                        writeln!(
                            self.out,
                            "{inner}naga_hit{} = {attribute};",
                            ray_struct.value()
                        )?;
                    }
                    None => {
                        let empty = self
                            .no_attributes
                            .clone()
                            .expect("written with the structs");
                        writeln!(self.out, "{inner}{empty} naga_hit = ({empty})0;")?;
                    }
                }
                write!(self.out, "{inner}{name} = ReportHit(")?;
                self.write_expr(module, hit_t, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, hit_kind, func_ctx)?;
                writeln!(self.out, ", naga_hit);")?;
                writeln!(self.out, "{level}}}")?;
                self.named_expressions.insert(result, name);
            }
            crate::RayPipelineFunction::ExecuteCallable { index, data } => {
                let crate::Expression::GlobalVariable(global) = func_ctx.expressions[data] else {
                    unreachable!("checked when validating the module")
                };
                // Without callable stages, nothing to run.
                if groups.is_some_and(|groups| groups.callables == 0) {
                    return Ok(());
                }
                let data_name = self.names[&NameKey::GlobalVariable(global)].clone();
                let ray_struct = self.ray_structs[&module.global_variables[global].ty].clone();
                let value = ray_struct.value();
                writeln!(self.out, "{level}{{")?;
                let inner = level.next();
                writeln!(self.out, "{inner}{} naga_data;", ray_struct.name)?;
                writeln!(self.out, "{inner}naga_data{value} = {data_name};")?;
                write!(self.out, "{inner}CallShader(")?;
                match groups {
                    Some(groups) => {
                        write!(self.out, "min(")?;
                        self.write_expr(module, index, func_ctx)?;
                        write!(self.out, ", {}u)", groups.callables - 1)?;
                    }
                    None => self.write_expr(module, index, func_ctx)?,
                }
                writeln!(self.out, ", naga_data);")?;
                writeln!(self.out, "{inner}{data_name} = naga_data{value};")?;
                writeln!(self.out, "{level}}}")?;
            }
            crate::RayPipelineFunction::HitObjectRecordMiss { .. }
            | crate::RayPipelineFunction::HitObjectRecordEmpty { .. }
            | crate::RayPipelineFunction::HitObjectExecute { .. }
            | crate::RayPipelineFunction::ReorderThread { .. } => {
                return Err(Error::Unimplemented(
                    "hit objects: shader execution reordering needs shader model 6.9".into(),
                ))
            }
        }
        Ok(())
    }
}

/// The intrinsic a ray tracing stage reads a builtin by.
fn ray_tracing_built_in(binding: Option<&crate::Binding>) -> Result<String, Error> {
    use crate::BuiltIn as Bi;
    let Some(&crate::Binding::BuiltIn(built_in)) = binding else {
        return Err(Error::Custom(
            "a ray tracing stage's arguments are builtins".into(),
        ));
    };
    Ok(match built_in {
        Bi::RayInvocationId => "DispatchRaysIndex()",
        Bi::NumRayInvocations => "DispatchRaysDimensions()",
        Bi::InstanceCustomData => "InstanceID()",
        Bi::InstanceIndex => "InstanceIndex()",
        Bi::GeometryIndex => "GeometryIndex()",
        Bi::PrimitiveIndex => "PrimitiveIndex()",
        Bi::WorldRayOrigin => "WorldRayOrigin()",
        Bi::WorldRayDirection => "WorldRayDirection()",
        Bi::ObjectRayOrigin => "ObjectRayOrigin()",
        Bi::ObjectRayDirection => "ObjectRayDirection()",
        Bi::RayTmin => "RayTMin()",
        Bi::RayTCurrentMax => "RayTCurrent()",
        Bi::ObjectToWorld => "ObjectToWorld4x3()",
        Bi::WorldToObject => "WorldToObject4x3()",
        Bi::HitKind => "HitKind()",
        Bi::IncomingRayFlags => "RayFlags()",
        Bi::SubgroupSize => "WaveGetLaneCount()",
        Bi::SubgroupInvocationId => "WaveGetLaneIndex()",
        Bi::HitBarycentrics => return Ok(format!("{ATTRIBUTES_PARAMETER}.barycentrics")),
        other => {
            return Err(Error::Unimplemented(format!(
                "builtin {other:?} in a ray tracing stage"
            )))
        }
    }
    .into())
}
