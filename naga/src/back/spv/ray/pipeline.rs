//! Code for ray tracing pipelines

use alloc::vec;

use crate::back::spv::{
    Block, BlockContext, Instruction, LocalType, LookupRaytracingFunction, Writer, WriterFlags,
};

impl Writer {
    /// The function a trace with `payload` calls; with `motion`, a trace at
    /// a time, its last argument.
    fn write_trace_ray(
        &mut self,
        ir_module: &crate::Module,
        payload: crate::Handle<crate::GlobalVariable>,
        motion: bool,
    ) -> Result<spirv::Word, super::super::Error> {
        let lookup = LookupRaytracingFunction::TraceRay { payload, motion };
        if let Some(&word) = self.ray_tracing_functions.get(&lookup) {
            return Ok(word);
        }
        if motion {
            self.use_extension("SPV_NV_ray_tracing_motion_blur");
            self.require_any(
                "rays traced at a time",
                &[spirv::Capability::RayTracingMotionBlurNV],
            )?;
        }
        self.use_opacity_micromaps_where_available();

        let acceleration_structure_type_id =
            self.get_localtype_id(LocalType::AccelerationStructure);

        let ray_desc_type_id = self.get_handle_type_id(
            ir_module
                .special_types
                .ray_desc
                .expect("ray desc should be set if `traceRays` is called"),
        );

        let u32_type_id = self.get_u32_type_id();
        let f32_type_id = self.get_f32_type_id();
        let mut parameters = vec![
            acceleration_structure_type_id,
            ray_desc_type_id,
            u32_type_id,
            u32_type_id,
            u32_type_id,
        ];
        if motion {
            parameters.push(f32_type_id);
        }
        let (func_id, mut function, arg_ids) =
            self.write_function_signature(&parameters, self.void_type);

        let acceleration_structure_id = arg_ids[0];
        let desc_id = arg_ids[1];
        let payload_id = self.global_variables[payload].access_id;

        let label_id = self.id_gen.next();
        let mut block = Block::new(label_id);

        let (hit_group_offset_id, geometry_stride_id, miss_index_id) =
            self.write_clamped_groups(&mut block, arg_ids[2], arg_ids[3], arg_ids[4]);

        let super::ExtractedRayDesc {
            ray_flags_id,
            cull_mask_id,
            tmin_id,
            tmax_id,
            ray_origin_id,
            ray_dir_id,
            valid_id,
        } = self.write_extract_ray_desc(&mut block, desc_id, self.trace_ray_argument_validation);

        let merge_label_id = self.id_gen.next();
        let merge_block = Block::new(merge_label_id);

        // NOTE: this block will be unreachable if trace ray validation is disabled.
        let invalid_label_id = self.id_gen.next();
        let mut invalid_block = Block::new(invalid_label_id);

        let valid_label_id = self.id_gen.next();
        let mut valid_block = Block::new(valid_label_id);

        match valid_id {
            Some(all_valid_id) => {
                block.body.push(Instruction::selection_merge(
                    merge_label_id,
                    spirv::SelectionControl::NONE,
                ));
                function.consume(
                    block,
                    Instruction::branch_conditional(all_valid_id, valid_label_id, invalid_label_id),
                );
            }
            None => {
                function.consume(block, Instruction::branch(valid_label_id));
            }
        }

        valid_block.body.push(match motion {
            true => Instruction::trace_ray_motion(
                acceleration_structure_id,
                ray_flags_id,
                cull_mask_id,
                hit_group_offset_id,
                geometry_stride_id,
                miss_index_id,
                ray_origin_id,
                tmin_id,
                ray_dir_id,
                tmax_id,
                arg_ids[5],
                payload_id,
            ),
            false => Instruction::trace_ray(
                acceleration_structure_id,
                ray_flags_id,
                cull_mask_id,
                hit_group_offset_id,
                geometry_stride_id,
                miss_index_id,
                ray_origin_id,
                tmin_id,
                ray_dir_id,
                tmax_id,
                payload_id,
            ),
        });

        function.consume(valid_block, Instruction::branch(merge_label_id));

        if self.flags.contains(WriterFlags::PRINT_ON_TRACE_RAYS_FAIL) {
            self.write_debug_printf(
                &mut invalid_block,
                "Naga ignored invalid arguments to traceRay with flags: %u t_min: %f t_max: %f origin: %v4f dir: %v4f",
                &[
                    ray_flags_id,
                    tmin_id,
                    tmax_id,
                    ray_origin_id,
                    ray_dir_id,
                ],
            );
        }

        function.consume(invalid_block, Instruction::branch(merge_label_id));

        function.consume(merge_block, Instruction::return_void());

        function.to_words(&mut self.logical_layout.function_definitions);

        self.ray_tracing_functions.insert(lookup, func_id);

        Ok(func_id)
    }

    /// The hit group offset, the geometry stride and the miss index, kept
    /// inside the pipeline's shader binding table where its counts are
    /// known. The hardware reads only their low bits (4, 4 and 16), which
    /// are taken first, so that what it reads is what was clamped.
    fn write_clamped_groups(
        &mut self,
        block: &mut Block,
        hit_group_offset: spirv::Word,
        geometry_stride: spirv::Word,
        miss_index: spirv::Word,
    ) -> (spirv::Word, spirv::Word, spirv::Word) {
        let Some(groups) = self.ray_tracing_pipeline_groups else {
            return (hit_group_offset, geometry_stride, miss_index);
        };
        let u32_type_id = self.get_u32_type_id();
        let mut clamped = |value: spirv::Word, bits: u32, count: u32| {
            let mask = self.get_constant_scalar(crate::Literal::U32((1 << bits) - 1));
            let last = self.get_constant_scalar(crate::Literal::U32(count.max(1) - 1));
            let masked = self.id_gen.next();
            block.body.push(Instruction::binary(
                spirv::Op::BitwiseAnd,
                u32_type_id,
                masked,
                value,
                mask,
            ));
            let id = self.id_gen.next();
            block.body.push(Instruction::ext_inst_gl_op(
                self.gl450_ext_inst_id,
                spirv::GlslStd450Op::UMin,
                u32_type_id,
                id,
                &[masked, last],
            ));
            id
        };
        (
            clamped(hit_group_offset, 4, groups.hit_groups),
            clamped(geometry_stride, 4, groups.max_geometry_stride + 1),
            clamped(miss_index, 16, groups.misses),
        )
    }

    /// The function an `executeCallable` with `data` calls: it runs the
    /// callable stage at its index, clamped to the pipeline's last, or does
    /// nothing in a pipeline without any.
    fn write_execute_callable(
        &mut self,
        data: crate::Handle<crate::GlobalVariable>,
    ) -> spirv::Word {
        let lookup = LookupRaytracingFunction::ExecuteCallable { data };
        if let Some(&word) = self.ray_tracing_functions.get(&lookup) {
            return word;
        }

        let u32_type_id = self.get_u32_type_id();
        let (func_id, mut function, arg_ids) =
            self.write_function_signature(&[u32_type_id], self.void_type);
        let data_id = self.global_variables[data].access_id;

        let label_id = self.id_gen.next();
        let mut block = Block::new(label_id);
        match self.ray_tracing_pipeline_groups {
            Some(groups) if groups.callables == 0 => {}
            Some(groups) => {
                let last = self.get_constant_scalar(crate::Literal::U32(groups.callables - 1));
                let index_id = self.id_gen.next();
                block.body.push(Instruction::ext_inst_gl_op(
                    self.gl450_ext_inst_id,
                    spirv::GlslStd450Op::UMin,
                    u32_type_id,
                    index_id,
                    &[arg_ids[0], last],
                ));
                block
                    .body
                    .push(Instruction::execute_callable(index_id, data_id));
            }
            None => block
                .body
                .push(Instruction::execute_callable(arg_ids[0], data_id)),
        }
        function.consume(block, Instruction::return_void());
        function.to_words(&mut self.logical_layout.function_definitions);

        self.ray_tracing_functions.insert(lookup, func_id);
        func_id
    }
}

impl BlockContext<'_> {
    pub(in super::super) fn write_ray_tracing_pipeline_function(
        &mut self,
        function: &crate::RayPipelineFunction,
        block: &mut Block,
    ) -> Result<(), super::super::Error> {
        match *function {
            crate::RayPipelineFunction::TraceRay {
                acceleration_structure,
                descriptor,
                payload,
                groups,
                time,
                hit_object,
            } => {
                // Checked for when validating the module in `validate_block_impl`.
                let crate::Expression::GlobalVariable(payload) =
                    self.ir_function.expressions[payload]
                else {
                    unreachable!()
                };

                let desc_id = self.cached[descriptor];
                let acc_struct_id = self.get_handle_id(acceleration_structure);

                // A stride of zero: every geometry of an instance has its
                // hit group.
                let zero = self.writer.get_constant_scalar(crate::Literal::U32(0));
                let (hit_group_offset_id, geometry_stride_id, miss_index_id) = match groups {
                    Some(groups) => (
                        self.cached[groups.hit_group_offset],
                        groups
                            .geometry_stride
                            .map_or(zero, |stride| self.cached[stride]),
                        self.cached[groups.miss_index],
                    ),
                    None => (zero, zero, zero),
                };

                if let Some(hit_object) = hit_object {
                    // Into a hit object: the instruction itself, its
                    // arguments as the helper takes them.
                    let (offset_id, stride_id, miss_id) = self.writer.write_clamped_groups(
                        block,
                        hit_group_offset_id,
                        geometry_stride_id,
                        miss_index_id,
                    );
                    let desc = self.writer.write_extract_ray_desc(block, desc_id, false);
                    self.writer.use_opacity_micromaps_where_available();
                    let mut operands = vec![
                        self.cached[hit_object],
                        acc_struct_id,
                        desc.ray_flags_id,
                        desc.cull_mask_id,
                        offset_id,
                        stride_id,
                        miss_id,
                        desc.ray_origin_id,
                        desc.tmin_id,
                        desc.ray_dir_id,
                        desc.tmax_id,
                    ];
                    let op = match time {
                        Some(time) => {
                            self.writer.use_extension("SPV_NV_ray_tracing_motion_blur");
                            self.writer.require_any(
                                "rays traced at a time",
                                &[spirv::Capability::RayTracingMotionBlurNV],
                            )?;
                            operands.push(self.cached[time]);
                            spirv::Op::HitObjectTraceRayMotionNV
                        }
                        None => spirv::Op::HitObjectTraceRayNV,
                    };
                    operands.push(self.writer.global_variables[payload].access_id);
                    block
                        .body
                        .push(Instruction::hit_object(op, None, &operands));
                    return Ok(());
                }

                let func = self
                    .writer
                    .write_trace_ray(self.ir_module, payload, time.is_some())?;

                let mut arguments = vec![
                    acc_struct_id,
                    desc_id,
                    hit_group_offset_id,
                    geometry_stride_id,
                    miss_index_id,
                ];
                arguments.extend(time.map(|time| self.cached[time]));
                let func_id = self.gen_id();
                block.body.push(Instruction::function_call(
                    self.writer.void_type,
                    func_id,
                    func,
                    &arguments,
                ));
            }
            // Block terminators: written in `write_block`.
            crate::RayPipelineFunction::IgnoreIntersection
            | crate::RayPipelineFunction::TerminateRay => unreachable!(),
            crate::RayPipelineFunction::ReportIntersection {
                hit_t,
                hit_kind,
                result,
            } => {
                let bool_type_id = self.writer.get_bool_type_id();
                let id = self.gen_id();
                block.body.push(Instruction::report_intersection(
                    bool_type_id,
                    id,
                    self.cached[hit_t],
                    self.cached[hit_kind],
                ));
                self.cached[result] = id;
            }
            crate::RayPipelineFunction::ExecuteCallable { index, data } => {
                // Checked for when validating the module in `validate_block_impl`.
                let crate::Expression::GlobalVariable(data) = self.ir_function.expressions[data]
                else {
                    unreachable!()
                };
                let func = self.writer.write_execute_callable(data);
                let func_id = self.gen_id();
                block.body.push(Instruction::function_call(
                    self.writer.void_type,
                    func_id,
                    func,
                    &[self.cached[index]],
                ));
            }
            crate::RayPipelineFunction::HitObjectRecordMiss {
                hit_object,
                miss_index,
                descriptor,
            } => {
                let zero = self.writer.get_constant_scalar(crate::Literal::U32(0));
                let (_, _, miss_id) =
                    self.writer
                        .write_clamped_groups(block, zero, zero, self.cached[miss_index]);
                let desc_id = self.cached[descriptor];
                let desc = self.writer.write_extract_ray_desc(block, desc_id, false);
                block.body.push(Instruction::hit_object(
                    spirv::Op::HitObjectRecordMissNV,
                    None,
                    &[
                        self.cached[hit_object],
                        miss_id,
                        desc.ray_origin_id,
                        desc.tmin_id,
                        desc.ray_dir_id,
                        desc.tmax_id,
                    ],
                ));
            }
            crate::RayPipelineFunction::HitObjectRecordEmpty { hit_object } => {
                block.body.push(Instruction::hit_object(
                    spirv::Op::HitObjectRecordEmptyNV,
                    None,
                    &[self.cached[hit_object]],
                ));
            }
            crate::RayPipelineFunction::HitObjectExecute {
                hit_object,
                payload,
            } => {
                // Checked for when validating the module in `validate_block_impl`.
                let crate::Expression::GlobalVariable(payload) =
                    self.ir_function.expressions[payload]
                else {
                    unreachable!()
                };
                block.body.push(Instruction::hit_object(
                    spirv::Op::HitObjectExecuteShaderNV,
                    None,
                    &[
                        self.cached[hit_object],
                        self.writer.global_variables[payload].access_id,
                    ],
                ));
            }
            crate::RayPipelineFunction::ReorderThread { hit_object, hint } => {
                let hint = hint.map(|hint| [self.cached[hint.hint], self.cached[hint.bits]]);
                let instruction = match hit_object {
                    Some(hit_object) => {
                        let mut operands = vec![self.cached[hit_object]];
                        operands.extend(hint.into_iter().flatten());
                        Instruction::hit_object(
                            spirv::Op::ReorderThreadWithHitObjectNV,
                            None,
                            &operands,
                        )
                    }
                    None => Instruction::hit_object(
                        spirv::Op::ReorderThreadWithHintNV,
                        None,
                        &hint.expect("a reordering by a hint or a hit object"),
                    ),
                };
                block.body.push(instruction);
            }
        }
        Ok(())
    }
}
