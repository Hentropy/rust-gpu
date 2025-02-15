use super::CodegenCx;
use crate::builder_spirv::SpirvValue;
use crate::spirv_type::SpirvType;
use crate::symbols::{parse_attrs, Entry, SpirvAttribute};
use rspirv::dr::Operand;
use rspirv::spirv::{Decoration, ExecutionModel, FunctionControl, StorageClass, Word};
use rustc_hir as hir;
use rustc_middle::{
    mir::terminator::Mutability,
    ty::{layout::HasParamEnv, AdtDef, Instance, Ty, TyKind},
};
use rustc_span::Span;
use rustc_target::abi::{
    call::{ArgAbi, ArgAttribute, ArgAttributes, FnAbi, PassMode},
    Size,
};
use std::collections::HashMap;

impl<'tcx> CodegenCx<'tcx> {
    // Entry points declare their "interface" (all uniforms, inputs, outputs, etc.) as parameters.
    // spir-v uses globals to declare the interface. So, we need to generate a lil stub for the
    // "real" main that collects all those global variables and calls the user-defined main
    // function.
    pub fn entry_stub(
        &self,
        instance: &Instance<'_>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        entry_func: SpirvValue,
        name: String,
        entry: Entry,
    ) {
        let local_id = match instance.def_id().as_local() {
            Some(id) => id,
            None => {
                self.tcx
                    .sess
                    .err(&format!("Cannot declare {} as an entry point", name));
                return;
            }
        };
        let fn_hir_id = self.tcx.hir().local_def_id_to_hir_id(local_id);
        let body = self.tcx.hir().body(self.tcx.hir().body_owned_by(fn_hir_id));
        const EMPTY: ArgAttribute = ArgAttribute::empty();
        for (abi, arg) in fn_abi.args.iter().zip(body.params) {
            if let PassMode::Direct(_) = abi.mode {
            } else if let PassMode::Pair(
                // plain DST/RTA/VLA
                ArgAttributes {
                    pointee_size: Size::ZERO,
                    ..
                },
                ArgAttributes { regular: EMPTY, .. },
            ) = abi.mode
            {
            } else if let PassMode::Pair(
                // DST struct with fields before the DST member
                ArgAttributes { .. },
                ArgAttributes {
                    pointee_size: Size::ZERO,
                    ..
                },
            ) = abi.mode
            {
            } else {
                self.tcx.sess.span_err(
                    arg.span,
                    &format!("PassMode {:?} invalid for entry point parameter", abi.mode),
                )
            }
        }
        if let PassMode::Ignore = fn_abi.ret.mode {
        } else {
            self.tcx.sess.span_err(
                self.tcx.hir().span(fn_hir_id),
                &format!(
                    "PassMode {:?} invalid for entry point return type",
                    fn_abi.ret.mode
                ),
            )
        }
        let execution_model = entry.execution_model;
        let fn_id = if execution_model == ExecutionModel::Kernel {
            self.kernel_entry_stub(entry_func, name, execution_model)
        } else {
            self.shader_entry_stub(
                self.tcx.def_span(instance.def_id()),
                entry_func,
                body.params,
                &fn_abi.args,
                name,
                execution_model,
            )
        };
        let mut emit = self.emit_global();
        entry
            .execution_modes
            .iter()
            .for_each(|(execution_mode, execution_mode_extra)| {
                emit.execution_mode(fn_id, *execution_mode, execution_mode_extra);
            });
    }

    fn shader_entry_stub(
        &self,
        span: Span,
        entry_func: SpirvValue,
        hir_params: &[hir::Param<'tcx>],
        arg_abis: &[ArgAbi<'tcx, Ty<'tcx>>],
        name: String,
        execution_model: ExecutionModel,
    ) -> Word {
        let void = SpirvType::Void.def(span, self);
        let fn_void_void = SpirvType::Function {
            return_type: void,
            arguments: vec![],
        }
        .def(span, self);
        let (entry_func_return_type, entry_func_arg_types) = match self.lookup_type(entry_func.ty) {
            SpirvType::Function {
                return_type,
                arguments,
            } => (return_type, arguments),
            other => self.tcx.sess.fatal(&format!(
                "Invalid entry_stub type: {}",
                other.debug(entry_func.ty, self)
            )),
        };
        let mut decoration_locations = HashMap::new();
        // Create OpVariables before OpFunction so they're global instead of local vars.
        let new_spirv = self.emit_global().version().unwrap() > (1, 3);
        let arg_len = arg_abis.len();
        let mut arguments = Vec::with_capacity(arg_len);
        let mut interface = Vec::with_capacity(arg_len);
        let mut rta_lens = Vec::with_capacity(arg_len / 2);
        let mut arg_types = entry_func_arg_types.iter();
        for (hir_param, arg_abi) in hir_params.iter().zip(arg_abis) {
            // explicit next because there are two args for scalar pairs, but only one param & abi
            let arg_t = *arg_types.next().unwrap_or_else(|| {
                self.tcx.sess.span_fatal(
                    hir_param.span,
                    &format!(
                        "Invalid function arguments: Param {:?} Abi {:?} missing type",
                        hir_param, arg_abi.layout.ty
                    ),
                )
            });
            let (argument, storage_class) =
                self.declare_parameter(arg_t, hir_param, arg_abi, &mut decoration_locations);
            // SPIR-V <= v1.3 only includes Input and Output in the interface.
            if new_spirv
                || storage_class == StorageClass::Input
                || storage_class == StorageClass::Output
            {
                interface.push(argument);
            }
            arguments.push(argument);
            if let SpirvType::Pointer { pointee } = self.lookup_type(arg_t) {
                if let SpirvType::Adt {
                    size: None,
                    field_types,
                    ..
                } = self.lookup_type(pointee)
                {
                    let len_t = *arg_types.next().unwrap_or_else(|| {
                        self.tcx.sess.span_fatal(
                            hir_param.span,
                            &format!(
                                "Invalid function arguments: Param {:?} Abi {:?} fat pointer missing length",
                                hir_param, arg_abi.layout.ty
                            ),
                        )
                    });
                    rta_lens.push((arguments.len() as u32, len_t, field_types.len() as u32 - 1));
                    arguments.push(u32::MAX);
                }
            }
        }
        let mut emit = self.emit_global();
        let fn_id = emit
            .begin_function(void, None, FunctionControl::NONE, fn_void_void)
            .unwrap();
        emit.begin_block(None).unwrap();
        rta_lens.iter().for_each(|&(len_idx, len_t, member_idx)| {
            arguments[len_idx as usize] = emit
                .array_length(len_t, None, arguments[len_idx as usize - 1], member_idx)
                .unwrap()
        });
        emit.function_call(
            entry_func_return_type,
            None,
            entry_func.def_cx(self),
            arguments,
        )
        .unwrap();
        emit.ret().unwrap();
        emit.end_function().unwrap();
        emit.entry_point(execution_model, fn_id, name, interface);
        fn_id
    }

    fn declare_parameter(
        &self,
        arg: Word,
        hir_param: &hir::Param<'tcx>,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        decoration_locations: &mut HashMap<StorageClass, u32>,
    ) -> (Word, StorageClass) {
        let (storage_class, mut spirv_binding) =
            self.get_storage_class(arg_abi).unwrap_or_else(|| {
                self.tcx.sess.span_fatal(
                    hir_param.span,
                    &format!("invalid entry param type `{}`", arg_abi.layout.ty),
                );
            });
        // Note: this *declares* the variable too.
        let variable = self.emit_global().variable(arg, None, storage_class, None);
        if let hir::PatKind::Binding(_, _, ident, _) = &hir_param.pat.kind {
            self.emit_global().name(variable, ident.to_string());
        }
        for attr in parse_attrs(self, self.tcx.hir().attrs(hir_param.hir_id)) {
            match attr {
                SpirvAttribute::Builtin(builtin) => {
                    self.emit_global().decorate(
                        variable,
                        Decoration::BuiltIn,
                        std::iter::once(Operand::BuiltIn(builtin)),
                    );
                    spirv_binding = SpirvBinding::Builtin;
                }
                SpirvAttribute::Flat => {
                    self.emit_global()
                        .decorate(variable, Decoration::Flat, std::iter::empty());
                }
                _ => {}
            }
        }
        match spirv_binding {
            SpirvBinding::DescriptorSet { set, binding } => {
                self.emit_global().decorate(
                    variable,
                    Decoration::DescriptorSet,
                    std::iter::once(Operand::LiteralInt32(set)),
                );
                self.emit_global().decorate(
                    variable,
                    Decoration::Binding,
                    std::iter::once(Operand::LiteralInt32(binding)),
                );
            }
            SpirvBinding::Location(location) => {
                let last_location = decoration_locations.entry(storage_class).or_insert(0);
                if location >= *last_location {
                    *last_location = location + 1;
                } else {
                    self.tcx
                        .sess
                        .span_err(hir_param.span, "Locations must appear in ascending order");
                }
                self.emit_global().decorate(
                    variable,
                    Decoration::Location,
                    std::iter::once(Operand::LiteralInt32(location)),
                );
            }
            SpirvBinding::InferredLocation => {
                // Assign locations from left to right, incrementing each storage class
                // individually.
                // TODO: Is this right for UniformConstant? Do they share locations with
                // input/outpus?
                let location = decoration_locations.entry(storage_class).or_insert(0);
                self.emit_global().decorate(
                    variable,
                    Decoration::Location,
                    std::iter::once(Operand::LiteralInt32(*location)),
                );
                *location += 1;
            }
            _ => {}
        }
        (variable, storage_class)
    }

    fn get_storage_class(
        &self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
    ) -> Option<(StorageClass, SpirvBinding)> {
        let (adt, substs) = match arg_abi.layout.ty.kind() {
            TyKind::Adt(adt, substs) => (adt, substs),
            TyKind::Ref(_, _, Mutability::Not) => {
                return Some((StorageClass::Input, SpirvBinding::InferredLocation))
            }
            TyKind::Ref(_, _, Mutability::Mut) => {
                return Some((StorageClass::Output, SpirvBinding::InferredLocation))
            }
            _ => return None,
        };
        for attr in parse_attrs(self, self.tcx.get_attrs(adt.did)) {
            match attr {
                SpirvAttribute::StorageClass(StorageClass::Output) => {
                    let mut consts = substs.consts();
                    return if let (Some(location), None) = (consts.next(), consts.next()) {
                        Some((
                            StorageClass::Output,
                            SpirvBinding::Location(
                                location.eval_usize(self.tcx, self.param_env()) as u32
                            ),
                        ))
                    } else {
                        None
                    };
                }
                SpirvAttribute::StorageClass(StorageClass::Input) => {
                    let mut consts = substs.consts();
                    return if let (Some(location), None) = (consts.next(), consts.next()) {
                        Some((
                            StorageClass::Input,
                            SpirvBinding::Location(
                                location.eval_usize(self.tcx, self.param_env()) as u32
                            ),
                        ))
                    } else {
                        None
                    };
                }
                SpirvAttribute::StorageClass(StorageClass::PushConstant) => {
                    return Some((StorageClass::PushConstant, SpirvBinding::PushConstant))
                }
                SpirvAttribute::Bind => {
                    let parse_storage_class_attr = |adt: &AdtDef| {
                        for attr in parse_attrs(self, self.tcx.get_attrs(adt.did)) {
                            if let SpirvAttribute::StorageClass(storage_class) = attr {
                                return Some(storage_class);
                            }
                        }
                        None
                    };
                    let descriptor_set = {
                        let mut consts = substs.consts();
                        if let (Some(set), Some(binding), None) =
                            (consts.next(), consts.next(), consts.next())
                        {
                            SpirvBinding::DescriptorSet {
                                set: set.eval_usize(self.tcx, self.param_env()) as u32,
                                binding: binding.eval_usize(self.tcx, self.param_env()) as u32,
                            }
                        } else {
                            return None;
                        }
                    };
                    match substs.types().next().unwrap().kind() {
                        TyKind::Adt(adt, _) => {
                            if let Some(storage_class) = parse_storage_class_attr(adt) {
                                return Some((storage_class, descriptor_set));
                            }
                        }
                        TyKind::Slice(ty) => {
                            if let TyKind::Adt(adt, _) = ty.kind() {
                                if let Some(storage_class) = parse_storage_class_attr(adt) {
                                    return Some((storage_class, descriptor_set));
                                }
                            }
                        }
                        TyKind::Array(ty, _) => {
                            if let TyKind::Adt(adt, _) = ty.kind() {
                                if let Some(storage_class) = parse_storage_class_attr(adt) {
                                    return Some((storage_class, descriptor_set));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        None
    }

    // Kernel mode takes its interface as function parameters(??)
    // OpEntryPoints cannot be OpLinkage, so write out a stub to call through.
    fn kernel_entry_stub(
        &self,
        entry_func: SpirvValue,
        name: String,
        execution_model: ExecutionModel,
    ) -> Word {
        let (entry_func_return, entry_func_args) = match self.lookup_type(entry_func.ty) {
            SpirvType::Function {
                return_type,
                arguments,
            } => (return_type, arguments),
            other => self.tcx.sess.fatal(&format!(
                "Invalid kernel_entry_stub type: {}",
                other.debug(entry_func.ty, self)
            )),
        };
        let mut emit = self.emit_global();
        let fn_id = emit
            .begin_function(
                entry_func_return,
                None,
                FunctionControl::NONE,
                entry_func.ty,
            )
            .unwrap();
        let arguments = entry_func_args
            .iter()
            .map(|&ty| emit.function_parameter(ty).unwrap())
            .collect::<Vec<_>>();
        emit.begin_block(None).unwrap();
        let call_result = emit
            .function_call(entry_func_return, None, entry_func.def_cx(self), arguments)
            .unwrap();
        if self.lookup_type(entry_func_return) == SpirvType::Void {
            emit.ret().unwrap();
        } else {
            emit.ret_value(call_result).unwrap();
        }
        emit.end_function().unwrap();

        emit.entry_point(execution_model, fn_id, name, &[]);
        fn_id
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum SpirvBinding {
    DescriptorSet { set: u32, binding: u32 },
    Location(u32),
    InferredLocation,
    Builtin,
    PushConstant,
}
