#include "llvm-version.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/Analysis/CallGraph.h>
#if JL_LLVM_VERSION >= 140000
#include <llvm/MC/TargetRegistry.h>
#else
#include <llvm/Support/TargetRegistry.h>
#endif
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Bitcode/BitcodeWriterPass.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#include <llvm/Passes/PassBuilder.h>

#include "sysimg.h"

using namespace llvm;

// #define ENABLE_MULTITHREADED_SYSIMG

extern Optional<bool> always_have_fma(Function&);

extern Optional<bool> always_have_fp16();

void addComdat(GlobalValue *G)
{
    if (!G->isDeclaration()) {
        // add __declspec(dllexport) to everything marked for export
        if (G->hasHiddenVisibility()) {
            G->setDLLStorageClass(GlobalValue::DefaultStorageClass);
        } else {
            switch (G->getLinkage()) {
            case GlobalValue::LinkOnceAnyLinkage:
            case GlobalValue::LinkOnceODRLinkage:
            case GlobalValue::WeakAnyLinkage:
            case GlobalValue::WeakODRLinkage:
            case GlobalValue::ExternalLinkage:
            case GlobalValue::AvailableExternallyLinkage:
            case GlobalValue::CommonLinkage:
            case GlobalValue::ExternalWeakLinkage:
                G->setDLLStorageClass(GlobalValue::DLLExportStorageClass);
                break;
            case GlobalValue::InternalLinkage:
            case GlobalValue::PrivateLinkage:
            case GlobalValue::AppendingLinkage:
                G->setDLLStorageClass(GlobalValue::DefaultStorageClass);
                break;
            }
        }
    }
}

static bool is_vector(FunctionType *ty)
{
    if (ty->getReturnType()->isVectorTy())
        return true;
    for (auto arg: ty->params()) {
        if (arg->isVectorTy()) {
            return true;
        }
    }
    return false;
}

constexpr uint32_t clone_mask =
    JL_TARGET_CLONE_LOOP | JL_TARGET_CLONE_SIMD | JL_TARGET_CLONE_MATH | JL_TARGET_CLONE_CPU;

static void annotate_reloc_slot(Function &F, const BitVector &cloned_groups, ArrayRef<std::pair<SmallVector<unsigned, 4>, bool>> groups_targets) {
    BitVector target_reloc(groups_targets.size());
    auto add_reloc_attr = [&target_reloc, &F]() {
        std::string relocs_str;
        for (int i = target_reloc.find_first(); i >= 0; i = target_reloc.find_next(i)) {
            relocs_str += std::to_string(i) + ",";
        }
        relocs_str.pop_back();
        F.addFnAttr("julia.mv.reloc", relocs_str);
    };
    for (auto uses = ConstantUses<GlobalValue>(&F, *F.getParent()); !uses.done(); uses.next()) {
        auto info = uses.get_info();
        if (isa<GlobalAlias>(info.val)) {
            //Global aliases always require a relocation slot everywhere
            target_reloc.clear();
            target_reloc.resize(groups_targets.size(), true);
            add_reloc_attr();
            return;
        }
    }
    SmallSet<Function *, 16> seen;
    BitVector use_clones(groups_targets.size());
    for (auto uses = ConstantUses<Instruction>(&F, *F.getParent()); !uses.done(); uses.next()) {
        auto info = uses.get_info();
        auto use_f = info.val->getFunction();
        if (seen.count(use_f))
            continue;
        //we've already counted the relevant clones
        seen.insert(use_f);
        // if we use this function in a function that is in the same group,
        // we need a relocation slot
        // if (!use_f->getName().endswith(suffix))
        //     continue;
        if (use_f->hasFnAttribute("julia.mv.clones")) {
            use_clones.clear();
            use_clones.resize(groups_targets.size(), false);
            auto idxs = use_f->getFnAttribute("julia.mv.clones").getValueAsString();
            do {
                uint32_t idx = 0;
                bool present = idxs.consumeInteger(10, idx);
                assert(!present);
                (void) present;
                use_clones.set(idx);
            } while (idxs.consume_front(","));
            for (int i = use_clones.find_first(); i >= 0; i = use_clones.find_next(i)) {
                if (!cloned_groups.test(i))
                    continue;
                //we should hit groups before hitting any targets
                assert(groups_targets[i].second && "Function was cloned in target but not in base clone_all!");
                target_reloc.set(i);
                for (auto &tgt: groups_targets[i].first) {
                    if (!use_clones[tgt]) {
                        target_reloc.set(tgt);
                    } else {
                        //won't consider it again, don't even iterate to it
                        use_clones.reset(tgt);
                    }
                }
            }
        }
    }
    if (target_reloc.any()) {
        add_reloc_attr();
    }
}


static uint32_t collect_func_info(Function &F, bool has_loop, bool &has_veccall)
{
    uint32_t flag = 0;
    if (has_loop)
        flag |= JL_TARGET_CLONE_LOOP;
    if (is_vector(F.getFunctionType())) {
        flag |= JL_TARGET_CLONE_SIMD;
        has_veccall = true;
    }
    for (auto &bb: F) {
        for (auto &I: bb) {
            if (auto call = dyn_cast<CallInst>(&I)) {
                if (is_vector(call->getFunctionType())) {
                    has_veccall = true;
                    flag |= JL_TARGET_CLONE_SIMD;
                }
                if (auto callee = call->getCalledFunction()) {
                    auto name = callee->getName();
                    if (name.startswith("llvm.muladd.") || name.startswith("llvm.fma.")) {
                        flag |= JL_TARGET_CLONE_MATH;
                    }
                    else if (name.startswith("julia.cpu.")) {
                        if (name.startswith("julia.cpu.have_fma.")) {
                            // for some platforms we know they always do (or don't) support
                            // FMA. in those cases we don't need to clone the function.
                            if (!always_have_fma(*callee).hasValue())
                                flag |= JL_TARGET_CLONE_CPU;
                        } else {
                            flag |= JL_TARGET_CLONE_CPU;
                        }
                    }
                }
            }
            else if (auto store = dyn_cast<StoreInst>(&I)) {
                if (store->getValueOperand()->getType()->isVectorTy()) {
                    flag |= JL_TARGET_CLONE_SIMD;
                }
            }
            else if (I.getType()->isVectorTy()) {
                flag |= JL_TARGET_CLONE_SIMD;
            }
            if (auto mathOp = dyn_cast<FPMathOperator>(&I)) {
                if (mathOp->getFastMathFlags().any()) {
                    flag |= JL_TARGET_CLONE_MATH;
                }
            }
            if(!always_have_fp16().hasValue()){
                for (size_t i = 0; i < I.getNumOperands(); i++) {
                    if(I.getOperand(i)->getType()->isHalfTy()){
                        flag |= JL_TARGET_CLONE_FLOAT16;
                    }
                    // Check for BFloat16 when they are added to julia can be done here
                }
            }
            if (has_veccall && (flag & JL_TARGET_CLONE_SIMD) && (flag & JL_TARGET_CLONE_MATH)) {
                return flag;
            }
        }
    }
    return flag;
}

static void annotate_clones(Module &M) {
    std::vector<jl_target_spec_t> specs = jl_get_llvm_clone_targets();
    SmallVector<std::pair<SmallVector<unsigned, 4>, bool>, 4> groups_targets(specs.size());
    SmallVector<unsigned> groups;
    for (unsigned i = 0; i < specs.size(); i++) {
        auto &spec = specs[i];
        if (i == 0 || (spec.flags & JL_TARGET_CLONE_ALL)) {
            groups.push_back(i);
            groups_targets[i].second = true;
        } else {
            groups_targets[spec.base].first.push_back(i);
        }
    }
    CallGraph graph(M);
    std::vector<Function *> orig_funcs;
    for (auto &F : M.functions()) {
        if (F.empty())
            continue;
        orig_funcs.push_back(&F);
    }
    bool has_veccall = false;
    std::vector<uint32_t> func_infos(orig_funcs.size());
    std::vector<SmallVector<uint32_t, 1>> clones(orig_funcs.size());
    for (size_t i = 0; i < orig_funcs.size(); i++) {
        if (groups.size() > 1) {
            for (uint32_t gid = 1; gid < groups.size(); gid++) {
                clones[i].push_back(groups[gid]);
            }
        }
        DominatorTree DT(*orig_funcs[i]);
        LoopInfo LI(DT);
        func_infos[i] = collect_func_info(*orig_funcs[i], !LI.empty(), has_veccall);
    }
    for (auto &grp : groups_targets) {
        if (!grp.second)
            continue;
        for (auto &tgt : grp.first) {
            auto flag = specs[tgt].flags & clone_mask;
            auto suffix = ".clone_" + std::to_string(tgt);
            uint32_t nfuncs = func_infos.size();

            std::set<Function*> all_origs;
            // Use a simple heuristic to decide which function we need to clone.
            for (uint32_t i = 0; i < nfuncs; i++) {
                if (!(func_infos[i] & flag))
                    continue;
                all_origs.insert(orig_funcs[i]);
                clones[i].push_back(tgt);
            }
            std::set<Function*> sets[2]{all_origs, std::set<Function*>{}};
            auto *cur_set = &sets[0];
            auto *next_set = &sets[1];
            // Reduce dispatch by expand the cloning set to functions that are directly called by
            // and calling cloned functions.
            while (!cur_set->empty()) {
                for (auto orig_f: *cur_set) {
                    // Use the uncloned function since it's already in the call graph
                    auto node = graph[orig_f];
                    for (const auto &I: *node) {
                        auto child_node = I.second;
                        auto orig_child_f = child_node->getFunction();
                        if (!orig_child_f)
                            continue;
                        // Already cloned
                        if (all_origs.count(orig_child_f))
                            continue;
                        bool calling_clone = false;
                        for (const auto &I2: *child_node) {
                            auto orig_child_f2 = I2.second->getFunction();
                            if (!orig_child_f2)
                                continue;
                            if (all_origs.count(orig_child_f2)) {
                                calling_clone = true;
                                break;
                            }
                        }
                        if (!calling_clone)
                            continue;
                        next_set->insert(orig_child_f);
                        all_origs.insert(orig_child_f);
                    }
                }
                std::swap(cur_set, next_set);
                next_set->clear();
            }
            for (uint32_t i = 0; i < nfuncs; i++) {
                // Only need to handle expanded functions
                if (func_infos[i] & flag)
                    continue;
                if (all_origs.count(orig_funcs[i])) {
                    clones[i].push_back(tgt);
                }
            }
        }
    }
    for (size_t i = 0; i < orig_funcs.size(); i++) {
        if (!ConstantUses<GlobalValue>(orig_funcs[i], M).done())
            orig_funcs[i]->addFnAttr("julia.mv.fvar", "");
        if (clones[i].empty())
            continue;
        auto *F = orig_funcs[i];
        std::string val = std::to_string(clones[i][0]);
        for (size_t j = 1; j < clones[i].size(); j++) {
            val += "," + std::to_string(clones[i][j]);
        }
        F->addFnAttr("julia.mv.clones", val);
    }
    if (groups.size() > 1) {
        //We only have a base_func if we have cloneall
        BitVector cloned_groups(specs.size());
        for (auto &grp : groups) {
            cloned_groups.set(grp);
        }
        cloned_groups.reset(0);
        for (auto F : orig_funcs) {
            annotate_reloc_slot(*F, cloned_groups, groups_targets);
        }
    }
    if (has_veccall) {
        M.addModuleFlag(Module::Max, "julia.mv.veccall", 1);
    }
}

void add_sysimage_targets(Module &M, bool has_veccall, uint32_t nshards, uint32_t nfvars, uint32_t ngvars) {
    // Generate `jl_dispatch_target_ids`
    auto specs = jl_get_llvm_clone_targets();
    const uint32_t base_flags = has_veccall ? JL_TARGET_VEC_CALL : 0;
    std::vector<uint8_t> data;
    auto push_i32 = [&] (uint32_t v) {
        uint8_t buff[4];
        memcpy(buff, &v, 4);
        data.insert(data.end(), buff, buff + 4);
    };
    static_assert(sizeof(jl_sysimg_metadata_t) == 3 * sizeof(uint32_t), "jl_sysimg_metadata_t size mismatch");
    push_i32(nshards);
    push_i32(nfvars);
    push_i32(ngvars);
    push_i32(specs.size());
    for (uint32_t i = 0; i < specs.size(); i++) {
        push_i32(base_flags | (specs[i].flags & JL_TARGET_UNKNOWN_NAME));
        auto &specdata = specs[i].data;
        data.insert(data.end(), specdata.begin(), specdata.end());
    }
    auto value = ConstantDataArray::get(M.getContext(), data);
    auto targets_gv = new GlobalVariable(M, value->getType(), true,
                                    GlobalVariable::ExternalLinkage,
                                    value, "jl_sysimg_metadata");
    targets_gv->setAlignment(Align(M.getDataLayout().getPrefTypeAlign(Type::getInt32Ty(M.getContext()))));
    addComdat(targets_gv);

    {
        auto T_psize = M.getDataLayout().getIntPtrType(Type::getInt8PtrTy(M.getContext()))->getPointerTo();
        auto T_int32 = Type::getInt32PtrTy(M.getContext());
        // Generate shard table
        std::vector<Constant*> shard_metadatas(nshards * sizeof(jl_sysimg_shard_t) / sizeof(int32_t*));
        for (uint32_t i = 0; i < nshards; i++) {
            //Manually constructed dual of jl_sysimg_shard_metadata_t
            auto fvars_base = new GlobalVariable(M, T_psize, true, GlobalValue::ExternalLinkage,
                                                nullptr, "jl_sysimg_fvars_base_" + std::to_string(i+1));
            auto gvars_base = new GlobalVariable(M, T_psize, true, GlobalValue::ExternalLinkage,
                                                nullptr, "jl_sysimg_gvars_base_" + std::to_string(i+1));
            // We label the following as T_int32, but they're actually arrays of int32s
            // Since the module is not actually linked by LLVM with the data module,
            // we don't need to worry about the type mismatch
            auto fvars_offsets = new GlobalVariable(M, T_int32, true, GlobalValue::ExternalLinkage,
                                                nullptr, "jl_sysimg_fvars_offsets_" + std::to_string(i+1));
            auto fvars_idxs = new GlobalVariable(M, T_int32, true, GlobalValue::ExternalLinkage,
                                                nullptr, "jl_sysimg_fvars_idxs_" + std::to_string(i+1));
            auto gvars_offsets = new GlobalVariable(M, T_int32, true, GlobalValue::ExternalLinkage,
                                                nullptr, "jl_sysimg_gvars_offsets_" + std::to_string(i+1));
            auto gvars_idxs = new GlobalVariable(M, T_int32, true, GlobalValue::ExternalLinkage,
                                                nullptr, "jl_sysimg_gvars_idxs_" + std::to_string(i+1));
            auto dispatch_offsets = new GlobalVariable(M, T_int32, true, GlobalValue::ExternalLinkage,
                                                nullptr, "jl_dispatch_fvars_offsets_" + std::to_string(i+1));
            auto dispatch_idxs = new GlobalVariable(M, T_int32, true, GlobalValue::ExternalLinkage,
                                                nullptr, "jl_dispatch_fvars_idxs_" + std::to_string(i+1));
            auto dispatch_relocs = new GlobalVariable(M, T_int32, true, GlobalValue::ExternalLinkage,
                                                nullptr, "jl_dispatch_reloc_slots_" + std::to_string(i+1));
            //Add them all to shard_metadatas
            // Note that this must match jl_sysimg_shard_t ordering exactly
            auto offset = i * sizeof(jl_sysimg_shard_t) / sizeof(int32_t*);
            shard_metadatas[offsetof(jl_sysimg_shard_t, fbase) / sizeof(int32_t*) + offset] = ConstantExpr::getBitCast(fvars_base, T_psize);
            shard_metadatas[offsetof(jl_sysimg_shard_t, gbase) / sizeof(int32_t*) + offset] = ConstantExpr::getBitCast(gvars_base, T_psize);
            shard_metadatas[offsetof(jl_sysimg_shard_t, foffsets) / sizeof(int32_t*) + offset] = ConstantExpr::getBitCast(fvars_offsets, T_psize);
            shard_metadatas[offsetof(jl_sysimg_shard_t, fidxs) / sizeof(int32_t*) + offset] = ConstantExpr::getBitCast(fvars_idxs, T_psize);
            shard_metadatas[offsetof(jl_sysimg_shard_t, goffsets) / sizeof(int32_t*) + offset] = ConstantExpr::getBitCast(gvars_offsets, T_psize);
            shard_metadatas[offsetof(jl_sysimg_shard_t, gidxs) / sizeof(int32_t*) + offset] = ConstantExpr::getBitCast(gvars_idxs, T_psize);
            shard_metadatas[offsetof(jl_sysimg_shard_t, coffsets) / sizeof(int32_t*) + offset] = ConstantExpr::getBitCast(dispatch_offsets, T_psize);
            shard_metadatas[offsetof(jl_sysimg_shard_t, cidxs) / sizeof(int32_t*) + offset] = ConstantExpr::getBitCast(dispatch_idxs, T_psize);
            shard_metadatas[offsetof(jl_sysimg_shard_t, relocs) / sizeof(int32_t*) + offset] = ConstantExpr::getBitCast(dispatch_relocs, T_psize);
        }

        static_assert(sizeof(jl_sysimg_shard_t) == 9 * sizeof(void*), "jl_sysimg_shard_metadata_t has changed");

        dbgs() << "shard_metadatas.size() = " << shard_metadatas.size() << "\n";

        addComdat(new GlobalVariable(M, ArrayType::get(T_psize, shard_metadatas.size()), true,
                           GlobalVariable::ExternalLinkage,
                           ConstantArray::get(ArrayType::get(T_psize, shard_metadatas.size()), shard_metadatas),
                           "jl_sysimg_shards"));
    }
}

static void emit_index_table(Module &mod, std::vector<uint32_t> &idxs, StringRef name)
{
    auto table = new GlobalVariable(mod, ArrayType::get(Type::getInt32Ty(mod.getContext()), idxs.size()), true,
                       GlobalVariable::ExternalLinkage,
                       ConstantDataArray::get(mod.getContext(), idxs),
                       name);
    table->setVisibility(GlobalValue::HiddenVisibility);
}

static void emit_offset_table(Module &mod, const std::vector<GlobalValue*> &vars, StringRef name, Type *T_psize)
{
    // Emit a global variable with all the variable addresses.
    // The cloning pass will convert them into offsets.
    assert(!vars.empty());
    size_t nvars = vars.size();
    std::vector<Constant*> addrs(nvars);
    for (size_t i = 0; i < nvars; i++) {
        Constant *var = vars[i];
        addrs[i] = ConstantExpr::getBitCast(var, T_psize);
    }
    ArrayType *vars_type = ArrayType::get(T_psize, nvars);
    new GlobalVariable(mod, vars_type, true,
                       GlobalVariable::ExternalLinkage,
                       ConstantArray::get(vars_type, addrs),
                       name);
}

void add_sysimage_globals(Module &M, jl_native_code_desc_t *data) {
    annotate_clones(M);
    DenseSet<GlobalValue *> fvars(data->jl_sysimg_fvars.begin(), data->jl_sysimg_fvars.end());
    //Add some more fvars that were detected during the pseudocloning process
    for (auto &F : M) {
        if (F.isDeclaration())
            continue;
        if (F.hasFnAttribute("julia.mv.reloc") || F.hasFnAttribute("julia.mv.fvar")) {
            if (fvars.insert(&F).second) {
                data->jl_sysimg_fvars.push_back(&F);
            }
        }
    }
    auto T_psize = M.getDataLayout().getIntPtrType(M.getContext())->getPointerTo();
    emit_offset_table(M, data->jl_sysimg_gvars, "jl_sysimg_gvars", T_psize);
    emit_offset_table(M, data->jl_sysimg_fvars, "jl_sysimg_fvars", T_psize);

    std::vector<uint32_t> fvars_idxs(data->jl_sysimg_fvars.size());
    for (size_t i = 0; i < data->jl_sysimg_fvars.size(); i++) {
        fvars_idxs[i] = i;
    }
    emit_index_table(M, fvars_idxs, "jl_sysimg_fvars_idxs");
    std::vector<uint32_t> gvars_idxs(data->jl_sysimg_gvars.size());
    for (size_t i = 0; i < data->jl_sysimg_gvars.size(); i++) {
        gvars_idxs[i] = i;
    }
    emit_index_table(M, gvars_idxs, "jl_sysimg_gvars_idxs");

    // Multiversioning will suffix all the gvs with this
    M.addModuleFlag(Module::Error, "julia.mv.suffix", MDString::get(M.getContext(), "_1"));

    // reflect the address of the jl_RTLD_DEFAULT_handle variable
    // back to the caller, so that we can check for consistency issues
    GlobalValue *jlRTLD_DEFAULT_var = jl_emit_RTLD_DEFAULT_var(&M);
    addComdat(new GlobalVariable(M,
                                    jlRTLD_DEFAULT_var->getType(),
                                    true,
                                    GlobalVariable::ExternalLinkage,
                                    jlRTLD_DEFAULT_var,
                                    "jl_RTLD_DEFAULT_handle_pointer"));
}

static void injectCRTAlias(Module &M, StringRef name, StringRef alias, FunctionType *FT)
{
    Function *target = M.getFunction(alias);
    if (!target) {
        target = Function::Create(FT, Function::ExternalLinkage, alias, M);
    }
    Function *interposer = Function::Create(FT, Function::WeakAnyLinkage, name, M);
    appendToCompilerUsed(M, {interposer});

    llvm::IRBuilder<> builder(BasicBlock::Create(M.getContext(), "top", interposer));
    SmallVector<Value *, 4> CallArgs;
    for (auto &arg : interposer->args())
        CallArgs.push_back(&arg);
    auto val = builder.CreateCall(target, CallArgs);
    builder.CreateRet(val);
}

static std::unique_ptr<TargetMachine> createTM(TargetMachine &Orig) {
    return std::unique_ptr<TargetMachine>(
                                            Orig.getTarget().createTargetMachine(
                                                Orig.getTargetTriple().getTriple(),
                                                Orig.getTargetCPU(),
                                                Orig.getTargetFeatureString(),
                                                Orig.Options,
                                                Orig.getRelocationModel(),
                                                Orig.getCodeModel(),
                                                Orig.getOptLevel()));
}

static void add_output_impl(TargetMachine &DumpTM, Module &M,
                std::vector<std::string> &outputs,
                DumpOutput unopt, DumpOutput opt, DumpOutput obj, DumpOutput assm,
                bool inject_crt, unsigned outputs_offset,
                unsigned unopt_offset, unsigned opt_offset, unsigned obj_offset, unsigned assm_offset) {
    uint64_t start = 0;
    uint64_t end = 0;
    assert(!verifyModule(M, &errs()));

    auto &Context = M.getContext();

    auto TM = createTM(DumpTM);

    std::array<std::string, 4> bufs;

    if (!unopt.Name.empty()) {
        SmallVector<char, 0> unopt_bc_Buffer;
        raw_svector_ostream unopt_bc_OS(unopt_bc_Buffer);
        ModulePassManager MPM;
        MPM.addPass(BitcodeWriterPass(unopt_bc_OS));
        PassBuilder PB;
        AnalysisManagers AM(*TM, PB, OptimizationLevel::O0);
        MPM.run(M, AM.MAM);
        bufs[0] = { unopt_bc_Buffer.data(), unopt_bc_Buffer.size() };
    }
    if (opt.Name.empty() && obj.Name.empty() && assm.Name.empty())
        return;

    #ifndef JL_USE_NEW_PM
        legacy::PassManager optimizer;
        addTargetPasses(&optimizer, TM->getTargetTriple(), TM->getTargetIRAnalysis());
        addOptimizationPasses(&optimizer, jl_options.opt_level, true, true);
        addMachinePasses(&optimizer, jl_options.opt_level);
    #else
        NewPM optimizer{std::move(TM), getOptLevel(jl_options.opt_level), OptimizationOptions::defaults(true, true)};
    #endif
    start = jl_hrtime();
    optimizer.run(M);
    end = jl_hrtime();
    dbgs() << "Optimization time: " << (end - start) / 1e9 << "s\n";

    assert(!verifyModule(M, &errs()));

    if (inject_crt) {
        // We would like to emit an alias or an weakref alias to redirect these symbols
        // but LLVM doesn't let us emit a GlobalAlias to a declaration...
        // So for now we inject a definition of these functions that calls our runtime
        // functions. We do so after optimization to avoid cloning these functions.
        injectCRTAlias(M, "__gnu_h2f_ieee", "julia__gnu_h2f_ieee",
                FunctionType::get(Type::getFloatTy(Context), { Type::getHalfTy(Context) }, false));
        injectCRTAlias(M, "__extendhfsf2", "julia__gnu_h2f_ieee",
                FunctionType::get(Type::getFloatTy(Context), { Type::getHalfTy(Context) }, false));
        injectCRTAlias(M, "__gnu_f2h_ieee", "julia__gnu_f2h_ieee",
                FunctionType::get(Type::getHalfTy(Context), { Type::getFloatTy(Context) }, false));
        injectCRTAlias(M, "__truncsfhf2", "julia__gnu_f2h_ieee",
                FunctionType::get(Type::getHalfTy(Context), { Type::getFloatTy(Context) }, false));
        injectCRTAlias(M, "__truncdfhf2", "julia__truncdfhf2",
                FunctionType::get(Type::getHalfTy(Context), { Type::getDoubleTy(Context) }, false));
    }

    if (!opt.Name.empty()) {
        SmallVector<char, 0> opt_bc_Buffer;
        raw_svector_ostream opt_bc_OS(opt_bc_Buffer);
        ModulePassManager MPM;
        MPM.addPass(BitcodeWriterPass(opt_bc_OS));
        PassBuilder PB;
        AnalysisManagers AM(*TM, PB, OptimizationLevel::O0);
        MPM.run(M, AM.MAM);
        bufs[1] = { opt_bc_Buffer.data(), opt_bc_Buffer.size() };
    }

    if (!obj.Name.empty()) {
        SmallVector<char, 0> obj_Buffer;
        raw_svector_ostream obj_OS(obj_Buffer);
        legacy::PassManager emitter;
        addTargetPasses(&emitter, TM->getTargetTriple(), TM->getTargetIRAnalysis());
        if (TM->addPassesToEmitFile(emitter, obj_OS, nullptr, CGFT_ObjectFile, false))
            jl_safe_printf("ERROR: target does not support generation of object files\n");
        start = jl_hrtime();
        emitter.run(M);
        end = jl_hrtime();
        dbgs() << "Object emission time: " << (end - start) / 1e9 << "s\n";
        bufs[2] = { obj_Buffer.data(), obj_Buffer.size() };
    }

    if (!assm.Name.empty()) {
        SmallVector<char, 0> asm_Buffer;
        raw_svector_ostream asm_OS(asm_Buffer);
        legacy::PassManager emitter;
        addTargetPasses(&emitter, TM->getTargetTriple(), TM->getTargetIRAnalysis());
        if (TM->addPassesToEmitFile(emitter, asm_OS, nullptr, CGFT_AssemblyFile, false))
            jl_safe_printf("ERROR: target does not support generation of assembly files\n");
        start = jl_hrtime();
        emitter.run(M);
        end = jl_hrtime();
        dbgs() << "Assembly emission time: " << (end - start) / 1e9 << "s\n";
        bufs[3] = { asm_Buffer.data(), asm_Buffer.size() };
    }

    if (!unopt.Name.empty()) {
        outputs[outputs_offset] = std::move(bufs[0]);
        unopt.Archive[unopt_offset] = NewArchiveMember(MemoryBufferRef(outputs[outputs_offset], unopt.Name));
        outputs_offset++;
    }
    if (!opt.Name.empty()) {
        outputs[outputs_offset] = std::move(bufs[1]);
        opt.Archive[opt_offset] = NewArchiveMember(MemoryBufferRef(outputs[outputs_offset], opt.Name));
        outputs_offset++;
    }
    if (!obj.Name.empty()) {
        outputs[outputs_offset] = std::move(bufs[2]);
        obj.Archive[obj_offset] = NewArchiveMember(MemoryBufferRef(outputs[outputs_offset], obj.Name));
        outputs_offset++;
    }
    if (!assm.Name.empty()) {
        outputs[outputs_offset] = std::move(bufs[3]);
        assm.Archive[assm_offset] = NewArchiveMember(MemoryBufferRef(outputs[outputs_offset], assm.Name));
        outputs_offset++;
    }
}

static size_t getFunctionWeight(const Function &F)
{
    size_t weight = 1;
    for (const BasicBlock &BB : F) {
        weight += BB.size();
    }
    // more basic blocks = more complex than just sum of insts,
    // add some weight to it
    weight += F.size();
    if (F.hasFnAttribute("julia.mv.clones")) {
        weight *= F.getFnAttribute("julia.mv.clones").getValueAsString().count(',') + 1;
    }
    return weight;
}

struct DisjointSetPartitioner {
    struct Node {
        GlobalValue *GV;
        size_t weight;
        unsigned parent;
    };

    SmallVector<Node> nodes;
    DenseMap<GlobalValue *, unsigned> nodeMap;
    unsigned roots;

    void add(GlobalValue *GV) {
        unsigned idx = nodes.size();
        nodeMap[GV] = idx;
        nodes.push_back({GV, 0, idx});
        if (auto F = dyn_cast<Function>(GV)) {
            nodes.back().weight = getFunctionWeight(*F);
        } else {
            nodes.back().weight = 1;
        }
        ++roots;
    }

    unsigned root(unsigned i) {
        unsigned j = i;
        while (nodes[j].parent != j) {
            j = nodes[j].parent;
        }
        while (nodes[i].parent != j) {
            unsigned k = nodes[i].parent;
            nodes[i].parent = j;
            i = k;
        }
        return j;
    }

    void merge(unsigned i, unsigned j) {
        i = root(i);
        j = root(j);
        if (i == j)
            return;
        nodes[j].parent = i;
        nodes[i].weight += nodes[j].weight;
        --roots;
    }
};

struct Partition {
    StringSet<> GVNames;
    StringMap<uint32_t> fvars;
    StringMap<uint32_t> gvars;
};

bool verify_partitioning(const SmallVectorImpl<Partition> &partitions, const Module &M) {
    StringMap<uint32_t> GVNames;
    bool bad = false;
    for (uint32_t i = 0; i < partitions.size(); i++) {
        for (auto &name : partitions[i].GVNames) {
            if (GVNames.count(name.getKey())) {
                bad = true;
                dbgs() << "Duplicate global name " << name.getKey() << " in partitions " << i << " and " << GVNames[name.getKey()] << "\n";
            }
            GVNames[name.getKey()] = i;
        }
        dbgs() << "partition: " << i << " fvars: " << partitions[i].fvars.size() << " gvars: " << partitions[i].gvars.size() << "\n";
    }
    for (auto &GV : M.globals()) {
        if (GV.isDeclaration()) {
            if (GVNames.count(GV.getName())) {
                bad = true;
                dbgs() << "Global " << GV.getName() << " is a declaration but is in partition " << GVNames[GV.getName()] << "\n";
            }
        } else {
            if (!GVNames.count(GV.getName())) {
                bad = true;
                dbgs() << "Global " << GV << " not in any partition\n";
            }
        }
    }
    return !bad;
}

//We can't have functions that are defined in one module but whose address
//is used in a global variable initializer in another module, and likewise
//for global variables. Thus the partitioning algorithm must take this into
//account. We also would like a relatively even split of the burden of
//optimizing between modules, so we use basic block count as a heuristic.
static SmallVector<Partition, 16> partitionModule(Module &M, unsigned ways) {
    //Start by stripping fvars and gvars, which helpfully removes their uses as well
    DenseMap<GlobalValue *, uint32_t> fvars;
    DenseMap<GlobalValue *, uint32_t> gvars;
    {
        auto fvars_gv = M.getGlobalVariable("jl_sysimg_fvars");
        auto gvars_gv = M.getGlobalVariable("jl_sysimg_gvars");
        assert(fvars_gv);
        assert(gvars_gv);
        auto fvars_init = cast<ConstantArray>(fvars_gv->getInitializer());
        auto gvars_init = cast<ConstantArray>(gvars_gv->getInitializer());
        std::string suffix;
        if (auto md = M.getModuleFlag("julia.mv.suffix")) {
            suffix = cast<MDString>(md)->getString().str();
        }
        auto fvars_idxs = M.getGlobalVariable("jl_sysimg_fvars_idxs");
        auto gvars_idxs = M.getGlobalVariable("jl_sysimg_gvars_idxs");
        assert(fvars_idxs);
        assert(gvars_idxs);
        auto fvars_idxs_init = cast<ConstantDataArray>(fvars_idxs->getInitializer());
        auto gvars_idxs_init = cast<ConstantDataArray>(gvars_idxs->getInitializer());
        for (unsigned i = 0; i < fvars_init->getNumOperands(); ++i) {
            auto gv = cast<GlobalValue>(fvars_init->getOperand(i)->stripPointerCasts());
            auto idx = fvars_idxs_init->getElementAsInteger(i);
            fvars[gv] = idx;
        }
        for (unsigned i = 0; i < gvars_init->getNumOperands(); ++i) {
            auto gv = cast<GlobalValue>(gvars_init->getOperand(i)->stripPointerCasts());
            auto idx = gvars_idxs_init->getElementAsInteger(i);
            gvars[gv] = idx;
        }
        fvars_gv->eraseFromParent();
        gvars_gv->eraseFromParent();
        fvars_idxs->eraseFromParent();
        gvars_idxs->eraseFromParent();
    }
    DisjointSetPartitioner partitioner;
    int inc = 0;
    // everything is partionable
    for (auto &GV : M.global_values()) {
        if (GV.isDeclaration())
            continue;
        if (!GV.hasName()) {
            GV.setName("jl_ext_" + std::to_string(inc++));
        }
        partitioner.add(&GV);
    }
    // these uses must go together
    for (auto node : partitioner.nodes) {
        for (auto uses = ConstantUses<GlobalValue>(node.GV, M); !uses.done(); uses.next()) {
            auto info = uses.get_info();
            assert(isa<GlobalVariable>(info.val) || isa<GlobalAlias>(info.val));
            partitioner.merge(node.parent, partitioner.nodeMap[info.val]);
        }
    }
    SmallVector<Partition, 16> partitions(ways);
    typedef std::pair<size_t, Partition*> Processor;
    std::priority_queue<Processor, std::vector<Processor>, std::greater<Processor>> pq;
    auto push_partition([&](Partition &p, DisjointSetPartitioner::Node &node) {
        p.GVNames.insert(node.GV->getName());
        if (fvars.count(node.GV))
            p.fvars[node.GV->getName()] = fvars[node.GV];
        if (gvars.count(node.GV))
            p.gvars[node.GV->getName()] = gvars[node.GV];
        node.weight = &p - partitions.data();
        node.GV = nullptr;
    });
    unsigned way = 0;
    //Can't make the 1 fvar + 1 gvar assumption anymore
    // but try to split initial load evenly
    // // Require >= 1 fvar per partition
    for (auto &fvar : fvars) {
        auto i = partitioner.nodeMap[fvar.first];
        auto &node = partitioner.nodes[i];
        if (!node.GV)
            continue;
        if (node.parent != i) {
            continue;
        }
        auto &p = partitions[way];
        auto weight = node.weight;
        push_partition(p, node);
        pq.push(Processor(weight, &p));
        if (++way == ways) {
            break;
        }
    }
    // assert(way == ways);
    way = 0;
    // //Require >= 1 gvar per partition
    for (auto &gvar : gvars) {
        auto i = partitioner.nodeMap[gvar.first];
        auto &node = partitioner.nodes[i];
        if (node.parent != i) {
            continue;
        }
        auto &p = partitions[way];
        auto weight = node.weight;
        push_partition(p, node);
        pq.push(Processor(weight, &p));
        if (++way == ways) {
            break;
        }
    }
    // assert(way == ways);
    typedef std::pair<size_t, unsigned> WorkUnit;
    std::vector<WorkUnit> roots;
    roots.reserve(partitioner.roots);
    // Assign root workloads in reverse sorted order
    // (reduces chance of unlucky partitioning)
    for (unsigned i = 0; i < partitioner.nodes.size(); i++) {
        auto &node = partitioner.nodes[i];
        if (node.parent != i)
            continue;
        roots.push_back(WorkUnit(node.weight, i));
    }
    std::sort(roots.begin(), roots.end(), std::greater<WorkUnit>());
    for (auto &root : roots) {
        auto &node = partitioner.nodes[root.second];
        if (!node.GV)
            continue;
        auto q = pq.top();
        pq.pop();
        q.first += node.weight;
        push_partition(*q.second, node);
        pq.push(q);
    }
    // Assign non-root nodes to their root's partition
    for (unsigned i = 0; i < partitioner.nodes.size(); i++) {
        auto &node = partitioner.nodes[i];
        if (!node.GV)
            continue;
        if (node.parent != i) {
            auto &root = partitioner.nodes[partitioner.root(node.parent)];
            assert(!root.GV);
            push_partition(partitions[root.weight], node);
        } else {
            assert(!node.GV);
        }
    }
    assert(verify_partitioning(partitions, M));
    return partitions;
}

static auto deserializeModule(const SmallVector<char, 0> &Bitcode, LLVMContext &Ctx) {
    auto M = cantFail(getLazyBitcodeModule(MemoryBufferRef(StringRef(Bitcode.data(), Bitcode.size()), "Optimized"), Ctx), "Error loading module");
    // assert(!verifyModule(*M, &errs()) && "Module verification failed");
    return M;
}

static auto serializeModule(const Module &M) {
    SmallVector<char, 0> ClonedModuleBuffer;
    BitcodeWriter BCWriter(ClonedModuleBuffer);
    BCWriter.writeModule(M);
    BCWriter.writeSymtab();
    BCWriter.writeStrtab();
    return ClonedModuleBuffer;
}

void add_output(TargetMachine &DumpTM, Module &M,
                std::vector<std::string> &outputs,
                DumpOutput unopt, DumpOutput opt, DumpOutput obj, DumpOutput assm, bool inject_crt, unsigned threads) {
    assert(threads > 0);
    if (threads == 1) {
        outputs.resize(outputs.size() + (!unopt.Name.empty() + !opt.Name.empty() + !obj.Name.empty() + !assm.Name.empty()));
        unopt.Archive.resize(unopt.Archive.size() + !unopt.Name.empty());
        opt.Archive.resize(opt.Archive.size() + !opt.Name.empty());
        obj.Archive.resize(obj.Archive.size() + !obj.Name.empty());
        assm.Archive.resize(assm.Archive.size() + !assm.Name.empty());
        add_output_impl(DumpTM, M, outputs, unopt, opt, obj, assm, inject_crt,
            outputs.size() - 1, unopt.Archive.size() - 1, opt.Archive.size() - 1, obj.Archive.size() - 1, assm.Archive.size() - 1);
        return;
    }
    uint64_t start = 0;
    uint64_t end = 0;
    start = jl_hrtime();
    auto partitions = partitionModule(M, threads);
    end = jl_hrtime();
    dbgs() << "Partitioning time: " << (end - start) / 1e9 << "s\n";
    start = jl_hrtime();
    auto serialized = serializeModule(M);
    end = jl_hrtime();
    dbgs() << "Serialization time: " << (end - start) / 1e9 << "s\n";
    std::vector<std::thread> workers(threads);
    unsigned output_offset = outputs.size();
    unsigned output_inc = !unopt.Name.empty() + !opt.Name.empty() + !obj.Name.empty() + !assm.Name.empty();
    outputs.resize(output_offset + output_inc * threads);
    unsigned unopt_offset = unopt.Archive.size();
    unopt.Archive.resize(unopt_offset + !unopt.Name.empty() * threads);
    unsigned opt_offset = opt.Archive.size();
    opt.Archive.resize(opt_offset + !opt.Name.empty() * threads);
    unsigned obj_offset = obj.Archive.size();
    obj.Archive.resize(obj_offset + !obj.Name.empty() * threads);
    unsigned assm_offset = assm.Archive.size();
    assm.Archive.resize(assm_offset + !assm.Name.empty() * threads);
    start = jl_hrtime();
    for (unsigned i = 0; i < threads; ++i) {
        workers[i] = std::thread([&, i]() {
            LLVMContext ctx;
            uint64_t start = 0;
            uint64_t end = 0;
            start = jl_hrtime();
            auto M = deserializeModule(serialized, ctx);
            end = jl_hrtime();
            dbgs() << "Deserialization time: " << (end - start) / 1e9 << "s\n";
            dbgs() << "Starting shard " << i << "\n";
            DenseSet<GlobalValue *> Preserve;
            for (auto &GV : M->global_values()) {
                if (!GV.isDeclaration()) {
                    if (partitions[i].GVNames.count(GV.getName())) {
                        Preserve.insert(&GV);
                    }
                }
            }
            start = jl_hrtime();
            for (auto &F : M->functions()) {
                if (!F.isDeclaration()) {
                    if (!Preserve.contains(&F)) {
                        F.deleteBody();
                        F.setLinkage(GlobalValue::ExternalLinkage);
                    }
                }
            }
            for (auto &GV : M->globals()) {
                if (!GV.isDeclaration()) {
                    if (!Preserve.contains(&GV)) {
                        GV.setInitializer(nullptr);
                        GV.setLinkage(GlobalValue::ExternalLinkage);
                    }
                }
            }
            SmallVector<std::pair<GlobalAlias *, GlobalValue *>> DeletedAliases;
            for (auto &GA : M->aliases()) {
                if (!GA.isDeclaration()) {
                    if (!Preserve.contains(&GA)) {
                        if (GA.getValueType()->isFunctionTy()) {
                            DeletedAliases.push_back({ &GA, Function::Create(cast<FunctionType>(GA.getValueType()), GlobalValue::ExternalLinkage, "", M.get()) });
                        } else {
                            DeletedAliases.push_back({ &GA, new GlobalVariable(*M, GA.getValueType(), false, GlobalValue::ExternalLinkage, nullptr) });
                        }
                    }
                }
            }
            cantFail(M->materializeAll());
            for (auto &Deleted : DeletedAliases) {
                Deleted.second->takeName(Deleted.first);
                Deleted.first->replaceAllUsesWith(Deleted.second);
                Deleted.first->eraseFromParent();
            }
            end = jl_hrtime();
            dbgs() << "Clone time: " << (end - start) / 1e9 << "s\n";
            //Restore fvars, fvars_idxs, gvars, gvars_idxs
            std::vector<std::pair<uint32_t, GlobalValue *>> fvar_pairs;
            fvar_pairs.reserve(partitions[i].fvars.size());
            for (auto &fvar : partitions[i].fvars) {
                auto F = M->getFunction(fvar.first());
                assert(F);
                assert(!F->isDeclaration());
                fvar_pairs.push_back({ fvar.second, F });
            }
            std::vector<GlobalValue *> fvars;
            std::vector<uint32_t> fvar_idxs;
            fvars.reserve(fvar_pairs.size());
            fvar_idxs.reserve(fvar_pairs.size());
            std::sort(fvar_pairs.begin(), fvar_pairs.end());
            for (auto &fvar : fvar_pairs) {
                fvars.push_back(fvar.second);
                fvar_idxs.push_back(fvar.first);
            }
            std::vector<std::pair<uint32_t, GlobalValue *>> gvar_pairs;
            gvar_pairs.reserve(partitions[i].gvars.size());
            for (auto &gvar : partitions[i].gvars) {
                auto GV = M->getGlobalVariable(gvar.first());
                assert(GV);
                assert(!GV->isDeclaration());
                gvar_pairs.push_back({ gvar.second, GV });
            }
            std::vector<GlobalValue *> gvars;
            std::vector<uint32_t> gvar_idxs;
            gvars.reserve(gvar_pairs.size());
            gvar_idxs.reserve(gvar_pairs.size());
            std::sort(gvar_pairs.begin(), gvar_pairs.end());
            for (auto &gvar : gvar_pairs) {
                gvars.push_back(gvar.second);
                gvar_idxs.push_back(gvar.first);
            }
            auto T_psize = M->getDataLayout().getIntPtrType(M->getContext())->getPointerTo();
            emit_offset_table(*M, fvars, "jl_sysimg_fvars", T_psize);
            emit_offset_table(*M, gvars, "jl_sysimg_gvars", T_psize);
            emit_index_table(*M, fvar_idxs, "jl_sysimg_fvars_idxs");
            emit_index_table(*M, gvar_idxs, "jl_sysimg_gvars_idxs");
            M->setModuleFlag(Module::Error, "julia.mv.suffix", MDString::get(M->getContext(), "_" + std::to_string(i+1)));
            //Drop all of the unused declarations; multiversioning won't need them
            SmallVector<GlobalValue *> unused;
            for (auto &G : M->global_values()) {
                if (G.isDeclaration()) {
                    if (G.use_empty()) {
                        unused.push_back(&G);
                    } else {
                        G.setDSOLocal(false); // These are never going to be seen in the same module again
                        G.setVisibility(GlobalValue::DefaultVisibility);
                    }
                }
            }
            for (auto &G : unused)
                G->eraseFromParent();
            add_output_impl(DumpTM, *M, outputs, unopt, opt, obj, assm, inject_crt,
                            output_inc * i + output_offset, unopt_offset + i, opt_offset + i, obj_offset + i, assm_offset + i);
        });
    }
    end = jl_hrtime();
    dbgs() << "Shard setup time: " << (end - start) / 1e9 << "s\n";
    start = jl_hrtime();
    for (auto &w : workers)
        w.join();
    end = jl_hrtime();
    dbgs() << "Shard waiting time: " << (end - start) / 1e9 << "s\n";
}
