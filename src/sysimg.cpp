#include <llvm/ADT/SmallSet.h>
#include <llvm/Analysis/CallGraph.h>

#include "sysimg.h"

using namespace llvm;

#define ENABLE_MULTITHREADED_SYSIMG

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
            groups_targets[spec.base].second = false;
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
    //TODO soon to come...
    // push_i32(nshards);
    // push_i32(nfvars);
    // push_i32(ngvars);
    (void)nshards;
    (void)nfvars;
    (void)ngvars;
    push_i32(specs.size());
    for (uint32_t i = 0; i < specs.size(); i++) {
        push_i32(base_flags | (specs[i].flags & JL_TARGET_UNKNOWN_NAME));
        auto &specdata = specs[i].data;
        data.insert(data.end(), specdata.begin(), specdata.end());
    }
    auto value = ConstantDataArray::get(M.getContext(), data);
    auto targets_gv = new GlobalVariable(M, value->getType(), true,
                                    GlobalVariable::ExternalLinkage,
                                    value, "jl_dispatch_target_ids");
    targets_gv->setAlignment(Align(M.getDataLayout().getPrefTypeAlign(Type::getInt32Ty(M.getContext()))));
    addComdat(targets_gv);
}

#ifdef ENABLE_MULTITHREADED_SYSIMG
static void emit_index_table(Module &mod, std::vector<uint32_t> &idxs, StringRef name)
{
    auto table = new GlobalVariable(mod, ArrayType::get(Type::getInt32Ty(mod.getContext()), idxs.size()), true,
                       GlobalVariable::ExternalLinkage,
                       ConstantDataArray::get(mod.getContext(), idxs),
                       name);
    table->setVisibility(GlobalValue::HiddenVisibility);
}
#endif

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

#ifdef ENABLE_MULTITHREADED_SYSIMG
    std::vector<uint32_t> fvars_idxs(data->jl_sysimg_fvars.size());
    for (size_t i = 0; i < data->jl_sysimg_fvars.size(); i++) {
        fvars_idxs[i] = i;
    }
    emit_index_table(M, fvars_idxs, "jl_sysimg_fvars_idxs_1");
    std::vector<uint32_t> gvars_idxs(data->jl_sysimg_gvars.size());
    for (size_t i = 0; i < data->jl_sysimg_gvars.size(); i++) {
        gvars_idxs[i] = i;
    }
    emit_index_table(M, gvars_idxs, "jl_sysimg_gvars_idxs_1");

    M.addModuleFlag(Module::Error, "julia.mv.suffix", MDString::get(M.getContext(), "_1"));
#endif

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
