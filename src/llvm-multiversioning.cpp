// This file is a part of Julia. License is MIT: https://julialang.org/license

// Function multi-versioning
// LLVM pass to clone function for different archs

#include "llvm-version.h"
#include "passes.h"

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include <llvm/Pass.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include "julia.h"
#include "julia_internal.h"
#include "support/dtypes.h"
#include "sysimg.h"

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "codegen_shared.h"
#include "julia_assert.h"

#define DEBUG_TYPE "julia_multiversioning"
#undef DEBUG

using namespace llvm;

namespace {

// Treat identical mapping as missing and return `def` in that case.
// We mainly need this to identify cloned function using value map after LLVM cloning
// functions fills the map with identity entries.
template<typename T>
Value *map_get(T &&vmap, Value *key, Value *def=nullptr)
{
    auto val = vmap.lookup(key);
    if (!val || key == val)
        return def;
    return val;
}

struct CloneCtx {
    struct Target {
        int idx;
        uint32_t flags;
        std::unique_ptr<ValueToValueMapTy> vmap; // ValueToValueMapTy is not movable....
        // function ids that needs relocation to be initialized
        std::set<uint32_t> relocs{};
        Target(int idx, const jl_target_spec_t &spec) :
            idx(idx),
            flags(spec.flags),
            vmap(new ValueToValueMapTy)
        {
        }
    };
    struct Group : Target {
        std::vector<Target> clones;
        std::set<uint32_t> clone_fs;
        Group(int base, const jl_target_spec_t &spec) :
            Target(base, spec),
            clones{},
            clone_fs{}
        {}
        Function *base_func(Function *orig_f) const
        {
            if (idx == 0)
                return orig_f;
            return cast<Function>(vmap->lookup(orig_f));
        }
    };
    CloneCtx(Module &M, bool allow_bad_fvars);
    void prepare_relocs();
    void clone_decls();
    void clone_bases();
    void clone_all_partials();
    void fix_gv_uses();
    void fix_inst_uses();
    void emit_metadata();
private:
    void init_reloc_slot(Function *F);
    void prepare_vmap(ValueToValueMapTy &vmap);
    void clone_function(Function *F, Function *new_f, ValueToValueMapTy &vmap);
    void clone_partial(Group &grp, Target &tgt);
    uint32_t get_func_id(Function *F);
    std::pair<uint32_t,GlobalVariable*> get_reloc_slot(Function *F);
    void rewrite_alias(GlobalAlias *alias, Function* F);

    MDNode *tbaa_const;
    std::vector<jl_target_spec_t> specs;
    std::vector<Group> groups{};
    SmallVector<std::pair<Group *, Target *>> linearized;
    std::vector<Function*> fvars;
    std::vector<Constant*> gvars;
    Module &M;

    // Map from original function to one based index in `fvars`
    std::map<const Function*,uint32_t> func_ids{};
    std::vector<Function*> orig_funcs{};
    std::set<Function*> cloned{};
    // GV addresses and their corresponding function id (i.e. 0-based index in `fvars`)
    std::vector<std::pair<Constant*,uint32_t>> gv_relocs{};
    // Mapping from function id (i.e. 0-based index in `fvars`) to GVs to be initialized.
    std::map<uint32_t,GlobalVariable*> const_relocs;
    // Functions that were referred to by a global alias, and might not have other uses.
    std::set<uint32_t> alias_relocs;
    // Relocation that will be resolved in a different shard
    DenseMap<Function *, GlobalVariable *> extern_relocs{};
    std::string gv_suffix;
    bool has_veccall{false};
    bool allow_bad_fvars{false};
};

template<typename T>
static inline std::vector<T*> consume_gv(Module &M, const char *name, bool allow_bad_fvars)
{
    // Get information about sysimg export functions from the two global variables.
    // Strip them from the Module so that it's easier to handle the uses.
    GlobalVariable *gv = M.getGlobalVariable(name);
    assert(gv && gv->hasInitializer());
    ArrayType *Ty = cast<ArrayType>(gv->getInitializer()->getType());
    unsigned nele = Ty->getArrayNumElements();
    std::vector<T*> res(nele);
    ConstantArray *ary = nullptr;
    if (gv->getInitializer()->isNullValue()) {
        for (unsigned i = 0; i < nele; ++i)
            res[i] = cast<T>(Constant::getNullValue(Ty->getArrayElementType()));
    }
    else {
        ary = cast<ConstantArray>(gv->getInitializer());
        unsigned i = 0;
        while (i < nele) {
            llvm::Value *val = ary->getOperand(i)->stripPointerCasts();
            if (allow_bad_fvars && (!isa<T>(val) || (isa<Function>(val) && cast<Function>(val)->isDeclaration()))) {
                // Shouldn't happen in regular use, but can happen in bugpoint.
                nele--;
                continue;
            }
            res[i++] = cast<T>(val);
        }
        res.resize(nele);
    }
    assert(gv->use_empty());
    gv->eraseFromParent();
    if (ary && ary->use_empty())
        ary->destroyConstant();
    return res;
}

static bool verify_reloc_slots(ConstantDataArray &idxs, ConstantArray &reloc_slots, bool clone_all, uint32_t start, uint32_t end) {
    bool bad = false;
    auto nreloc_val = cast<ConstantInt>(reloc_slots.getOperand(0));
    auto nreloc = nreloc_val->getZExtValue();
    if (nreloc * 2 + 1 != reloc_slots.getNumOperands()) {
        bad = true;
        dbgs() << "ERROR: Bad reloc slot count\n";
    }
    uint32_t reloc_i = 0;
    for (uint32_t i = start; i < end; i++) {
        auto idx = idxs.getElementAsInteger(i);
        if (!clone_all) {
            if (idx & jl_sysimg_tag_mask) {
                idx &= jl_sysimg_val_mask;
            } else {
                continue;
            }
        }
        bool found = false;
        //This is for error recovery
        uint32_t prev_i = reloc_i;
        for (; reloc_i < nreloc; reloc_i++) {
            //we add 1 here to skip the length (processor just bumps the source pointer)
            auto reloc_idx = cast<ConstantInt>(reloc_slots.getOperand(reloc_i * 2 + 1))->getZExtValue();
            if (reloc_idx == idx) {
                found = true;
                // dbgs() << "Found reloc slot for " << idx << "\n";
            }
            else if (reloc_idx > idx) {
                break;
            }
        }
        if (!found) {
            bad = true;
            dbgs() << "ERROR: Missing reloc slot for " << idx << "\n";
            // try to recover to find more errors
            reloc_i = prev_i;
        }
    }
    // if (!bad) {
    //     dbgs() << "Verified " << end - start << " reloc slots\n";
    // }
    return bad;
}

//This mimics the logic during sysimage loading, so we fail earlier
//if multiversioning doesn't fit what the loader expects
// currently just checks reloc_slots against dispatch_fvars_idxs
static bool verify_multiversioning(Module &M) {
    std::string suffix;
    if (auto suffix_md = M.getModuleFlag("julia.mv.suffix")) {
        suffix = cast<MDString>(suffix_md)->getString().str();
    }
    bool bad = false;
    auto reloc_slots = M.getGlobalVariable("jl_dispatch_reloc_slots" + suffix);
    auto clone_idxs = M.getGlobalVariable("jl_dispatch_fvars_idxs" + suffix);
    if (!reloc_slots) {
        dbgs() << "ERROR: Missing jl_dispatch_reloc_slots" << suffix << "\n";
        bad = true;
    }
    if (!clone_idxs) {
        dbgs() << "ERROR: Missing jl_dispatch_fvars_idxs" << suffix << "\n";
        bad = true;
    }
    auto cidxs = dyn_cast<ConstantDataArray>(clone_idxs->getInitializer());
    if (!cidxs) {
        dbgs() << "ERROR: jl_dispatch_fvars_idxs" << suffix << " is not a constant data array\n";
        bad = true;
    }
    auto rslots = dyn_cast<ConstantArray>(reloc_slots->getInitializer());
    if (!rslots) {
        if (!isa<ConstantAggregateZero>(reloc_slots->getInitializer())) {
            dbgs() << "ERROR: jl_dispatch_reloc_slots" << suffix << " is not a constant array\n";
            dbgs() << *reloc_slots->getInitializer() << "\n";
            bad = true;
        }
    }
    if (cidxs && rslots) {
        auto specs = jl_get_llvm_clone_targets();
        uint32_t tag_len = cidxs->getElementAsInteger(0);
        uint32_t offset = 1;
        for (uint32_t i = 0; i < specs.size(); i++) {
            uint32_t len = jl_sysimg_val_mask & tag_len;
            bool clone_all = (tag_len & jl_sysimg_tag_mask);
            if (clone_all != (i == 0 || specs[i].flags & JL_TARGET_CLONE_ALL)) {
                dbgs() << "ERROR: clone_all mismatch for spec " << i << "\n";
                dbgs() << "  " << clone_all << " != " << (i == 0 || specs[i].flags & JL_TARGET_CLONE_ALL) << "\n";
                bad = true;
            }
            bad |= verify_reloc_slots(*cidxs, *rslots, clone_all, offset + !clone_all, offset + len);
            if (jl_sysimg_tag_mask & tag_len) {
                offset += len + 1;
            } else {
                offset += len + 2;
            }
            if (i != specs.size() - 1) {
                if (offset > cidxs->getNumElements()) {
                    dbgs() << "ERROR: out of bounds cloneidxs length " << i + 1 << "\n";
                    bad = true;
                }
                tag_len = cidxs->getElementAsInteger(offset - 1);
            }
        }
    }
    return bad;
}

// Collect basic information about targets and functions.
CloneCtx::CloneCtx(Module &M, bool allow_bad_fvars)
    : tbaa_const(tbaa_make_child_with_context(M.getContext(), "jtbaa_const", nullptr, true).first),
      specs(jl_get_llvm_clone_targets()),
      fvars(consume_gv<Function>(M, "jl_sysimg_fvars", allow_bad_fvars)),
      gvars(consume_gv<Constant>(M, "jl_sysimg_gvars", false)),
      M(M),
      allow_bad_fvars(allow_bad_fvars)
{
    groups.emplace_back(0, specs[0]);
    uint32_t ntargets = specs.size();
    for (uint32_t i = 1; i < ntargets; i++) {
        auto &spec = specs[i];
        if (spec.flags & JL_TARGET_CLONE_ALL) {
            groups.emplace_back(i, spec);
        }
        else {
            auto base = spec.base;
            bool found = false;
            for (auto &grp: groups) {
                if (grp.idx == base) {
                    found = true;
                    grp.clones.emplace_back(i, spec);
                    break;
                }
            }
            (void)found;
        }
    }
    linearized.resize(specs.size());
    for (auto &grp: groups) {
        linearized[grp.idx] = {&grp, nullptr};
        for (auto &clone: grp.clones) {
            linearized[clone.idx] = {&grp, &clone};
        }
    }
    uint32_t nfvars = fvars.size();
    for (uint32_t i = 0; i < nfvars; i++)
        func_ids[fvars[i]] = i + 1;
    for (auto &F: M) {
        if (F.empty() && !F.hasFnAttribute("julia.mv.clones"))
            continue;
        orig_funcs.push_back(&F);
    }
    if (auto metadata = M.getModuleFlag("julia.mv.veccall")) {
        has_veccall = cast<ConstantAsMetadata>(metadata)->getValue()->isOneValue();
    }
    if (auto md = M.getModuleFlag("julia.mv.suffix")) {
        gv_suffix = cast<MDString>(md)->getString().str();
    }
}

void CloneCtx::prepare_relocs() {

    for (size_t i = 0; i < orig_funcs.size(); i++) {
        auto &F = *orig_funcs[i];
        if (F.hasFnAttribute("julia.mv.reloc")) {
            // Only track relocations for actual definitions
            // if annotated with `julia.mv.reloc`,  then it's definitely an fvar
            // but if it's a declaration, another shard will handle the relocation
            if (!F.isDeclaration()) {
                auto val = F.getFnAttribute("julia.mv.reloc").getValueAsString();
                uint32_t idx = 0;
                do {
                    bool present = val.consumeInteger(10, idx);
                    assert(!present);
                    (void) present;
                    auto id = get_func_id(&F);
                    // dbgs() << "Adding fvar " << id << " to target " << idx << "\n";
                    if (linearized[idx].second) {
                        linearized[idx].second->relocs.insert(id);
                    } else {
                        linearized[idx].first->relocs.insert(id);
                    }
                } while (val.consume_front(","));
            }
            init_reloc_slot(&F);
        }
    }
}

void CloneCtx::init_reloc_slot(Function *F) {
    auto gvar = new GlobalVariable(M, F->getType(), false, GlobalVariable::ExternalLinkage,
                                F->isDeclaration() ? nullptr : ConstantPointerNull::get(F->getType()),
                                F->getName() + ".reloc_slot");
    gvar->setVisibility(GlobalValue::HiddenVisibility);
    if (F->isDeclaration()) {
        auto &slot = extern_relocs[F];
        assert(!slot);
        slot = gvar;
        return;
    }
    auto id = get_func_id(F);
    // dbgs() << "Adding reloc slot at " << id << "\n";
    auto &slot = const_relocs[id];
    assert(!slot);
    slot = gvar;
}

void CloneCtx::prepare_vmap(ValueToValueMapTy &vmap)
{
    // Workaround LLVM `CloneFunctionInfo` bug (?) pre-5.0
    // The `DICompileUnit`s are being cloned but are not added to the `llvm.dbg.cu` metadata
    // which triggers assertions when generating native code/in the verifier.
    // Fix this by forcing an identical mapping for all `DICompileUnit` recorded.
    // The `DISubprogram` cloning on LLVM 5.0 handles this
    // but it doesn't hurt to enforce the identity either.
    auto &MD = vmap.MD();
    for (auto cu: M.debug_compile_units()) {
        MD[cu].reset(cu);
    }
}

void CloneCtx::clone_function(Function *F, Function *new_f, ValueToValueMapTy &vmap)
{
    // Don't actually clone declarations
    if (F->isDeclaration())
        return;
    Function::arg_iterator DestI = new_f->arg_begin();
    for (Function::const_arg_iterator J = F->arg_begin(); J != F->arg_end(); ++J) {
        DestI->setName(J->getName());
        vmap[&*J] = &*DestI++;
    }
    SmallVector<ReturnInst*,8> Returns;
#if JL_LLVM_VERSION >= 130000
    // We are cloning into the same module
    CloneFunctionInto(new_f, F, vmap, CloneFunctionChangeType::GlobalChanges, Returns);
#else
    CloneFunctionInto(new_f, F, vmap, true, Returns);
#endif
}

void CloneCtx::clone_decls()
{
    std::vector<std::string> suffixes(specs.size());
    for (uint32_t i = 1; i < specs.size(); i++) {
        suffixes[i] = ".clone_" + i;
    }
    for (size_t i = 0; i < orig_funcs.size(); i++) {
        if (!orig_funcs[i]->hasFnAttribute("julia.mv.clones"))
            continue;
        auto clones = orig_funcs[i]->getFnAttribute("julia.mv.clones").getValueAsString();
        uint32_t idx = 0;
        do {
            bool isidx = clones.consumeInteger(10, idx);
            assert(!isidx);
            (void) isidx;
            //Clone the group functions first so base_func works
            if (!linearized[idx].second) {
                auto F = orig_funcs[i];
                auto &vmap = *linearized[idx].first->vmap;
                // Fill in old->new mapping. We need to do this before cloning the function so that
                // the intra target calls are automatically fixed up on cloning.
                Function *new_f = Function::Create(F->getFunctionType(), F->getLinkage(),
                                                F->getName() + suffixes[idx], &M);
                new_f->setVisibility(F->getVisibility());
                new_f->copyAttributesFrom(F);
                vmap[F] = new_f;
            }
        } while (clones.consume_front(","));
    }
    for (size_t i = 0; i < orig_funcs.size(); i++) {
        if (!orig_funcs[i]->hasFnAttribute("julia.mv.clones"))
            continue;
        auto clones = orig_funcs[i]->getFnAttribute("julia.mv.clones").getValueAsString();
        uint32_t idx = 0;
        do {
            bool isidx = clones.consumeInteger(10, idx);
            assert(!isidx);
            (void) isidx;
            //Clone the partial functions
            if (linearized[idx].second) {
                // Fill in old->new mapping. We need to do this before cloning the function so that
                // the intra target calls are automatically fixed up on cloning.
                auto orig_f = orig_funcs[i];
                auto F = linearized[idx].first->base_func(orig_f);
                auto &vmap = *linearized[idx].second->vmap;
                Function *new_f = Function::Create(F->getFunctionType(), F->getLinkage(),
                                                F->getName() + suffixes[idx], &M);
                new_f->setVisibility(F->getVisibility());
                new_f->copyAttributesFrom(F);
                vmap[F] = new_f;
                if (groups.size() == 1)
                    cloned.insert(orig_f);
                linearized[idx].first->clone_fs.insert(i);
            }
        } while (clones.consume_front(","));
    }
}

// Clone all clone_all targets. Makes sure that the base targets are all available.
void CloneCtx::clone_bases()
{
    if (groups.size() == 1)
        return;
    uint32_t ngrps = groups.size();
    for (uint32_t gid = 1; gid < ngrps; gid++) {
        auto &vmap = *groups[gid].vmap;
        prepare_vmap(vmap);
        for (auto F: orig_funcs) {
            clone_function(F, cast<Function>(vmap.lookup(F)), vmap);
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

static void add_features(Function *F, StringRef name, StringRef features, uint32_t flags)
{
    auto attr = F->getFnAttribute("target-features");
    if (attr.isStringAttribute()) {
        std::string new_features(attr.getValueAsString());
        new_features += ",";
        new_features += features;
        F->addFnAttr("target-features", new_features);
    }
    else {
        F->addFnAttr("target-features", features);
    }
    F->addFnAttr("target-cpu", name);
    if (!F->hasFnAttribute(Attribute::OptimizeNone)) {
        if (flags & JL_TARGET_OPTSIZE) {
            F->addFnAttr(Attribute::OptimizeForSize);
        }
        else if (flags & JL_TARGET_MINSIZE) {
            F->addFnAttr(Attribute::MinSize);
        }
    }
}

void CloneCtx::clone_all_partials()
{
    for (auto &grp: groups) {
        for (auto &tgt: grp.clones)
            clone_partial(grp, tgt);
        // Also set feature strings for base target functions
        // now that all the actual cloning is done.
        auto &base_spec = specs[grp.idx];
        for (auto orig_f: orig_funcs) {
            add_features(grp.base_func(orig_f), base_spec.cpu_name,
                         base_spec.cpu_features, base_spec.flags);
        }
    }
}

void CloneCtx::clone_partial(Group &grp, Target &tgt)
{
    auto &spec = specs[tgt.idx];
    auto &vmap = *tgt.vmap;
    uint32_t nfuncs = orig_funcs.size();
    prepare_vmap(vmap);
    for (uint32_t i = 0; i < nfuncs; i++) {
        auto orig_f = orig_funcs[i];
        auto F = grp.base_func(orig_f);
        if (auto new_v = map_get(vmap, F)) {
            auto new_f = cast<Function>(new_v);
            assert(new_f != F);
            clone_function(F, new_f, vmap);
            // We can set the feature strings now since no one is going to
            // clone these functions again.
            add_features(new_f, spec.cpu_name, spec.cpu_features, spec.flags);
        }
    }
}

uint32_t CloneCtx::get_func_id(Function *F)
{
    assert(!F->isDeclaration());
    auto &ref = func_ids[F];
    assert(ref);
    return ref - 1;
}

template<typename Stack>
static Constant *rewrite_gv_init(const Stack& stack)
{
    // Null initialize so that LLVM put it in the correct section.
    SmallVector<Constant*, 8> args;
    Constant *res = ConstantPointerNull::get(cast<PointerType>(stack[0].val->getType()));
    uint32_t nlevel = stack.size();
    for (uint32_t i = 1; i < nlevel; i++) {
        auto &frame = stack[i];
        auto val = frame.val;
        Use *use = frame.use;
        unsigned idx = use->getOperandNo();
        unsigned nargs = val->getNumOperands();
        args.resize(nargs);
        for (unsigned j = 0; j < nargs; j++) {
            if (idx == j) {
                args[j] = res;
            }
            else {
                args[j] = cast<Constant>(val->getOperand(j));
            }
        }
        if (auto expr = dyn_cast<ConstantExpr>(val)) {
            res = expr->getWithOperands(args);
        }
        else if (auto ary = dyn_cast<ConstantArray>(val)) {
            res = ConstantArray::get(ary->getType(), args);
        }
        else if (auto strct = dyn_cast<ConstantStruct>(val)) {
            res = ConstantStruct::get(strct->getType(), args);
        }
        else if (isa<ConstantVector>(val)) {
            res = ConstantVector::get(args);
        }
        else {
            jl_safe_printf("Unknown const use.");
            llvm_dump(val);
            abort();
        }
    }
    return res;
}

// replace an alias to a function with a trampoline and (uninitialized) global variable slot
void CloneCtx::rewrite_alias(GlobalAlias *alias, Function *F)
{
    assert(!is_vector(F->getFunctionType()));

    Function *trampoline =
        Function::Create(F->getFunctionType(), alias->getLinkage(), "", &M);
    trampoline->copyAttributesFrom(F);
    trampoline->takeName(alias);
    trampoline->setVisibility(alias->getVisibility());
    alias->eraseFromParent();

    uint32_t id;
    GlobalVariable *slot;
    std::tie(id, slot) = get_reloc_slot(F);
    assert(id != (uint32_t)-1); // should have been grouped with its base function
    for (auto &grp: groups) {
        assert(grp.relocs.count(id));
        for (auto &tgt: grp.clones) {
            assert(tgt.relocs.count(id));
        }
    }
    alias_relocs.insert(id);

    auto BB = BasicBlock::Create(F->getContext(), "top", trampoline);
    IRBuilder<> irbuilder(BB);

    auto ptr = irbuilder.CreateLoad(F->getType(), slot);
    ptr->setMetadata(llvm::LLVMContext::MD_tbaa, tbaa_const);
    ptr->setMetadata(llvm::LLVMContext::MD_invariant_load, MDNode::get(F->getContext(), None));

    std::vector<Value *> Args;
    for (auto &arg : trampoline->args())
        Args.push_back(&arg);
    auto call = irbuilder.CreateCall(F->getFunctionType(), ptr, makeArrayRef(Args));
    if (F->isVarArg())
#if (defined(_CPU_ARM_) || defined(_CPU_PPC_) || defined(_CPU_PPC64_))
        abort();    // musttail support is very bad on ARM, PPC, PPC64 (as of LLVM 3.9)
#else
        call->setTailCallKind(CallInst::TCK_MustTail);
#endif
    else
        call->setTailCallKind(CallInst::TCK_Tail);

    if (F->getReturnType() == Type::getVoidTy(F->getContext()))
        irbuilder.CreateRetVoid();
    else
        irbuilder.CreateRet(call);
}

void CloneCtx::fix_gv_uses()
{
    auto single_pass = [&] (Function *orig_f) {
        bool changed = false;
        for (auto uses = ConstantUses<GlobalValue>(orig_f, M); !uses.done(); uses.next()) {
            changed = true;
            auto &stack = uses.get_stack();
            auto info = uses.get_info();
            // We only support absolute pointer relocation.
            assert(info.samebits);
            GlobalVariable *val;
            if (auto alias = dyn_cast<GlobalAlias>(info.val)) {
                rewrite_alias(alias, orig_f);
                continue;
            }
            else {
                val = cast<GlobalVariable>(info.val);
            }
            assert(info.use->getOperandNo() == 0);
            assert(!val->isConstant());
            auto fid = get_func_id(orig_f);
            auto addr = ConstantExpr::getPtrToInt(val, getSizeTy(val->getContext()));
            if (info.offset)
                addr = ConstantExpr::getAdd(addr, ConstantInt::get(getSizeTy(val->getContext()), info.offset));
            gv_relocs.emplace_back(addr, fid);
            val->setInitializer(rewrite_gv_init(stack));
        }
        return changed;
    };
    for (auto orig_f: orig_funcs) {
        if (groups.size() == 1 && !cloned.count(orig_f))
            continue;
        while (single_pass(orig_f)) {
        }
    }
}

std::pair<uint32_t,GlobalVariable*> CloneCtx::get_reloc_slot(Function *F)
{
    if (F->isDeclaration()) {
        auto it = extern_relocs.find(F);
        assert(it != extern_relocs.end());
        return {(uint32_t)-1, it->second};
    }
    auto id = get_func_id(F);
    auto &slot = const_relocs[id];
    assert(slot);
    return std::make_pair(id, slot);
}

template<typename Stack>
static Value *rewrite_inst_use(const Stack& stack, Value *replace, Instruction *insert_before)
{
    SmallVector<Constant*, 8> args;
    uint32_t nlevel = stack.size();
    for (uint32_t i = 1; i < nlevel; i++) {
        auto &frame = stack[i];
        auto val = frame.val;
        Use *use = frame.use;
        unsigned idx = use->getOperandNo();
        if (auto expr = dyn_cast<ConstantExpr>(val)) {
            auto inst = expr->getAsInstruction();
            inst->replaceUsesOfWith(val->getOperand(idx), replace);
            inst->insertBefore(insert_before);
            replace = inst;
            continue;
        }
        unsigned nargs = val->getNumOperands();
        args.resize(nargs);
        for (unsigned j = 0; j < nargs; j++) {
            auto op = val->getOperand(j);
            if (idx == j) {
                args[j] = UndefValue::get(op->getType());
            }
            else {
                args[j] = cast<Constant>(op);
            }
        }
        if (auto ary = dyn_cast<ConstantArray>(val)) {
            replace = InsertValueInst::Create(ConstantArray::get(ary->getType(), args),
                                              replace, {idx}, "", insert_before);
        }
        else if (auto strct = dyn_cast<ConstantStruct>(val)) {
            replace = InsertValueInst::Create(ConstantStruct::get(strct->getType(), args),
                                              replace, {idx}, "", insert_before);
        }
        else if (isa<ConstantVector>(val)) {
            replace = InsertElementInst::Create(ConstantVector::get(args), replace,
                                                ConstantInt::get(getSizeTy(insert_before->getContext()), idx), "",
                                                insert_before);
        }
        else {
            jl_safe_printf("Unknown const use.");
            llvm_dump(val);
            abort();
        }
    }
    return replace;
}

void CloneCtx::fix_inst_uses()
{
    uint32_t nfuncs = orig_funcs.size();
    for (auto &grp: groups) {
        auto suffix = ".clone_" + std::to_string(grp.idx);
        for (uint32_t i = 0; i < nfuncs; i++) {
            if (!grp.clone_fs.count(i))
                continue;
            auto orig_f = orig_funcs[i];
            auto F = grp.base_func(orig_f);
            bool changed;
            do {
                changed = false;
                for (auto uses = ConstantUses<Instruction>(F, M); !uses.done(); uses.next()) {
                    auto info = uses.get_info();
                    auto use_i = info.val;
                    auto use_f = use_i->getFunction();
                    if (!use_f->getName().endswith(suffix))
                        continue;
                    Instruction *insert_before = use_i;
                    if (auto phi = dyn_cast<PHINode>(use_i))
                        insert_before = phi->getIncomingBlock(*info.use)->getTerminator();
                    uint32_t id;
                    GlobalVariable *slot;
                    std::tie(id, slot) = get_reloc_slot(orig_f);
                    Instruction *ptr = new LoadInst(orig_f->getType(), slot, "", false, insert_before);
                    ptr->setMetadata(llvm::LLVMContext::MD_tbaa, tbaa_const);
                    ptr->setMetadata(llvm::LLVMContext::MD_invariant_load, MDNode::get(ptr->getContext(), None));
                    use_i->setOperand(info.use->getOperandNo(),
                                      rewrite_inst_use(uses.get_stack(), ptr,
                                                       insert_before));

                    // externally defined relocations should not show up in relocs
                    if (id != (uint32_t)-1) {
                        // dbgs() << "fvar " << id << " had a use in group " << grp.idx << "\n";
                        assert(grp.relocs.count(id));
                        for (auto &tgt: grp.clones) {
                            // The enclosing function of the use is cloned,
                            // no need to deal with this use on this target.
                            if (map_get(*tgt.vmap, use_f)) {
                                // dbgs() << "fvar " << id << " is skipped in target " << tgt.idx << "\n";
                                continue;
                            }
                            // dbgs() << "fvar " << id << " is added to target " << tgt.idx << "\n";
                            assert(tgt.relocs.count(id));
                        }
                    }

                    changed = true;
                }
            } while (changed);
        }
    }
}

static Constant *get_ptrdiff32(Constant *ptr, Constant *base)
{
    if (ptr->getType()->isPointerTy())
        ptr = ConstantExpr::getPtrToInt(ptr, getSizeTy(ptr->getContext()));
    auto ptrdiff = ConstantExpr::getSub(ptr, base);
    return sizeof(void*) == 8 ? ConstantExpr::getTrunc(ptrdiff, Type::getInt32Ty(ptr->getContext())) : ptrdiff;
}

template<typename T>
static Constant *emit_offset_table(Module &M, const std::vector<T*> &vars, StringRef name, StringRef suffix)
{
    auto T_int32 = Type::getInt32Ty(M.getContext());
    auto T_size = getSizeTy(M.getContext());
    uint32_t nvars = vars.size();
    Constant *base = nullptr;
    if (nvars > 0) {
        base = ConstantExpr::getBitCast(vars[0], T_size->getPointerTo());
    } else {
        base = Constant::getNullValue(T_size->getPointerTo());
    }
    auto base_var = GlobalAlias::create(T_size, 0, GlobalValue::ExternalLinkage, name + "_base" + suffix, base, &M);
    base_var->setVisibility(GlobalValue::HiddenVisibility);
    auto vbase = ConstantExpr::getPtrToInt(base, T_size);
    std::vector<Constant*> offsets(nvars + 1);
    offsets[0] = ConstantInt::get(T_int32, nvars);
    if (nvars > 0) {
        offsets[1] = ConstantInt::get(T_int32, 0);
        for (uint32_t i = 1; i < nvars; i++)
            offsets[i + 1] = get_ptrdiff32(vars[i], vbase);
    }
    ArrayType *vars_type = ArrayType::get(T_int32, nvars + 1);
    auto offset_var = new GlobalVariable(M, vars_type, true,
                                  GlobalVariable::ExternalLinkage,
                                  ConstantArray::get(vars_type, offsets),
                                  name + "_offsets" + suffix);
    offset_var->setVisibility(GlobalValue::HiddenVisibility);
    return vbase;
}

void CloneCtx::emit_metadata()
{
    uint32_t nfvars = fvars.size();
    if (allow_bad_fvars && nfvars == 0) {
        // Will result in a non-loadable sysimg, but `allow_bad_fvars` is for bugpoint only
        return;
    }

    // Store back the information about exported functions.
    auto fbase = emit_offset_table(M, fvars, "jl_sysimg_fvars", gv_suffix);
    auto gbase = emit_offset_table(M, gvars, "jl_sysimg_gvars", gv_suffix);

    // Suffix the index variables
    M.getGlobalVariable("jl_sysimg_fvars_idxs")->setName("jl_sysimg_fvars_idxs" + gv_suffix);
    M.getGlobalVariable("jl_sysimg_gvars_idxs")->setName("jl_sysimg_gvars_idxs" + gv_suffix);

    uint32_t ntargets = specs.size();
    SmallVector<Target*, 8> targets(ntargets);
    for (auto &grp: groups) {
        targets[grp.idx] = &grp;
        for (auto &tgt: grp.clones) {
            targets[tgt.idx] = &tgt;
        }
    }

    // Generate `jl_dispatch_reloc_slots`
    std::set<uint32_t> shared_relocs;
    {
        auto T_int32 = Type::getInt32Ty(M.getContext());
        std::stable_sort(gv_relocs.begin(), gv_relocs.end(),
                         [] (const std::pair<Constant*,uint32_t> &lhs,
                             const std::pair<Constant*,uint32_t> &rhs) {
                             return lhs.second < rhs.second;
                         });
        std::vector<Constant*> values{nullptr};
        uint32_t gv_reloc_idx = 0;
        uint32_t ngv_relocs = gv_relocs.size();
        for (uint32_t id = 0; id < nfvars; id++) {
            // TODO:
            // explicitly set section? so that we are sure the relocation slots
            // are in the same section as `gbase`.
            auto id_v = ConstantInt::get(T_int32, id);
            for (; gv_reloc_idx < ngv_relocs && gv_relocs[gv_reloc_idx].second == id;
                 gv_reloc_idx++) {
                shared_relocs.insert(id);
                values.push_back(id_v);
                values.push_back(get_ptrdiff32(gv_relocs[gv_reloc_idx].first, gbase));
            }
            auto it = const_relocs.find(id);
            if (it != const_relocs.end()) {
                values.push_back(id_v);
                values.push_back(get_ptrdiff32(it->second, gbase));
            }
            if (alias_relocs.find(id) != alias_relocs.end()) {
                shared_relocs.insert(id);
            }
        }
        values[0] = ConstantInt::get(T_int32, values.size() / 2);
        ArrayType *vars_type = ArrayType::get(T_int32, values.size());
        auto reloc_slots = new GlobalVariable(M, vars_type, true, GlobalVariable::ExternalLinkage,
                                      ConstantArray::get(vars_type, values),
                                      "jl_dispatch_reloc_slots" + gv_suffix);
        reloc_slots->setVisibility(GlobalValue::HiddenVisibility);
    }

    // Generate `jl_dispatch_fvars_idxs` and `jl_dispatch_fvars_offsets`
    {
        std::vector<uint32_t> idxs;
        std::vector<Constant*> offsets;
        for (uint32_t i = 0; i < ntargets; i++) {
            auto tgt = targets[i];
            auto &spec = specs[i];
            uint32_t len_idx = idxs.size();
            idxs.push_back(0); // We will fill in the real value later.
            uint32_t count = 0;
            if (i == 0 || spec.flags & JL_TARGET_CLONE_ALL) {
                auto grp = static_cast<Group*>(tgt);
                count = jl_sysimg_tag_mask;
                for (uint32_t j = 0; j < nfvars; j++) {
                    if (shared_relocs.count(j) || tgt->relocs.count(j)) {
                        count++;
                        idxs.push_back(j);
                    }
                    if (i != 0) {
                        offsets.push_back(get_ptrdiff32(grp->base_func(fvars[j]), fbase));
                    }
                }
            }
            else {
                auto baseidx = spec.base;
                auto grp = static_cast<Group*>(targets[baseidx]);
                idxs.push_back(baseidx);
                for (uint32_t j = 0; j < nfvars; j++) {
                    auto base_f = grp->base_func(fvars[j]);
                    if (shared_relocs.count(j) || tgt->relocs.count(j)) {
                        count++;
                        idxs.push_back(jl_sysimg_tag_mask | j);
                        auto f = map_get(*tgt->vmap, base_f, base_f);
                        offsets.push_back(get_ptrdiff32(cast<Function>(f), fbase));
                    }
                    else if (auto f = map_get(*tgt->vmap, base_f)) {
                        count++;
                        idxs.push_back(j);
                        offsets.push_back(get_ptrdiff32(cast<Function>(f), fbase));
                    }
                }
            }
            idxs[len_idx] = count;
        }
        auto idxval = ConstantDataArray::get(M.getContext(), idxs);
        auto clone_idxs = new GlobalVariable(M, idxval->getType(), true,
                                      GlobalVariable::ExternalLinkage,
                                      idxval, "jl_dispatch_fvars_idxs" + gv_suffix);
        clone_idxs->setVisibility(GlobalValue::HiddenVisibility);
        ArrayType *offsets_type = ArrayType::get(Type::getInt32Ty(M.getContext()), offsets.size());
        auto clone_offsets = new GlobalVariable(M, offsets_type, true,
                                      GlobalVariable::ExternalLinkage,
                                      ConstantArray::get(offsets_type, offsets),
                                      "jl_dispatch_fvars_offsets" + gv_suffix);
        clone_offsets->setVisibility(GlobalValue::HiddenVisibility);
    }
}

static bool runMultiVersioning(Module &M, bool allow_bad_fvars)
{
    // Group targets and identify cloning bases.
    // Also initialize function info maps (we'll update these maps as we go)
    // Maps that we need includes,
    //
    //     * Original function -> ID (initialize from `fvars` and allocate ID lazily)
    //     * Cloned function -> Original function (add as we clone functions)
    //     * Original function -> Base function (target specific and updated by LLVM)
    //     * ID -> relocation slots (const).
    if (M.getName() == "sysimage")
        return false;

    GlobalVariable *fvars = M.getGlobalVariable("jl_sysimg_fvars");
    GlobalVariable *gvars = M.getGlobalVariable("jl_sysimg_gvars");
    if (allow_bad_fvars && (!fvars || !fvars->hasInitializer() || !isa<ConstantArray>(fvars->getInitializer()) ||
                            !gvars || !gvars->hasInitializer() || !isa<ConstantArray>(gvars->getInitializer())))
        return false;

    CloneCtx clone(M, allow_bad_fvars);

    // Prepare relocation slots
    // We must do this upfront as sharded multiversioning
    // requires prior knowledge of which function actually
    // need relocation slots and which functions are part
    // of the fvars list, so we cannot add to these on
    // the fly. 
    clone.prepare_relocs();

    // Clone function declarations
    // This needs to happen first so that the intra-function
    // intra-target calls are fixed up properly. We fix up
    // the intra-function inter-target calls later.
    // In the case of sharded multiversioning, this may
    // clone a declaration that has no body, and this
    // declaration will be ignored during the cloning
    // of bodies. 
    clone.clone_decls();

    // Collect a list of original functions and clone base functions
    clone.clone_bases();

    // If any partially cloned target exist decide which functions to clone for these targets.
    // Clone functions for each group and collect a list of them.
    // We can also add feature strings for cloned functions
    // now that no additional cloning needs to be done.
    clone.clone_all_partials();

    // Scan **ALL** cloned functions (including full cloning for base target)
    // for global variables initialization use.
    // Replace them with `null` slot to be initialized at runtime and record relocation slot.
    // These relocations must be initialized for **ALL** targets.
    clone.fix_gv_uses();

    // For each group, scan all functions cloned by **PARTIALLY** cloned targets for
    // instruction use.
    // A function needs a const relocation slot if it is cloned and is called by a
    // uncloned function for at least one partially cloned target in the group.
    // This is also the condition that a use in an uncloned function needs to be replaced with
    // a slot load (i.e. if both the caller and the callee are always cloned or not cloned
    // on all targets, the caller site does not need a relocation slot).
    // A target needs a slot to be initialized iff at least one caller is not initialized.
    clone.fix_inst_uses();

    // Store back sysimg information with the correct format.
    // At this point, we should have fixed up all the uses of the cloned functions
    // and collected all the shared/target-specific relocations.
    clone.emit_metadata();

    assert(!verify_multiversioning(M));
#ifdef JL_VERIFY_PASSES
    assert(!verifyModule(M, &errs()));
#endif

    return true;
}

struct MultiVersioningLegacy: public ModulePass {
    static char ID;
    MultiVersioningLegacy(bool allow_bad_fvars=false)
        : ModulePass(ID), allow_bad_fvars(allow_bad_fvars)
    {}

private:
    bool runOnModule(Module &M) override;
    bool allow_bad_fvars;
};

bool MultiVersioningLegacy::runOnModule(Module &M)
{
    return runMultiVersioning(M, allow_bad_fvars);
}


char MultiVersioningLegacy::ID = 0;
static RegisterPass<MultiVersioningLegacy> X("JuliaMultiVersioning", "JuliaMultiVersioning Pass",
                                       false /* Only looks at CFG */,
                                       false /* Analysis Pass */);

} // anonymous namespace

PreservedAnalyses MultiVersioning::run(Module &M, ModuleAnalysisManager &AM)
{
    if (runMultiVersioning(M, external_use)) {
        auto preserved = PreservedAnalyses::allInSet<CFGAnalyses>();
        preserved.preserve<LoopAnalysis>();
        return preserved;
    }
    return PreservedAnalyses::all();
}

Pass *createMultiVersioningPass(bool allow_bad_fvars)
{
    return new MultiVersioningLegacy(allow_bad_fvars);
}

extern "C" JL_DLLEXPORT void LLVMExtraAddMultiVersioningPass_impl(LLVMPassManagerRef PM)
{
    unwrap(PM)->add(createMultiVersioningPass(false));
}
