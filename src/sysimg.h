#include "jitlayers.h"

#include <map>


typedef struct {
    llvm::orc::ThreadSafeModule M;
    std::vector<llvm::GlobalValue*> jl_sysimg_fvars;
    std::vector<llvm::GlobalValue*> jl_sysimg_gvars;
    std::map<jl_code_instance_t*, std::tuple<uint32_t, uint32_t>> jl_fvar_map;
    std::vector<void*> jl_value_to_llvm;
} jl_native_code_desc_t;

// Iterate through uses of a particular type.
// Recursively scan through `ConstantExpr` and `ConstantAggregate` use.
template<typename U>
struct ConstantUses {
    template<typename T>
    struct Info {
        llvm::Use *use;
        T *val;
        // If `samebits == true`, the offset the original value appears in the constant.
        size_t offset;
        // This specify whether the original value appears in the current value in exactly
        // the same bit pattern (with possibly an offset determined by `offset`).
        bool samebits;
        Info(llvm::Use *use, T *val, size_t offset, bool samebits) :
            use(use),
            val(val),
            offset(offset),
            samebits(samebits)
        {
        }
        Info(llvm::Use *use, size_t offset, bool samebits) :
            use(use),
            val(cast<T>(use->getUser())),
            offset(offset),
            samebits(samebits)
        {
        }
    };
    using UseInfo = Info<U>;
    struct Frame : Info<llvm::Constant> {
        template<typename... Args>
        Frame(Args &&... args) :
            Info<llvm::Constant>(std::forward<Args>(args)...),
            cur(this->val->use_empty() ? nullptr : &*this->val->use_begin()),
            _next(cur ? cur->getNext() : nullptr)
        {
        }
    private:
        void next()
        {
            cur = _next;
            if (!cur)
                return;
            _next = cur->getNext();
        }
        llvm::Use *cur;
        llvm::Use *_next;
        friend struct ConstantUses;
    };
    ConstantUses(llvm::Constant *c, llvm::Module &M)
        : stack{Frame(nullptr, c, 0u, true)},
          M(M)
    {
        forward();
    }
    UseInfo get_info() const
    {
        auto &top = stack.back();
        return UseInfo(top.cur, top.offset, top.samebits);
    }
    const llvm::SmallVector<Frame, 4> &get_stack() const
    {
        return stack;
    }
    void next()
    {
        stack.back().next();
        forward();
    }
    bool done()
    {
        return stack.empty();
    }
private:
    void forward();
    llvm::SmallVector<Frame, 4> stack;
    llvm::Module &M;
};

template<typename U>
void ConstantUses<U>::forward()
{
    assert(!stack.empty());
    auto frame = &stack.back();
    const llvm::DataLayout &DL = M.getDataLayout();
    auto pop = [&] {
        stack.pop_back();
        if (stack.empty()) {
            return false;
        }
        frame = &stack.back();
        return true;
    };
    auto push = [&] (llvm::Use *use, llvm::Constant *c, size_t offset, bool samebits) {
        stack.emplace_back(use, c, offset, samebits);
        frame = &stack.back();
    };
    auto handle_constaggr = [&] (llvm::Use *use, llvm::ConstantAggregate *aggr) {
        if (!frame->samebits) {
            push(use, aggr, 0, false);
            return;
        }
        if (auto strct = dyn_cast<llvm::ConstantStruct>(aggr)) {
            auto layout = DL.getStructLayout(strct->getType());
            push(use, strct, frame->offset + layout->getElementOffset(use->getOperandNo()), true);
        }
        else if (auto ary = dyn_cast<llvm::ConstantArray>(aggr)) {
            auto elty = ary->getType()->getElementType();
            push(use, ary, frame->offset + DL.getTypeAllocSize(elty) * use->getOperandNo(), true);
        }
        else if (auto vec = dyn_cast<llvm::ConstantVector>(aggr)) {
            auto elty = vec->getType()->getElementType();
            push(use, vec, frame->offset + DL.getTypeAllocSize(elty) * use->getOperandNo(), true);
        }
        else {
            jl_safe_printf("Unknown ConstantAggregate:\n");
            abort();
        }
    };
    auto handle_constexpr = [&] (llvm::Use *use, llvm::ConstantExpr *expr) {
        if (!frame->samebits) {
            push(use, expr, 0, false);
            return;
        }
        auto opcode = expr->getOpcode();
        if (opcode == llvm::Instruction::PtrToInt || opcode == llvm::Instruction::IntToPtr ||
            opcode == llvm::Instruction::AddrSpaceCast || opcode == llvm::Instruction::BitCast) {
            push(use, expr, frame->offset, true);
        }
        else {
            push(use, expr, 0, false);
        }
    };
    while (true) {
        auto use = frame->cur;
        if (!use) {
            if (!pop())
                return;
            continue;
        }
        auto user = use->getUser();
        if (isa<U>(user))
            return;
        frame->next();
        if (auto aggr = dyn_cast<llvm::ConstantAggregate>(user)) {
            handle_constaggr(use, aggr);
        }
        else if (auto expr = dyn_cast<llvm::ConstantExpr>(user)) {
            handle_constexpr(use, expr);
        }
    }
}

void add_sysimage_globals(llvm::Module &M, jl_native_code_desc_t *data);

void add_sysimage_targets(llvm::Module &M, bool has_veccall, uint32_t nshards, uint32_t nfvars, uint32_t ngvars);

void addComdat(llvm::GlobalValue *G);
