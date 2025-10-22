# Tensor Logic Execution Architecture

## Overview: VM with Multi-Backend Dispatch
```
Source Code → Parser → AST → Compiler → Bytecode → VM → Backend(s)
```

---

## Execution Models (Choose One)

### Option A: Interpreted VM (Recommended for Phase 1-2)
```cpp
class TensorLogicVM {
private:
    vector<unique_ptr<TensorBackend>> backends_;
    BackendRouter router_;
    Environment env_;  // Symbol table for tensors
    
public:
    Tensor execute(const Program& program) {
        for (const auto& equation : program.equations) {
            // Analyze which backend to use
            BackendType type = router_.analyze(equation, env_);
            auto& backend = backends_[type];
            
            // Execute on chosen backend
            Tensor result = backend->compute(equation);
            
            // Store in environment
            env_.bind(equation.lhs, result);
        }
        return env_.lookup(program.query);
    }
};
```

**Execution Flow**:
1. Walk through equations sequentially (forward chaining)
2. For each equation, route to appropriate backend
3. Store intermediate results in environment
4. Return final query result

---

### Option B: JIT Compiled VM (Phase 3-4)
```cpp
class CompiledProgram {
private:
    vector<Instruction> bytecode_;
    
public:
    struct Instruction {
        Opcode op;              // EINSUM, PROJECT, JOIN, etc.
        BackendType backend;    // Which backend executes this
        vector<int> operands;   // Tensor IDs
        int result;             // Where to store result
    };
    
    Tensor execute(TensorLogicVM& vm) {
        for (const auto& instr : bytecode_) {
            vm.dispatch(instr);
        }
        return vm.getTensor(bytecode_.back().result);
    }
};
```

**Compilation Pipeline**:
```
AST → SSA IR → Optimize → Backend Assignment → Bytecode
```

---

## Backend Routing Strategies

### Strategy 1: Static Analysis (Fast)
```cpp
class BackendRouter {
public:
    BackendType analyze(const Equation& eq, const Environment& env) {
        // Check tensor properties
        for (const auto& var : eq.rhs_vars) {
            Tensor t = env.lookup(var);
            
            // Boolean + sparse → Database
            if (t.dtype() == Bool && t.sparsity() > 0.95)
                return BackendType::Database;
            
            // Dense + requires_grad → LibTorch
            if (t.dense() && t.requires_grad())
                return BackendType::LibTorch;
        }
        
        // Default to LibTorch
        return BackendType::LibTorch;
    }
};
```

### Strategy 2: Cost-Based (Optimal)
```cpp
class CostBasedRouter {
public:
    BackendType analyze(const Equation& eq, const Environment& env) {
        float db_cost = estimateDatabaseCost(eq, env);
        float torch_cost = estimateTorchCost(eq, env);
        float tucker_cost = estimateTuckerCost(eq, env);
        
        return min({db_cost, torch_cost, tucker_cost});
    }
    
private:
    float estimateDatabaseCost(const Equation& eq, const Environment& env) {
        // Estimate based on relation cardinalities
        size_t join_size = 1;
        for (auto& var : eq.rhs_vars) {
            join_size *= env.lookup(var).cardinality();
        }
        return join_size * COST_PER_TUPLE;
    }
};
```

---

## Concrete VM Implementation
```cpp
class TensorLogicVM {
public:
    enum class ExecutionMode {
        Interpreted,    // Execute equations one by one
        Compiled,       // Pre-compile to bytecode
        Hybrid          // Mix both
    };
    
    TensorLogicVM(ExecutionMode mode) : mode_(mode) {
        // Register backends
        backends_[BackendType::LibTorch] = 
            make_unique<LibTorchBackend>();
        backends_[BackendType::Database] = 
            make_unique<DatabaseBackend>();
        backends_[BackendType::Tucker] = 
            make_unique<TuckerBackend>();
    }
    
    Tensor execute(const Program& program) {
        if (mode_ == ExecutionMode::Compiled) {
            return executeCompiled(program);
        } else {
            return executeInterpreted(program);
        }
    }
    
private:
    Tensor executeInterpreted(const Program& program) {
        // Forward chaining interpreter
        for (const auto& equation : program.equations) {
            BackendType backend = router_.analyze(equation, env_);
            
            // Get input tensors
            vector<Tensor> inputs;
            for (const auto& var : equation.rhs_vars) {
                inputs.push_back(env_.lookup(var));
            }
            
            // Execute on backend
            Tensor result = backends_[backend]->compute(
                equation.operation, 
                inputs
            );
            
            // Store result
            env_.bind(equation.lhs_var, result);
        }
        
        // Return query result
        return env_.lookup(program.query_var);
    }
    
    Tensor executeCompiled(const Program& program) {
        // Compile once
        if (!compiled_cache_.contains(program.id)) {
            compiled_cache_[program.id] = compiler_.compile(program);
        }
        
        // Execute bytecode
        return compiled_cache_[program.id]->execute(*this);
    }
    
    // Called by bytecode instructions
    void dispatch(const Instruction& instr) {
        auto& backend = backends_[instr.backend];
        
        // Get operands
        vector<Tensor> operands;
        for (int id : instr.operands) {
            operands.push_back(tensor_store_[id]);
        }
        
        // Execute
        Tensor result = backend->executeOp(instr.op, operands);
        
        // Store
        tensor_store_[instr.result] = result;
    }
    
private:
    ExecutionMode mode_;
    map<BackendType, unique_ptr<TensorBackend>> backends_;
    BackendRouter router_;
    Environment env_;
    Compiler compiler_;
    map<ProgramId, unique_ptr<CompiledProgram>> compiled_cache_;
    vector<Tensor> tensor_store_;  // For compiled execution
};
```

---

## Example Execution Trace

### Input Program:
```
X(p, t)  // Text input (sparse Boolean)
Emb[p, d] = X(p, t) Emb[t, d]  // Dense embedding
Y[p, c] = softmax(W[c, d] Emb[p, d])  // Classification
```

### VM Execution:
```
Step 1: Load X(p, t)
  Router: Sparse Boolean → DatabaseBackend
  Action: CREATE TABLE X(p INT, t INT)
  Store: env_["X"] = DatabaseTensor(...)

Step 2: Compute Emb[p, d]
  Router: Sparse × Dense → HybridBackend
  Action: SQL JOIN + LibTorch matmul
    - Database computes join: SELECT p, t FROM X
    - LibTorch: torch.matmul(sparse_X, Emb_t)
  Store: env_["Emb"] = DenseTensor(...)

Step 3: Compute Y[p, c]
  Router: Dense × Dense with grad → LibTorchBackend
  Action: torch.softmax(torch.matmul(W, Emb))
  Store: env_["Y"] = DenseTensor(..., requires_grad=true)

Step 4: Query Y?
  Return: env_["Y"]
```

---

## Key Design Decisions

### 1. **Lazy vs Eager Execution**
```cpp
// Eager (Phase 1-2): Execute immediately
Tensor result = backend->compute(eq);

// Lazy (Phase 3-4): Build computation graph
Graph g = backend->buildGraph(eq);
Tensor result = g.execute();  // Allows fusion
```

### 2. **Data Movement**
```cpp
class TensorWrapper {
    BackendType home_;  // Where data lives
    
    Tensor materializeOn(BackendType target) {
        if (home_ == target) return data_;
        
        // Convert between backends
        return converter_.convert(data_, home_, target);
    }
};
```

### 3. **Memory Management**
```cpp
class Environment {
    // Automatically free unused tensors
    void bind(const string& name, Tensor t) {
        if (refcount_[name] == 0) {
            tensors_.erase(name);  // Free memory
        }
        tensors_[name] = t;
    }
};
```

---

## Advantages of VM Approach

**Clean separation**: Backends don't know about each other  
**Easy debugging**: Can trace execution step-by-step  
**Flexible**: Add new backends without changing VM  
**Optimizable**: Can analyze whole program before execution  
**Testable**: Can mock backends for testing  

---

## Alternative: Direct AST Interpretation (Simpler but less flexible)
```cpp
class ASTExecutor {
    Tensor visit(const Equation& eq) {
        // Directly execute, no VM
        vector<Tensor> operands = evaluateOperands(eq.rhs);
        return backend_->einsum(eq.indices, operands);
    }
};
```

Use this for early prototyping, migrate to VM in Phase 2.
