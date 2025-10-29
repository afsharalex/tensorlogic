#include "TL/Runtime/Executors/PoolingExecutor.hpp"
#include "TL/vm.hpp"
#include <torch/torch.h>
#include <unordered_map>
#include <limits>

namespace tl {

    bool PoolingExecutor::canExecute(const TensorEquation &eq, const Environment &env) const {
        // Check if this is a pooling operation (+=, max=, min=, avg=)
        if (eq.projection != "+=" && eq.projection != "max=" &&
            eq.projection != "min=" && eq.projection != "avg=") {
            return false;
        }

        // Only handle single-clause, no-guard equations
        if (eq.clauses.size() != 1 || eq.clauses[0].guard.has_value()) {
            return false;
        }

        // RHS must be a tensor reference
        if (!eq.clauses[0].expr) return false;
        const Expr &e = *eq.clauses[0].expr;
        const auto *eref = std::get_if<ExprTensorRef>(&e.node);
        if (!eref) return false;

        // RHS tensor must exist
        const std::string srcName = eref->ref.name.name;
        return env.has(srcName);
    }

    Tensor PoolingExecutor::execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) {
        const Expr &e = *eq.clauses[0].expr;
        const auto *eref = std::get_if<ExprTensorRef>(&e.node);
        if (!eref) {
            throw ExecutionError("PoolingExecutor: expected tensor ref on RHS");
        }

        const Tensor &src = env.lookup(eref->ref);
        const int64_t rank = src.dim();

        // Build mapping from RHS index variable name -> axis position
        std::unordered_map<std::string, int64_t> rhsAxis;
        for (int64_t ax = 0; ax < static_cast<int64_t>(eref->ref.indices.size()); ++ax) {
            const auto &ios = eref->ref.indices[ax];
            // Skip slices - they don't contribute to variable mapping
            if (std::holds_alternative<Index>(ios.value)) {
                const auto &idx = std::get<Index>(ios.value);
                if (const auto *id = std::get_if<Identifier>(&idx.value)) {
                    rhsAxis[id->name] = ax;
                }
            }
        }

        // Parse LHS indices to determine output shape and mapping
        struct MapItem {
            std::string base;
            int64_t divisor;
        };
        std::vector<MapItem> lhsMap;
        std::vector<int64_t> outShape;

        // Helper to parse division expressions (e.g., "i/2" -> {"i", 2})
        auto parseDiv = [](const std::string &s) -> std::pair<std::string, int64_t> {
            auto pos = s.find('/');
            if (pos == std::string::npos) return {s, 1};
            std::string base = s.substr(0, pos);
            int64_t div = 1;
            try {
                div = std::stoll(s.substr(pos + 1));
            } catch (...) {
                div = 1;
            }
            if (div <= 0) div = 1;
            return {base, div};
        };

        for (const auto &ios : eq.lhs.indices) {
            if (std::holds_alternative<tl::Slice>(ios.value)) {
                // Slice on LHS - treat as full dimension pass-through
                // For pooling, this is unusual but we'll treat it as a variable with no division
                lhsMap.push_back({"", 1});
                outShape.push_back(1);
            } else {
                const auto &idx = std::get<Index>(ios.value);
                if (const auto *id = std::get_if<Identifier>(&idx.value)) {
                    auto [base, div] = parseDiv(id->name);
                    lhsMap.push_back({base, div});

                    auto it = rhsAxis.find(base);
                    int64_t size = 1;
                    if (it != rhsAxis.end()) {
                        const int64_t inSize = src.size(it->second);
                        size = (div <= 1) ? inSize : ((inSize + div - 1) / div);
                    }
                    outShape.push_back(size);
                } else if (const auto *num = std::get_if<NumberLiteral>(&idx.value)) {
                    // Numeric fixed index contributes shape 1
                    outShape.push_back(1);
                    lhsMap.push_back({"", 1});
                }
            }
        }

        // Prepare output tensor
        torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor out;

        if (eq.projection == "max=") {
            out = torch::full(outShape.empty() ? std::vector<int64_t>{1} : outShape,
                            -std::numeric_limits<float>::infinity(), opts);
        } else if (eq.projection == "min=") {
            out = torch::full(outShape.empty() ? std::vector<int64_t>{1} : outShape,
                            std::numeric_limits<float>::infinity(), opts);
        } else {
            out = torch::zeros(outShape.empty() ? std::vector<int64_t>{1} : outShape, opts);
        }

        // Counts tensor for average pooling
        torch::Tensor counts;
        bool needCounts = (eq.projection == "avg=");
        if (needCounts) {
            counts = torch::zeros_like(out);
        }

        const bool scalarOut = outShape.empty();

        // Handle scalar source
        if (rank == 0) {
            float v = src.item<float>();
            if (scalarOut) {
                float acc = v;
                out = torch::tensor(acc, opts);
                if (needCounts) counts = torch::tensor(1.0f, opts);
            } else {
                std::vector<torch::indexing::TensorIndex> outIdxTI;
                for (size_t li = 0; li < lhsMap.size(); ++li) {
                    outIdxTI.emplace_back(0);
                }
                if (eq.projection == "+=") {
                    out.index_put_(outIdxTI, out.index(outIdxTI) + v);
                } else if (eq.projection == "avg=") {
                    out.index_put_(outIdxTI, out.index(outIdxTI) + v);
                    counts.index_put_(outIdxTI, counts.index(outIdxTI) + 1.0f);
                } else if (eq.projection == "max=") {
                    out.index_put_(outIdxTI, torch::maximum(out.index(outIdxTI), torch::tensor(v, opts)));
                } else if (eq.projection == "min=") {
                    out.index_put_(outIdxTI, torch::minimum(out.index(outIdxTI), torch::tensor(v, opts)));
                }
            }
        } else {
            // Iterate over all coordinates in source tensor
            std::vector<int64_t> sizes(rank, 1);
            for (int64_t d = 0; d < rank; ++d) {
                sizes[d] = src.size(d);
            }

            std::vector<int64_t> coord(rank, 0);

            auto stepCoord = [&]() {
                for (int64_t d = rank - 1; d >= 0; --d) {
                    coord[d] += 1;
                    if (coord[d] < sizes[d]) return true;
                    coord[d] = 0;
                }
                return false;
            };

            bool cont = true;
            while (cont) {
                // Build output index from lhsMap
                std::vector<torch::indexing::TensorIndex> outIdxTI;
                for (size_t li = 0; li < lhsMap.size(); ++li) {
                    const auto &mi = lhsMap[li];
                    if (mi.base.empty()) {
                        outIdxTI.emplace_back(0);
                        continue;
                    }
                    auto it = rhsAxis.find(mi.base);
                    int64_t v = 0;
                    if (it != rhsAxis.end()) {
                        v = coord[it->second];
                        if (mi.divisor > 1) {
                            v = v / mi.divisor;
                        }
                    }
                    outIdxTI.emplace_back(v);
                }

                // Fetch value at coord
                std::vector<torch::indexing::TensorIndex> coordIdx;
                for (int64_t d = 0; d < rank; ++d) {
                    coordIdx.emplace_back(coord[d]);
                }
                float val = src.index(coordIdx).item<float>();

                if (scalarOut) {
                    float current = out.item<float>();
                    if (eq.projection == "+=") {
                        current += val;
                        out = torch::tensor(current, opts);
                    } else if (eq.projection == "avg=") {
                        current += val;
                        out = torch::tensor(current, opts);
                        counts = counts.numel() ? (counts + 1.0f) : torch::tensor(1.0f, opts);
                    } else if (eq.projection == "max=") {
                        out = torch::tensor(std::max(current, val), opts);
                    } else if (eq.projection == "min=") {
                        out = torch::tensor(std::min(current, val), opts);
                    }
                } else {
                    if (eq.projection == "+=") {
                        out.index_put_(outIdxTI, out.index(outIdxTI) + val);
                    } else if (eq.projection == "avg=") {
                        out.index_put_(outIdxTI, out.index(outIdxTI) + val);
                        counts.index_put_(outIdxTI, counts.index(outIdxTI) + 1.0f);
                    } else if (eq.projection == "max=") {
                        out.index_put_(outIdxTI, torch::maximum(out.index(outIdxTI), torch::tensor(val, opts)));
                    } else if (eq.projection == "min=") {
                        out.index_put_(outIdxTI, torch::minimum(out.index(outIdxTI), torch::tensor(val, opts)));
                    }
                }

                cont = stepCoord();
            }
        }

        // Finalize average pooling
        if (needCounts) {
            if (out.dim() == 0 && counts.dim() == 0) {
                float denom = std::max(1.0f, counts.item<float>());
                out = out / denom;
            } else {
                torch::Tensor denom = torch::clamp_min(counts, 1.0f);
                out = out / denom;
            }
        }

        return out;
    }

    std::string PoolingExecutor::name() const {
        return "PoolingExecutor";
    }

    int PoolingExecutor::priority() const {
        return 50; // Medium priority - after basic operations, before general expressions
    }

} // namespace tl
