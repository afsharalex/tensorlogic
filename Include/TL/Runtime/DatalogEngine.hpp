#pragma once

#include "TL/AST.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <iostream>

namespace tl {

// Forward declarations
class Environment;

/**
 * @brief Datalog subsystem for logic programming
 *
 * Handles:
 * - Fact storage and retrieval
 * - Rule registration
 * - Forward chaining (fixpoint saturation)
 * - Query evaluation with joins and conditions
 *
 * This class manages all Datalog-specific logic, keeping it separate
 * from tensor operations handled by executors.
 */
class DatalogEngine {
public:
    /**
     * @brief Construct a DatalogEngine
     * @param env Reference to the environment (for fact storage and tensor queries)
     * @param out Output stream for query results (default: stdout)
     */
    explicit DatalogEngine(Environment& env, std::ostream* out = &std::cout);

    /**
     * @brief Add a Datalog fact to the database
     * @param fact The fact to add
     * @return true if the fact was newly inserted, false if it already existed
     */
    bool addFact(const DatalogFact& fact);

    /**
     * @brief Register a Datalog rule for forward chaining
     * @param rule The rule to register
     */
    void addRule(const DatalogRule& rule);

    /**
     * @brief Get all registered rules
     */
    const std::vector<DatalogRule>& rules() const { return rules_; }

    /**
     * @brief Run forward chaining to fixpoint
     *
     * Applies all rules iteratively until no new facts are derived.
     * Only runs if closure_dirty_ flag is set.
     */
    void saturate();

    /**
     * @brief Check if saturation is needed
     * @return true if new facts/rules have been added since last saturation
     */
    bool needsSaturation() const { return closure_dirty_; }

    /**
     * @brief Execute a Datalog or tensor query
     * @param query The query to execute
     * @param out Output stream for results
     *
     * For Datalog queries, performs join evaluation with conditions.
     * For tensor queries, delegates to VM (via callback if needed).
     */
    void query(const Query& query, std::ostream& out);

    /**
     * @brief Enable or disable debug logging
     */
    void setDebug(bool enabled) { debug_ = enabled; }

    /**
     * @brief Check if debug logging is enabled
     */
    bool debug() const { return debug_; }

private:
    Environment& env_;
    std::ostream* output_stream_;
    std::vector<DatalogRule> rules_;
    bool closure_dirty_{false};
    bool debug_{false};

    /**
     * @brief Apply a single rule once, deriving new facts
     * @param rule The rule to apply
     * @return Number of new facts derived
     */
    size_t applyRule(const DatalogRule& rule);

    /**
     * @brief Execute a Datalog atom query
     * @param atom The query atom
     * @param body Additional atoms/conditions for conjunctive queries
     * @param out Output stream
     */
    void execDatalogQuery(const DatalogAtom& atom,
                          const std::vector<std::variant<DatalogAtom, DatalogCondition>>& body,
                          std::ostream& out);

    /**
     * @brief Evaluate expression in Datalog context
     * @param expr Expression to evaluate
     * @param binding Current variable bindings
     * @param outStr Result as string
     * @param outNum Result as number
     * @param isNumeric Whether result is numeric
     * @return true if evaluation succeeded
     */
    bool evalExprBinding(const ExprPtr& expr,
                        const std::unordered_map<std::string, std::string>& binding,
                        std::string& outStr,
                        double& outNum,
                        bool& isNumeric) const;

    /**
     * @brief Evaluate a condition (comparison)
     * @param cond The condition to evaluate
     * @param binding Current variable bindings
     * @return true if condition is satisfied
     */
    bool evalCondition(const DatalogCondition& cond,
                      const std::unordered_map<std::string, std::string>& binding) const;

    /**
     * @brief Log debug message
     */
    void debugLog(const std::string& msg) const;
};

} // namespace tl
