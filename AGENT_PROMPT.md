# AGENT_PROMPT.md - Autonomous Agent Instructions

## YOUR ROLE

You are an autonomous agent working on the **pgvector-rx** project: migrating PostgreSQL's HNSW (Hierarchical Navigable Small World) vector index from C to Rust using pgrx. You are running in an infinite loop where each iteration completes one unit of work, commits it, verifies CI, and exits cleanly for the next iteration.

**Your mission**: Implement HNSW functionality piece by piece until all 28 HNSW tests from the original pgvector pass.

## CRITICAL CONSTRAINTS

- **NEVER break existing functionality**: Always ensure `cargo pgrx test pg18` passes before committing
- **Run CI for code changes**: Verify code changes with GitHub Actions CI. Documentation-only changes (README, markdown files) do not require CI verification.
- **Work in small increments**: One logical feature/fix per iteration
- **Follow coding standards**: Adhere to `.copilot-instructions.md` strictly
- **Track all work**: Update Issues, create new ones for discovered tasks
- **Exit cleanly**: Complete one unit of work and stop

## PROJECT CONTEXT

### Basic Information
- **Repository**: `/Users/maropu/Repositories/pgvector-rx`
- **Branch**: master (work directly on master for now)
- **Technology**: Rust with pgrx 0.17.0, targeting PostgreSQL 18
- **Documentation**:
  - `DESIGNDOC.md`: Architecture and 7-phase roadmap
  - `references/pgvector/`: Original C implementation to port

### Goal
Pass all 28 HNSW-related tests from original pgvector:
- 4 SQL regression tests
- 24 Perl integration tests

### Open Major Issues
Check GitHub Issues for initial tasks. Issues are organized by phases:
- **Phase 1** (#1): Foundation and Data Types
- **Phase 2** (#2): Utility Functions and Distance Metrics
- **Phase 3** (#3): Index Building (Sequential)
- **Phase 4** (#4): Search and Scanning
- **Phase 5** (#5): Insertion and Updates
- **Phase 6** (#6): Vacuum and Maintenance
- **Phase 7** (#8): Testing and Optimization

## ITERATION WORKFLOW

Follow this workflow for EVERY iteration:

### 1. ORIENT YOURSELF (5 minutes)

```bash
# Check repository state
git status
git log --oneline -5

# Review open issues and select next task
gh issue list --label "critical" --state open
gh issue list --label "high" --state open
gh issue list --state open --limit 20

# Check current build status
cargo check
```

**Decision criteria for selecting an Issue**:
1. **Prioritize by labels**: `critical` > `high` > unlabeled
2. **Respect dependencies**: Check Issue descriptions for prerequisites
3. **Start with Phase 1** if nothing else is in progress
4. **Choose manageable scope**: Target 300-400 lines of code per iteration (not time-based)
5. **Prefer failing tests**: If tests exist, fix failures before adding features

### 2. ANALYZE THE TASK (10 minutes)

Before implementing, understand:

```bash
# Read relevant documentation
cat DESIGNDOC.md | grep -A 20 "Phase [N]"

# Study original C implementation (if porting)
ls -la references/pgvector/src/hnsw*.c
cat references/pgvector/src/[relevant-file].c

# Review reference Rust extensions for patterns
find references/paradedb -name "*.rs" | grep -E "(index|postgres)"
```

**Questions to answer**:
- What is the minimum viable implementation?
- What tests should verify this?
- What are the dependencies on other components?
- Are there any new tasks discovered?

### 3. IMPLEMENT THE SOLUTION (1-3 hours)

**Work in this order**:

1. **Create/modify types and structures** (if needed)
2. **Write tests FIRST** (TDD approach)
   - Add to `src/lib.rs` test module or create new test files
   - Use `#[pg_test]` for integration tests
3. **Implement minimal working code**
4. **Iterate until tests pass locally**

```bash
# Run tests frequently during development
cargo pgrx test pg18

# Check for common issues
cargo clippy --all-targets --all-features
cargo fmt
```

**Implementation guidelines**:
- Add `SAFETY:` comments for all unsafe blocks
- Document public functions with `///` comments
- Keep functions small and focused
- Match C behavior exactly (verify with original tests)
- **Monitor code size**: If implementation exceeds ~400 lines, STOP and consider splitting into sub-tasks

**If implementation grows too large (>400 lines)**:
1. **STOP coding immediately**
2. Analyze if the task can be split into smaller sub-tasks
3. If splittable:
   - **Discard current implementation** (git reset or stash)
   - **Create sub-task Issues on GitHub** with detailed scope (~300-400 lines each)
   - Use `gh issue create` to register each sub-task
   - Link sub-tasks to parent Issue
   - Pick ONE sub-task and restart implementation from scratch
4. If not splittable: Continue but document why in commit message

**Example work units** (good scope for one iteration):
- ✅ Implement `Vector` type with serialization/deserialization (~250 lines)
- ✅ Add L2 distance function with tests (~150 lines)
- ✅ Create HNSW meta page structure (~200 lines)
- ✅ Implement neighbor selection algorithm (~350 lines)
- ❌ Complete entire index building (~1500+ lines - too large, split into sub-tasks)

### 4. VERIFY LOCALLY (30 minutes)

**Before verification, check code size**:
```bash
# Count lines added (excluding tests, comments, blank lines)
git diff --cached | grep '^+' | grep -v '^+++' | grep -v '^+\s*$' | grep -v '^+\s*//' | wc -l
# Target: 300-400 lines. If >500 lines, consider splitting.
```

**Must pass before committing**:

```bash
# 1. Format code
cargo fmt

# 2. Run linter
cargo clippy --all-targets --all-features -- -D warnings

# 3. Build
cargo build --release

# 4. Run all tests
cargo pgrx test pg18

# 5. Verify extension loads
cargo pgrx run pg18
# In psql: CREATE EXTENSION pgvector_rx; -- should work
```

**If any step fails**: Fix immediately. Do NOT commit broken code.

### 5. COMMIT AND PUSH (10 minutes)

**Commit message format**:
```
[#IssueNumber] Brief description (imperative mood)

Detailed explanation:
- What was implemented
- Why this approach was chosen
- Any limitations or TODOs

Refs #IssueNumber

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

**Example**:
```
[#1] Implement Vector type with FromDatum/IntoDatum

Add Vector type supporting float32 arrays up to 2000 dimensions.
Implements PostgreSQL Datum conversion for seamless integration.
Includes basic validation and dimension checks.

Refs #1

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

```bash
# Stage changes
git add -A

# Commit with proper message
git commit -m "[#N] Your commit message

Details here...

Refs #N

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"

# Push (CI will run automatically for code changes)
git push origin master
```

### 6. VERIFY CI (15 minutes)

**Note**: CI verification is required only for code changes (Rust files, Cargo.toml, SQL files, etc.). For documentation-only changes (*.md files, README updates), you can skip this step and proceed to step 7.

```bash
# For code changes only - wait for CI to start
sleep 30

# Check CI status
gh run list --limit 1

# Watch CI progress (optional)
gh run watch

# If CI fails: investigate immediately
gh run view --log-failed
```

**When to verify CI**:
- ✅ Code changes (*.rs, Cargo.toml, *.sql, etc.): Must verify CI passes
- ❌ Documentation only (*.md, README, etc.): CI verification optional

**If CI fails**:
1. Pull the logs: `gh run view --log-failed`
2. Reproduce locally: `cargo pgrx test pg18`
3. Fix the issue
4. Commit fix with message: `[#N] Fix CI: [description]`
5. Verify CI again

**Do not proceed to next step until CI is green (for code changes only).**

### 7. UPDATE ISSUE TRACKING (10 minutes)

**Check if Issue is complete**:
```bash
# Review Issue checklist
gh issue view [NUMBER]
```

**If Issue is complete**:
```bash
# Close with comment
gh issue close [NUMBER] -c "Completed in commit $(git rev-parse --short HEAD). All acceptance criteria met."
```

**If Issue has more work**:
```bash
# Add progress comment
gh issue comment [NUMBER] -b "Completed: [what you did]. Remaining: [what's left]"
```

**Create new Issues for discovered work**:
```bash
# For each new task discovered during implementation
gh issue create \
  --title "[Brief description]" \
  --body "## Context

[Why this is needed]

## Tasks
- [ ] Task 1
- [ ] Task 2

## Acceptance Criteria
- Criterion 1
- Criterion 2

## Dependencies
Depends on #[IssueNumber] (if applicable)" \
  --label "phase-N" \
  --label "[priority: critical/high/low]"
```

**Examples of new Issues to create**:
- Missing utility functions discovered during implementation
- Performance optimizations needed
- Additional test coverage required
- Documentation updates needed
- Edge cases found in original C code

### 8. DOCUMENT AND EXIT (5 minutes)

**Update documentation if needed**:
- Update `DESIGNDOC.md` if architecture changed
- Add comments to complex algorithms

**Final check**:
```bash
# Verify everything is clean
git status  # Should show "working tree clean"
cargo pgrx test pg18  # Should pass (for code changes)
gh run list --limit 1  # Should show "completed" and "success" (for code changes)
```

**Exit cleanly**: Your iteration is complete. The loop will start again.

## DECISION MAKING GUIDELINES

### When to Split Large Tasks

**CRITICAL**: Split tasks based on code size, not time estimates.

**Split if**:
- Initial analysis suggests >500 lines of implementation
- During implementation, you've written >400 lines and task is incomplete
- Task involves multiple independent components

**How to split during implementation**:
1. **Stop coding** when you reach ~400 lines
2. Review what you've implemented and what remains
3. If remaining work is substantial:
   - **Discard current work**: `git reset --hard` or `git stash`
   - **Create sub-task Issues on GitHub** (each targeting 300-400 lines):
     ```bash
     gh issue create \
       --title "[Parent #N] Sub-task: [description]" \
       --body "## Context\nSplit from #N due to code size (>400 lines)\n\n## Estimated Lines\n~300-400 lines\n\n## Tasks\n- [ ] Task details\n\n## Parent Issue\nPart of #N" \
       --label "phase-X" \
       --label "subtask"
     ```
   - Document split rationale in parent Issue comments
   - Pick smallest sub-task and restart fresh

**Example**: Issue #3 "Index Building" (estimated 1500+ lines)
1. Split into sub-issues (~300-400 lines each):
   - "Implement HNSW meta page structure" (~200 lines)
   - "Implement graph node allocation" (~300 lines)
   - "Implement sequential insert during build" (~400 lines)
   - "Implement neighbor selection algorithm" (~350 lines)
   - "Implement ambuild callback" (~250 lines)
2. Add labels, dependencies, and line estimates
3. Work on smallest sub-issue first

### When to Add Tests

**Always add tests when**:
- Implementing a new function
- Fixing a bug
- Adding a new feature

**Test types**:
1. **Unit tests**: In `#[cfg(test)] mod tests`
2. **Integration tests**: In `#[pg_test]` blocks
3. **Regression tests**: Port from `references/pgvector/test/`

### When to Optimize

**Don't optimize prematurely**. Focus on:
1. ✅ Correctness first (tests pass)
2. ✅ Functionality complete (all features work)
3. ✅ Then optimize (Phase 7)

### When to Ask for Help

You're autonomous, but if you encounter:
- Ambiguous requirements → Check DESIGNDOC.md, create Issue for clarification
- Unclear C code → Study references, add comments
- pgrx API confusion → Check `references/paradedb` examples
- Test failures → Debug systematically, check CI logs

**Never leave the codebase broken**. If stuck, revert changes and create an Issue describing the blocker.

## COMMON PITFALLS TO AVOID

### ❌ Don't Do This
1. **Committing broken code**: Always verify tests pass
2. **Ignoring CI failures**: Must be green before moving on
3. **Working on multiple Issues**: One at a time
4. **Skipping tests**: Tests are mandatory
5. **Large monolithic commits**: >400 lines requires split consideration
6. **Continuing when code grows too large**: Stop at ~400 lines and split
7. **Forgetting Co-authored-by trailer**: Always include it
8. **Not creating Issues for new work**: Track everything
9. **Copying C code directly**: Translate to idiomatic Rust

### ✅ Do This
1. **Test-driven development**: Write tests first
2. **Small commits**: Target 300-400 lines of implementation code
3. **Monitor code size**: Check line count regularly during implementation
4. **Split when too large**:
   - Discard implementation if >400 lines
   - Create GitHub Issues for each sub-task using `gh issue create`
   - Link sub-tasks to parent Issue
5. **Clear commit messages**: Reference Issue numbers
6. **Update Issues**: Comment on progress and line counts
7. **Follow coding standards**: Read `.copilot-instructions.md`
8. **Verify CI**: Always check GitHub Actions
9. **Document decisions**: Explain non-obvious code

## EXAMPLE ITERATION

Here's what a successful iteration looks like:

```bash
# 1. ORIENT
gh issue list --label critical
# Select Issue #1 (Phase 1: Vector type)

# 2. ANALYZE
cat references/pgvector/src/vector.c
# Understand: Need FromDatum/IntoDatum for Vector type

# 3. IMPLEMENT
# Create src/types/vector.rs
# Write tests in src/types/vector.rs
# Implement Vector struct with FromDatum/IntoDatum
cargo pgrx test pg18  # Passes!

# 4. VERIFY
cargo fmt
cargo clippy --all-targets --all-features
cargo build --release
cargo pgrx test pg18
# All pass!

# 5. COMMIT
git add src/types/
git commit -m "[#1] Implement Vector type with FromDatum/IntoDatum

Add Vector type supporting float32 arrays up to 2000 dimensions.
Implements PostgreSQL Datum conversion for seamless integration.

- FromDatum: Deserialize from PostgreSQL internal format
- IntoDatum: Serialize to PostgreSQL internal format
- Validation: Check dimensions <= HNSW_MAX_DIM

Refs #1

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git push origin master

# 6. VERIFY CI (code changes only)
gh run watch
# CI passes!

# 7. UPDATE ISSUES
gh issue comment 1 -b "✅ Vector type implemented and tested. Next: halfvec type."

# 8. DOCUMENT
# (No doc updates needed for this internal type)

# EXIT - Iteration complete!
```

## QUICK REFERENCE

### Essential Commands
```bash
# Orientation
gh issue list --state open
git log --oneline -10

# Development
cargo pgrx test pg18
cargo clippy --all-targets --all-features
cargo fmt

# Git
git status
git add -A
git commit -m "..."
git push origin master

# CI
gh run list --limit 5
gh run watch
gh run view --log-failed

# Issues
gh issue view [N]
gh issue comment [N] -b "..."
gh issue close [N]
gh issue create --title "..." --body "..." --label "..."
```

### File Locations
- **Architecture**: `DESIGNDOC.md`
- **Coding standards**: `.copilot-instructions.md`
- **Original C code**: `references/pgvector/src/hnsw*.{c,h}`
- **Rust examples**: `references/paradedb/`, `references/plrust/`, `references/postgresml/`
- **Tests**: `src/lib.rs` (unit/integration), `tests/` (regression)
- **CI config**: `.github/workflows/ci.yml`

### Priority Order
1. Fix broken builds/tests (CRITICAL)
2. Issues labeled "critical"
3. Issues labeled "high"
4. Phase 1 → Phase 2 → ... → Phase 7
5. Add test coverage
6. Documentation improvements

## YOUR GOAL FOR THIS ITERATION

**Pick ONE Issue. Implement ONE feature. Add tests. Verify CI. Update tracking. Exit.**

Start now. Good luck! 🚀

---

*This prompt is designed for autonomous operation in an infinite loop. Each iteration should take 2-4 hours and leave the codebase in a clean, tested, CI-passing state.*
