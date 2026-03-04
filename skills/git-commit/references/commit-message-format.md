# Commit Message Format

Use this format by default:

```text
<type>(<optional-scope>): <summary>

<optional-body>
```

## Types

- `feat`: add new functionality
- `fix`: fix a bug or regression
- `refactor`: improve structure without behavior change
- `perf`: improve performance
- `docs`: documentation-only changes
- `test`: add or change tests
- `chore`: tooling, dependencies, or maintenance work
- `build`: build system or packaging changes
- `ci`: CI pipeline changes

## Summary Line

- Keep under 72 characters when practical.
- Use imperative mood, for example `add`, `fix`, `remove`, `rename`.
- Mention user-visible impact when relevant.

## Scope

Use scope when it adds clarity, for example:

- `fix(image-chunking): handle missing clip_processor`
- `feat(search): add sparse reranking`

## Body

Add a body when change intent is not obvious from the diff.
Use short lines that explain why the change was needed and any tradeoffs.
