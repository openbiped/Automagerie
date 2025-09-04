# Contributing to Automagerie

We want Automagerie to be a true community-driven effort that continuously
improves and grows over time for the benefit of the entire research community.
As such, we welcome contributions that:

- Fix issues with an existing model
- Improve the realism of a model (e.g. via
  [system identification](https://en.wikipedia.org/wiki/System_identification))
- Add an entirely new model

Note that Automagerie follows [Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## How to contribute

Whether you want to fix an issue with an existing model, improve it, or add a
completely new model, please get in touch with us first (ideally _before_
starting work if it's something major) by opening a new
[issue](https://github.com/rimim/Automagerie/issues).
Coordinating up front makes it much easier to avoid frustration later on.

Once we reach an agreement on the proposed change, please submit a
[pull request](https://github.com/rimim/Automagerie/pulls) (PR)
so that we can review your implementation.

## Unit Tests

Before submitting your PR, you can test your change locally by invoking pytest:

```bash
uv run test/model_test.py
```

## Changelog & Contributors

Please document your changes in the appropriate changelog:

- For updates that affect the general repository (e.g., CI, tooling, documentation, shared infrastructure), add an entry to the [global `CHANGELOG.md`](./CHANGELOG.md).
- For changes specific to a model, update the `CHANGELOG.md` in that model’s directory (e.g., `go_bdx/CHANGELOG.md`).

Make sure to also add your name to the [`CONTRIBUTORS.md`](./CONTRIBUTORS.md), keeping the list sorted alphabetically by first name.
