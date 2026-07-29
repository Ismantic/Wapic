# Publishing to PyPI

Wapic publishes two distributions:

- `wapic-cws-model`: a platform-independent wheel containing the pinned model.
- `wapic`: native wheels that depend on that model package.

Users install both with a single `pip install wapic`.

## One-time setup

Create GitHub Trusted Publishers for both project names on
[TestPyPI](https://test.pypi.org/manage/account/publishing/) and
[PyPI](https://pypi.org/manage/account/publishing/). Use:

| Field | Main package | Model package |
| --- | --- | --- |
| TestPyPI environment | `testpypi` | `testpypi-model` |
| PyPI environment | `pypi` | `pypi-model` |

Add one pending publisher for `wapic` and another for `wapic-cws-model` on
each index. For all four entries, use owner `Ismantic`, repository `Wapic`,
and workflow `publish.yml`. No API token or repository secret is required.

In the GitHub repository, create environments named `testpypi`,
`testpypi-model`, `pypi`, and `pypi-model`. Requiring approval for the two
production environments is recommended.

## Release

1. Keep the versions in the root and model `pyproject.toml` files synchronized.
   Also update the model dependency and `wapic_model.__version__`.
2. Run the `Publish Python packages` workflow manually. This publishes to
   TestPyPI.
3. Verify installation in a clean environment:

   ```bash
   python -m pip install \
     --index-url https://test.pypi.org/simple \
     --extra-index-url https://pypi.org/simple wapic
   python -c "import wapic; print(wapic.Segmenter().segment('中华人民共和国'))"
   ```

4. Create and publish a GitHub release such as `v0.1.0`. The release event
   builds fresh artifacts and publishes the model first, then Wapic, to PyPI.

PyPI release files are immutable. Increment the version before retrying after
a partial or failed upload.
