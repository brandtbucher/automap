import sys

import invoke


ARTIFACTS = ["*.egg-info", "*.so", "MANIFEST", "build", "dist"]


@invoke.task
def clean(context):
    context.run("{} setup.py develop --uninstall".format(sys.executable))
    for artifact in sorted(ARTIFACTS):
        context.run("rm -rf {artifact}".format(artifact=artifact))
    if (3, 6) <= sys.version_info:
        context.run("black .")


@invoke.task(clean)
def build(context):
    context.run(
        "{} setup.py develop sdist".format(sys.executable),
        env={"CPPFLAGS": "-Werror -Wno-deprecated-declarations"},
        replace_env=False,
    )
    if (3, 6) <= sys.version_info:
        context.run("twine check dist/*")


@invoke.task(build)
def test(context):
    # context.run("mypy --strict .")
    context.run("pytest")


if (3, 6) <= sys.version_info:

    @invoke.task(test)
    def release(context):
        context.run("twine upload dist/*")
