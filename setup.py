from setuptools import setup

setup(
    use_scm_version={"version_scheme": "pre-release", "write_to": "pKAI/_version.py",},
    setup_requires=["setuptools_scm"],
)
