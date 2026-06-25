def test_package_imports_and_has_version():
    import forgery_pipeline
    assert isinstance(forgery_pipeline.__version__, str)
    assert forgery_pipeline.__version__
