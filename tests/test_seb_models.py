import seb


def test_all_model_are_unique():
    models = seb.get_all_models()

    # model name should be unique
    model_names = [model.meta.name for model in models]
    assert len(model_names) == len(set(model_names))

    # cache path should be unique
    cache_paths = [model.meta.get_path_name() for model in models]
    assert len(cache_paths) == len(set(cache_paths))
