class Model:
    _model_classes = {}

    @classmethod
    def get(cls, model_type:str):
        try:
            return cls._model_classes[model_type]
        except KeyError:
            raise ValueError(f"unknown product type : {model_type}")

    @classmethod
    def register(cls, model_type:str):
        def inner_wrapper(wrapped_class):
            cls._model_classes[model_type] = wrapped_class
            return wrapped_class
        return inner_wrapper
