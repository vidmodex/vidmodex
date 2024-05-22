from .Tgan import VideoGenerator as Tgan
from .ref_gan import ReferenceGenerator as RefGan
from .condTgan import VideoGenerator as CondTgan
from .shap_discriminator import ConditionalShapDiscriminator3D as ShapDiscriminator3D, ConditionalShapDiscriminator2D as ShapDiscriminator2D
from .imgGan import Generator28, Generator32, Generator64, Generator128, Generator224, Generator256, CondGenerator28, CondGenerator32, CondGenerator64, CondGenerator128, CondGenerator224, CondGenerator256

class GeneratorFactory:
    _generator_classes = {}

    @classmethod
    def get(cls, generator_type:str):
        try:
            return cls._generator_classes[generator_type]
        except KeyError:
            raise ValueError(f"unknown product type : {generator_type}")

    @classmethod
    def register(cls, generator_type:str):
        def inner_wrapper(wrapped_class):
            cls._generator_classes[generator_type] = wrapped_class
            return wrapped_class
        return inner_wrapper

GeneratorFactory.register("tgan")(Tgan)
GeneratorFactory.register("refgan")(RefGan)
GeneratorFactory.register("condtgan")(CondTgan)
GeneratorFactory.register("shapdiscriminator3D")(ShapDiscriminator3D)
GeneratorFactory.register("shapdiscriminator2D")(ShapDiscriminator2D)
GeneratorFactory.register("generator28")(Generator28)
GeneratorFactory.register("generator32")(Generator32)
GeneratorFactory.register("generator64")(Generator64)
GeneratorFactory.register("generator128")(Generator128)
GeneratorFactory.register("generator224")(Generator224)
GeneratorFactory.register("generator256")(Generator256)
GeneratorFactory.register("condgenerator28")(CondGenerator28)
GeneratorFactory.register("condgenerator32")(CondGenerator32)
GeneratorFactory.register("condgenerator64")(CondGenerator64)
GeneratorFactory.register("condgenerator128")(CondGenerator128)
GeneratorFactory.register("condgenerator224")(CondGenerator224)
GeneratorFactory.register("condgenerator256")(CondGenerator256)


    