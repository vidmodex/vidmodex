from vidmodex.models import Model
from .movinet import MoViNet, MoViNetA5, MoViNetA2
from .vivit import ViViT
from .swint import SwinT

Model.register("movinet")(MoViNet)
Model.register("movineta5")(MoViNetA5)
Model.register("movineta2")(MoViNetA2)
Model.register("vivit")(ViViT)
Model.register("swint")(SwinT)
