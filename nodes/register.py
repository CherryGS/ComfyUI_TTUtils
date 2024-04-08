from .core import BaseNode


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

ClassMapping = dict[str, type[BaseNode]]
NameMapping = dict[str, str]


def update(cls: ClassMapping, name: NameMapping):
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    NODE_CLASS_MAPPINGS.update(cls)
    NODE_DISPLAY_NAME_MAPPINGS.update(name)


def ezupd(
    src: list[
        tuple[type[BaseNode]]
        | tuple[type[BaseNode], str]
        | tuple[type[BaseNode], str, str]
    ]
):
    """第一个 `str` 是显示在 comfyui 里的名称 , 第三个是用来做 key 映射的名称 , 如果没有会用 Node 的名称代替"""
    cls: ClassMapping = {}
    name: NameMapping = {}
    for i in src:
        j = i
        if len(j) == 1:
            j = (*j, i[0].__name__)
        if len(j) == 2:
            j = (*j, i[0].__name__)
        cls.update({j[2]: j[0]})
        name.update({j[2]: j[1]})
    update(cls, name)
