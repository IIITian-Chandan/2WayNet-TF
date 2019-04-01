import sys
import importlib
__all__ = ["COCO", "FLICKR8k", "FLICKR30k", "MNIST", "XRMB"]


g_dataset_name_to_class = None
def _build_class_map(name):
    global g_dataset_name_to_class
    if g_dataset_name_to_class and name in g_dataset_name_to_class:
        return
    if g_dataset_name_to_class is None:
        g_dataset_name_to_class = {}
    if name in __all__:
        module_name = name
        # Our convention for Params class name: <dataset_name>_Params
        class_name = module_name + "_Params"
        lname = module_name.lower()

        importlib.import_module("params." + module_name)
        cls = getattr(sys.modules["params." + module_name], class_name)
        if cls.name.lower() != lname:
            raise ValueError("Params class {} should have a 'name' field".format(class_name))
        g_dataset_name_to_class[lname] = cls

def get_params_class_for_dataset_name(name):
    _build_class_map(name)
    return g_dataset_name_to_class[name.lower()]

