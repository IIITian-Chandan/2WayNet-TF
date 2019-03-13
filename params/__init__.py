import sys
__all__ = ["COCO", "FLICKR8k", "FLICKR30k", "MNIST", "XRMB"]


g_dataset_name_to_class = None
def _build_class_map():
    global g_dataset_name_to_class
    if g_dataset_name_to_class:
        return
    g_dataset_name_to_class = {}
    for module_name in __all__:
        # Our convention for Params class name: <dataset_name>_Params
        class_name = module_name + "_Params"
        lname = module_name.lower()
        cls = getattr(sys.modules[module_name], class_name)
        if cls.name.lower() != lname:
            raise ValueError("Params class {} should have a 'name' field".format(class_name))
        g_dataset_name_to_class[lname] = cls

def get_params_class_for_dataset_name(name):
    _build_class_map()
    return g_dataset_name_to_class[name.lower()]