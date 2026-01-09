def skip_all_class_members(app, what, name, obj, skip, options):
    """
    Forces Sphinx to skip documenting members (methods/attributes)
    if they belong to the esapp.grid module.
    """
    # 1. Identify the module where the object is defined
    # We check the __module__ attribute of the object itself
    obj_module = getattr(obj, "__module__", "")

    # 2. Check if the object belongs to esapp.grid
    if obj_module == "esapp.grid" or "esapp.grid" in name:
        # 3. Skip if it's a member (method, attribute, etc.)
        # This allows 'class' and 'exception' but hides their contents
        if what in ("method", "attribute", "property", "data"):
            return True
            
    # Default: do not override Sphinx's decision
    return None

def setup(app):
    app.connect('autodoc-skip-member', skip_all_class_members)
