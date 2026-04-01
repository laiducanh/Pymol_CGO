from pymol import cmd

def delete_all(obj_types=['object:cgo']):
    # Get all PyMOL objects
    all_objects = cmd.get_names('all')

    # Filter for objects of predefined type
    for t in obj_types:
        cgo_objects = [obj for obj in all_objects if cmd.get_type(obj) == str(t)]
        for obj in cgo_objects:
            cmd.delete(obj)