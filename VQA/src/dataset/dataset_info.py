import os, json

def get_RAG_classes_dict(dataset, config):

    """
    Retrieves the number of unique head entities for the specified dataset.
    """
    root = config.data_path

    if dataset == "Slake":
        path = os.path.join(root, "Slake1.0/KG_Slake_Train.json")
        kg = json.load(open(path, 'r'))
        path_append = "Slake1.0/imgs"

    elif dataset == "PathVQA":
        path = os.path.join(root, "PathVQA/KG_PathVQA_Train.json")
        kg = json.load(open(path, 'r'))
        path_append = "PathVQA/images"
    
    elif dataset == "VQARAD":
        path = os.path.join(root, "VQARAD/KG_VQARAD_Train_map.json")
        kg = json.load(open(path, 'r'))
        path_append = "VQARAD/VQA_RAD_Image_Folder"

    RAG_classes_dict = {}
    for entity in kg:
        if entity['head_entity'].lower() not in RAG_classes_dict:
            RAG_classes_dict[entity['head_entity'].lower()] = []
        #check str in [str]

        image_path = os.path.join(root, path_append, entity['image'])
        if  image_path not in RAG_classes_dict[entity['head_entity'].lower()]:
            RAG_classes_dict[entity['head_entity'].lower()].append(image_path)

    num_classes = len(list(RAG_classes_dict.keys()))
    RAG_classes_dict["num_classes"] = num_classes

    return RAG_classes_dict
